"""Pretrained point encoders with native transforms and reversible grasp mappings."""
import copy
import importlib.util
import json
import sys

import numpy as np
import torch
from torch import nn


def native_module(family):
    from ..config import ROOT
    if family not in ('utonia', 'concerto'):
        raise ValueError('Unknown point foundation encoder')
    name = '_grasppanda_' + family
    if name not in sys.modules:
        path = ROOT / 'environments/sources/cv' / family / family / '__init__.py'
        if not path.is_file():
            raise ValueError(f'{family} source is missing; run ./panda install')
        spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            for key in list(sys.modules):
                if key == name or key.startswith(name + '.'):
                    sys.modules.pop(key, None)
            raise
    return sys.modules[name]


class FoundationFeatures(nn.Module):
    def __init__(self, family, out_channels, variant='base', pretrained=True,
                 trainable_blocks=(0, 0, 0, 0, 0), train_embedding=False,
                 feature_levels=(0, 1, 2, 3, 4), input_scale=None, grid_size=None,
                 enc_patch_size=None, drop_path=None, attn_drop=None,
                 proj_drop=None, shuffle_orders=True, projection_norm='none', adaptation=None):
        super().__init__()
        from ..weights import component_records
        self.weight_id = 'utonia' if family == 'utonia' else 'concerto_' + variant
        record = component_records()[self.weight_id]
        self.family = family
        self.pretrained_requested = pretrained
        self.native_config = copy.deepcopy(record['model_config'])
        config = copy.deepcopy(self.native_config)
        self.feature_levels = tuple(feature_levels)
        if not self.feature_levels or tuple(sorted(set(self.feature_levels))) != self.feature_levels or not set(self.feature_levels) <= set(range(5)):
            raise ValueError('feature_levels must be distinct increasing stage indices from 0 to 4')
        if len(trainable_blocks) != 5 or any(type(n) is not int or not 0 <= n <= depth
                                           for n, depth in zip(trainable_blocks, config['enc_depths'])):
            raise ValueError('trainable_blocks must give five counts within the selected encoder depths')
        self.trainable_blocks = tuple(trainable_blocks)
        self.train_embedding = train_embedding
        self.shuffle_orders = shuffle_orders
        config['shuffle_orders'] = False
        for key, value in dict(enc_patch_size=enc_patch_size, drop_path=drop_path,
                               attn_drop=attn_drop, proj_drop=proj_drop).items():
            if value is not None: config[key] = value
        source = native_module(family)
        self.network = source.model.PointTransformerV3(**config)
        self.network.requires_grad_(False)
        for index, count in enumerate(self.trainable_blocks):
            if not count: continue
            stage = getattr(self.network.enc, f'enc{index}')
            for name, module in stage.named_children():
                if name == 'down' or (name.startswith('block') and int(name[5:]) >= config['enc_depths'][index] - count):
                    module.requires_grad_(True)
        if train_embedding:
            self.network.embedding.requires_grad_(True)
            # The grasp input has no masked-pretraining token positions.
            if self.network.embedding.mask_token is not None:
                self.network.embedding.mask_token.requires_grad_(False)
        self.adaptation = None
        if adaptation is not None:
            from .pointtpa import attach
            self.adaptation = attach(self.network, source.model.Block, adaptation,
                                     config['enc_depths'], max(self.feature_levels))
        self.projection = nn.Linear(sum(config['enc_channels'][i] for i in self.feature_levels), out_channels)
        if projection_norm not in ('none', 'layer'):
            raise ValueError('projection_norm must be none or layer')
        self.projection_norm = nn.LayerNorm(out_channels) if projection_norm == 'layer' else nn.Identity()
        self.input_scale = input_scale if input_scale is not None else (4. if family == 'utonia' else 1.)
        self.grid_size = grid_size if grid_size is not None else (.01 if family == 'utonia' else .02)
        self.transform = source.transform.Compose([
            *([dict(type='RandomScale', scale=[self.input_scale, self.input_scale])]
              if family == 'utonia' or self.input_scale != 1. else []),
            dict(type='CenterShift', apply_z=True),
            dict(type='GridSample', grid_size=self.grid_size, hash_type='fnv', mode='train',
                 return_grid_coord=True, return_inverse=True),
            dict(type='NormalizeColor'), dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'inverse'),
                 feat_keys=('coord', 'color', 'normal'))])
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not hasattr(self, 'network'): return self
        self.network.eval()
        self.network.shuffle_orders = mode and self.shuffle_orders and (any(self.trainable_blocks) or self.adaptation is not None)
        native = native_module(self.family)
        for module in self.network.modules():
            if isinstance(module, native.model.GridPooling):
                module.shuffle_orders = self.network.shuffle_orders
        for index, count in enumerate(self.trainable_blocks):
            if not count: continue
            stage = getattr(self.network.enc, f'enc{index}')
            for name, module in stage.named_children():
                if name == 'down' or (name.startswith('block') and int(name[5:]) >= self.native_config['enc_depths'][index] - count):
                    module.train(mode)
        if self.train_embedding: self.network.embedding.train(mode)
        if self.adaptation is not None:
            for module in self.network.modules():
                if hasattr(module, 'adaptive_module'): module.adaptive_module.train(mode)
        return self

    def initialize_pretrained(self):
        if not self.pretrained_requested: return None
        from ..weights import component_records, fetch_component
        record = component_records()[self.weight_id]
        checkpoint = torch.load(fetch_component(self.weight_id), map_location='cpu', weights_only=True)
        if json.loads(json.dumps(checkpoint['config'])) != self.native_config:
            raise ValueError('Pretrained point encoder architecture differs from its registered configuration')
        if self.adaptation is None:
            self.network.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            from .pointtpa import load_base_state
            load_base_state(self.network, checkpoint['state_dict'])
        return dict(id=self.weight_id, source=record['source'], sha256=record['sha256'],
                    trainable_blocks=list(self.trainable_blocks), train_embedding=self.train_embedding)

    def prepare(self, xyz, batch):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or batch.shape != (len(xyz),):
            raise ValueError('Foundation encoders require aligned XYZ [N,3] and batch IDs [N]')
        if not xyz.is_cuda or xyz.dtype != torch.float32 or batch.dtype != torch.long or batch.device != xyz.device:
            raise ValueError('Foundation encoders require CUDA float32 XYZ and int64 batch IDs on the same device')
        if not len(batch) or not torch.isfinite(xyz).all() or batch.min() < 0 or batch.max() >= len(batch) or (batch.bincount() == 0).any():
            raise ValueError('Foundation encoders require finite points and nonempty consecutive batch IDs')
        chunks, inverse, rows, offset = [], [], [], 0
        for scene in range(int(batch.max()) + 1):
            selected = (batch == scene).nonzero().flatten()
            points = xyz[selected].detach().cpu().numpy().copy()
            state = np.random.get_state()
            try:
                # Stable per-scene evaluation does not consume augmentation RNG.
                if not self.training: np.random.seed(0)
                data = self.transform(dict(coord=points, color=np.zeros_like(points), normal=np.zeros_like(points)))
            finally:
                if not self.training: np.random.set_state(state)
            if data['grid_coord'].max() >= 65535:
                raise ValueError('Foundation encoder lattice exceeds 16 bits; increase grid_size or reduce input_scale')
            chunks.append({key: data[key].to(xyz.device) for key in ('coord', 'grid_coord', 'feat')})
            chunks[-1]['batch'] = torch.full((len(data['coord']),), scene, dtype=torch.long, device=xyz.device)
            inverse.append(data['inverse'].to(xyz.device).long() + offset)
            rows.append(selected)
            offset += len(data['coord'])
        return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}, torch.cat(inverse), torch.cat(rows)

    def encode(self, data):
        point = self.network(data)
        level = 4
        if level not in self.feature_levels: point.feat = point.feat[:, :0]
        while 'pooling_parent' in point:
            parent, mapping = point.pop('pooling_parent'), point.pop('pooling_inverse')
            level -= 1
            local = parent.feat if level in self.feature_levels else parent.feat[:, :0]
            parent.feat = torch.cat([local, point.feat[mapping]], dim=-1)
            point = parent
        return point.feat

    def forward(self, xyz, batch):
        data, inverse, rows = self.prepare(xyz, batch)
        encoded = []
        encoder_grad = torch.is_grad_enabled() and any(p.requires_grad for p in self.network.parameters())
        sparse_type = native_module(self.family).model.spconv.SubMConv3d
        sparse_modes = [(module, module.training) for module in self.network.modules()
                        if isinstance(module, sparse_type)]
        try:
            # SpConv eval kernels detach features and omit backward indices shared
            # by later blocks. Frozen weights still need differentiable kernels
            # when gradients pass through them. Dropout and parameter flags stay
            # governed by the selected fine-tuning policy.
            for module, _ in sparse_modes:
                module.training = encoder_grad
            with torch.set_grad_enabled(encoder_grad):
                # Native Hilbert serialization derives its depth from the whole batch.
                # Independent encoding retains the author's single-scene windows.
                for scene in range(int(data['batch'].max()) + 1):
                    mask = data['batch'] == scene
                    item = {key: value[mask] for key, value in data.items()}
                    item['batch'] = torch.zeros_like(item['batch'])
                    encoded.append(self.encode(item))
        finally:
            for module, mode in sparse_modes:
                module.training = mode
        values = self.projection_norm(self.projection(torch.cat(encoded)))[inverse]
        return values.new_empty(values.shape).index_copy(0, rows, values)


class FoundationBackbone(nn.Module):
    def __init__(self, family, **options):
        super().__init__()
        self.features = FoundationFeatures(family, 256, **options)

    def initialize_pretrained(self):
        return self.features.initialize_pretrained()

    def forward(self, points, end_points=None):
        from pointnet2 import _ext
        from .sampling import sample_indices
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1024:
            raise ValueError('Foundation grasp encoders require camera XYZ [B,N,3] with N >= 1024')
        size, count, _ = points.shape
        points = points.contiguous()
        batch = torch.arange(size, device=points.device).repeat_interleave(count)
        dense = self.features(points.reshape(-1, 3), batch).reshape(size, count, 256)
        indices = sample_indices(points, 1024, getattr(self, 'seed_sampling', 'upstream'),
                                native=_ext.furthest_point_sampling, training=self.training)
        seeds = points.gather(1, indices.long()[..., None].expand(-1, -1, 3)).contiguous()
        features = dense.gather(1, indices.long()[..., None].expand(-1, -1, 256)).transpose(1, 2).contiguous()
        end_points = {} if end_points is None else end_points
        end_points.update(input_xyz=points, input_features=None, fp2_xyz=seeds,
                          fp2_features=features, fp2_inds=indices)
        return features, seeds, end_points


class SparseFoundationBackbone(nn.Module):
    def __init__(self, family, out_channels=512, voxel_size=.005, **options):
        super().__init__()
        self.voxel_size = voxel_size
        self.features = FoundationFeatures(family, out_channels, **options)

    def initialize_pretrained(self):
        return self.features.initialize_pretrained()

    def forward(self, sparse):
        import MinkowskiEngine as ME
        coords = sparse.C.to(device=sparse.F.device).long()
        xyz = coords[:, 1:].to(torch.float32) * self.voxel_size
        values = self.features(xyz, coords[:, 0])
        return ME.SparseTensor(values, coordinate_map_key=sparse.coordinate_map_key,
                              coordinate_manager=sparse.coordinate_manager)
