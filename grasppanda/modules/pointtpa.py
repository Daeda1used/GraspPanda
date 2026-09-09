"""Author PointTPA branches with native PTv3 serialization and residual placement."""
import hashlib
import importlib.util
import sys
import types


def native_module():
    from ..config import ROOT
    name = '_grasppanda_pointtpa'
    if name not in sys.modules:
        path = ROOT/'environments/sources/cv/pointtpa/pointcept/models/peft/pointtpa.py'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != '1964f6e7c7a476cfa1bc6fbfe1a09dd8b73a1afa80ebd7e1bcc82f533bde49ca':
            raise ValueError('PointTPA source is missing or differs from its pin; run ./panda install')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def forward(self, point):
    """Native PTv3 block with the PointTPA branch parallel to its MLP."""
    shortcut = point.feat
    point = self.cpe(point)
    point.feat = shortcut + point.feat
    shortcut = point.feat
    if self.pre_norm:
        point = self.norm1(point)
    point = self.drop_path(self.ls1(self.attn(point)))
    point.feat = shortcut + point.feat
    if not self.pre_norm:
        point = self.norm1(point)
    shortcut = point.feat
    shortcut = self.adaptive_module(point) + shortcut
    if self.pre_norm:
        point = self.norm2(point)
    point = self.drop_path(self.ls2(self.mlp(point)))
    point.feat = shortcut + point.feat
    if not self.pre_norm:
        point = self.norm2(point)
    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
    return point


def attach(network, block_type, options, depths, max_level=4):
    from .pointtpa_options import resolve
    values = resolve(options, depths, max_level)
    encoder = [(name, module) for name, module in network.named_modules()
               if name.startswith('enc.') and isinstance(module, block_type)]
    if len(encoder) != sum(depths):
        raise ValueError('PointTPA encoder blocks differ from the selected architecture')
    author = native_module()
    for ordinal, index in enumerate(values['blocks']):
        name, block = encoder[index]
        stage = int(name.split('.')[1].removeprefix('enc'))
        layer = int(name.split('.')[2].removeprefix('block'))
        block.adaptive_module = author.DynamicParameterModule(
            in_channel=block.channels, mid=values['bottleneck_channels'][ordinal],
            stage=stage, layer=layer, K=values['experts'][ordinal],
            param=values['group_size'][ordinal], mode=values['group_mode'][ordinal],
            scale=values['scale'][ordinal], dy_down=values['dynamic_down'][ordinal],
            dy_up=values['dynamic_up'][ordinal], bias=True,
            order_idx=block.attn.order_index).to(next(block.parameters()))
        block.forward = types.MethodType(forward, block)
    return values


def load_base_state(network, state):
    """Require every original pretrained tensor; initialize only new adapter state."""
    current = network.state_dict()
    added = {name for name in current if '.adaptive_module.' in name}
    expected = set(current) - added
    if set(state) != expected:
        raise ValueError('PointTPA initialization must match every original pretrained encoder tensor')
    network.load_state_dict({**state, **{name:current[name] for name in added}}, strict=True)
