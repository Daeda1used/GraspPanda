"""GPU sampling with explicit input-row indices and scene isolation."""
import torch
from .sampling_options import normalize


def _check_points(points):
    if (not isinstance(points, torch.Tensor) or points.ndim != 3 or points.shape[-1] != 3
            or not points.is_cuda or points.dtype != torch.float32):
        raise ValueError('Network sampling requires CUDA float32 camera XYZ [B, N, 3]')
    batch, count, _ = points.shape
    if not batch or not count or batch * count > (2**31 - 1) // 3:
        raise ValueError('Sampling requires nonempty input within native coordinate index capacity')
    if not torch.isfinite(points).all(): raise ValueError('Sampling coordinates must be finite')
    return points.contiguous()


@torch.no_grad()
def density_weights(points, neighbors=20, quantile=.5):
    normalize({'type': 'pointsp_wrs', 'neighbors': neighbors, 'density_quantile': quantile})
    points = _check_points(points)
    from _grasppanda_pointops import knn_query
    batch, count, _ = points.shape
    neighbors = min(neighbors, count)
    if batch * count * neighbors > 2**31 - 1:
        raise ValueError('Sampling neighborhoods exceed native output index capacity')
    flat = points.reshape(-1, 3).contiguous()
    offsets = torch.arange(1, batch + 1, dtype=torch.int32, device=points.device) * count
    indices, _ = knn_query(neighbors, flat, offsets)
    indices = indices.long()
    lower = torch.arange(batch, device=points.device).repeat_interleave(count)[:, None] * count
    if ((indices < lower) | (indices >= lower + count)).any():
        raise ValueError('Native sampling KNN returned incomplete or cross-scene neighbors')
    squared = (flat[indices] - flat[:, None]).square().sum(-1).reshape(batch, count, -1)
    threshold = torch.quantile(squared.mean(-1), quantile, dim=-1, keepdim=True)
    weights = (squared <= threshold[..., None]).sum(-1).float()
    return weights / weights.sum(-1, keepdim=True)


@torch.no_grad()
def sample_indices(points, count, options='upstream', native=None, training=False):
    policy = normalize(options)
    if 'train' in policy: policy = policy['train' if training else 'eval']
    kind = policy['type']
    if kind == 'upstream':
        if native is None: raise ValueError('Upstream sampling requires the original sampler')
        return native(points, count)
    points = _check_points(points)
    batch, n, _ = points.shape
    if type(count) is not int or not 1 <= count <= n:
        raise ValueError('Sampling count must fit nonempty input rows')
    if kind == 'uniform':
        return torch.multinomial(torch.ones((batch, n), device=points.device), count, replacement=False)
    weights = (density_weights(points, policy.get('neighbors', 20), policy.get('density_quantile', .5))
               if kind in ('pointsp_wrs', 'pointsp_ffps') else None)
    if kind == 'pointsp_wrs': return torch.multinomial(weights, count, replacement=False)
    eligible = torch.ones((batch, n), dtype=torch.bool, device=points.device)
    if kind == 'pointsp_ffps':
        keep = max(count, int(n * policy.get('keep_ratio', .95)))
        order = torch.argsort(weights, dim=1, descending=True, stable=True)
        eligible.zero_().scatter_(1, order[:, :keep], True)
    starts = (torch.multinomial(eligible.float(), 1).squeeze(1) if policy.get('start', 'first') == 'random'
              else eligible.long().argmax(1))
    try:
        import _grasppanda_sampling_cuda as ops
    except ImportError as error:
        raise ValueError('Sampling operator is missing; rerun ./panda install') from error
    if ops.api_version != 1: raise ValueError('Sampling operator version differs; rerun ./panda install')
    return ops.sample(points, eligible, starts, count)


class NetworkSampler(torch.nn.Module):
    def __init__(self, policy, native):
        super().__init__()
        self.policy, self.native = policy, native

    def forward(self, points, count):
        return sample_indices(points, count, self.policy, self.native, self.training)


def configure_sampling(backbone, options):
    """Bind policies to this model instance; original module globals are untouched."""
    if 'seed_sampling' in options: backbone.seed_sampling = normalize(options['seed_sampling'])
    if 'stage_sampling' in options:
        blocks = [m for m in backbone.encoder.modules() if hasattr(m, 'sample_fn')]
        if len(blocks) != 4: raise ValueError('Expected four native hierarchy sampling boundaries')
        for block, value in zip(blocks, options['stage_sampling']):
            policy = normalize(value)
            if policy.get('type') != 'upstream':
                block.sample_fn = NetworkSampler(policy, block.sample_fn)
