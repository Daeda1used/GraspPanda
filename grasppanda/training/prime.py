"""PRIME photometric augmentation for aligned RGB-D observations.

The chunked color calculation adapts the Apache-2.0 PRIME implementation.
Source, modifications and license: docs/THIRD_PARTY.md.
"""
import ast
from functools import lru_cache
import hashlib
from types import SimpleNamespace


def validate(options):
    from .options import finite
    if not isinstance(options, dict): raise ValueError('augmentation.prime must be a mapping')
    bounds = {'probability': (0, 1), 'color_temperature': (0, .1),
              'filter_sigma': (0, 4)}
    integers = {'mixture_width': (1, 8), 'max_depth': (1, 8), 'color_cut': (1, 500),
                'color_bandwidth': (1, 500), 'filter_kernel': (3, 15)}
    allowed = {'primitives', 'mixture_depth', 'stochastic', *bounds, *integers}
    if set(options) - allowed: raise ValueError('Unknown PRIME options: ' + str(sorted(set(options)-allowed)))
    primitives = options.get('primitives', ['color', 'filter'])
    if (not isinstance(primitives, list) or not primitives or
        any(not isinstance(p, str) or p not in ('color', 'filter') for p in primitives) or
        len(set(primitives)) != len(primitives)):
        raise ValueError('prime.primitives must be a nonempty unique list of color and/or filter')
    for name, (low, high) in bounds.items():
        if name in options and not finite(options[name], low, high): raise ValueError('Invalid prime.' + name)
    for name, (low, high) in integers.items():
        if name in options and (type(options[name]) is not int or not low <= options[name] <= high):
            raise ValueError('Invalid prime.' + name)
    if options.get('filter_kernel', 3) % 2 != 1: raise ValueError('prime.filter_kernel must be odd')
    depth = options.get('mixture_depth', -1)
    if type(depth) is not int or (depth != -1 and not 1 <= depth <= options.get('max_depth', 3)):
        raise ValueError('prime.mixture_depth must be -1 or an integer from 1 to max_depth')
    if type(options.get('stochastic', True)) is not bool: raise ValueError('prime.stochastic must be boolean')


@lru_cache(maxsize=1)
def native():
    from ..config import ROOT
    namespace = {}
    sources = (
        ('color_jitter.py', '071b5f7da55d52766c1acba7afdcc8213937dff7cdec50d8e84c7b3436aca195'),
        ('rand_filter.py', 'd4c60cd6b628ac96a16a9d7dff54de856c0efb9c72bda529d6a0d68914b91b8d'),
        ('prime.py', 'd1efedb76bce77462008518b61f67ef6cd046c0080293c875bc6f99b2330bf6f'))
    for filename, digest in sources:
        path = ROOT/'environments/sources/cv/prime/utils'/filename
        if not path.is_file(): raise ValueError('PRIME sources are missing; run ./panda install')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest: raise ValueError('PRIME source differs from its pin; restore it and reinstall')
        tree = ast.parse(content, filename=str(path))
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
        fixes = 0
        for node in ast.walk(tree):
            if filename == 'rand_filter.py' and isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'center':
                # A zero-noise odd filter must be the identity at the same pixel.
                node.value = ast.parse('self.kernel_size // 2', mode='eval').body
                fixes += 1
            if filename == 'prime.py' and isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'depth_mask.data[:, :, 0, 0, 0]':
                # The release randomly shortens even explicitly fixed chains.
                node.value = ast.parse('self.depth_combos[depth_idx] if self.mixture_depth < 0 else torch.ones_like(self.depth_combos[depth_idx])', mode='eval').body
                fixes += 1
        if filename != 'color_jitter.py' and fixes != 1: raise ValueError('PRIME adaptation no longer matches the pinned source')
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return SimpleNamespace(**{k: namespace[k] for k in ('RandomSmoothColor', 'RandomFilter', 'PRIMEAugModule', 'GeneralizedPRIMEModule')})


def chunked_color(self, img, cut, T, freq_bandwidth=None):
    """Retain author coefficient draws while bounding full-resolution memory."""
    import math
    import torch
    from opt_einsum import contract
    b, c, h, w = img.shape
    colors = img.reshape(b, c, -1)
    if freq_bandwidth is not None:
        start = torch.randint(1, cut + 1, (1,)).item()
        k = torch.arange(start, min(start + freq_bandwidth, cut + 1), device=img.device)
    else:
        k = torch.arange(1, cut + 1, device=img.device)
    coeff = torch.randn((b, c, len(k)), device=img.device) * torch.sqrt(torch.tensor(T))
    result = torch.empty_like(colors)
    # At most ~32 MiB for the frequency tensor, independent of image dimensions.
    chunk = max(1, (8 * 1024 * 1024) // (b * c * len(k)))
    for start in range(0, h * w, chunk):
        value = colors[..., start:start+chunk]
        frequencies = torch.sin(value[..., None] * k[None, None, None, :] * math.pi)
        result[..., start:start+chunk] = (contract('bcf,bcnf->bcn', coeff, frequencies) + value).clamp(0, 1)
    return result.reshape(b, c, h, w)


def build(options):
    """Build CPU primitives lazily inside the dataset worker that uses them."""
    from types import MethodType
    import torch
    validate(options)
    author = native()
    primitives = []
    for name in options.get('primitives', ['color', 'filter']):
        if name == 'color':
            operation = author.RandomSmoothColor(options.get('color_cut', 500),
                options.get('color_temperature', .05), options.get('color_bandwidth', 20),
                stochastic=options.get('stochastic', True))
            operation.random_smooth_color = MethodType(chunked_color, operation)
        else:
            operation = author.RandomFilter(options.get('filter_kernel', 3),
                options.get('filter_sigma', 4.), stochastic=options.get('stochastic', True))
        primitives.append(operation)
    return author.GeneralizedPRIMEModule(torch.nn.Identity(), author.PRIMEAugModule(primitives),
        mixture_width=options.get('mixture_width', 3), mixture_depth=options.get('mixture_depth', -1),
        max_depth=options.get('max_depth', 3), no_jsd=True)


class PrimePhotometric:
    """Picklable full-resolution PIL transform; no CUDA state in data workers."""
    def __init__(self, options):
        from copy import deepcopy
        validate(options)
        self.options = deepcopy(options)

    def __call__(self, image):
        import torch
        from torchvision.transforms import functional as F
        probability = self.options.get('probability', 1.)
        if probability == 0 or (probability < 1 and torch.rand(()).item() >= probability): return image
        if image.mode != 'RGB': raise ValueError('PRIME photometric augmentation requires an RGB image')
        value = F.pil_to_tensor(image).float().div_(255).unsqueeze(0)
        value = build(self.options)(value).squeeze(0)
        return F.to_pil_image(value.mul(255).round().clamp(0, 255).to(torch.uint8))
