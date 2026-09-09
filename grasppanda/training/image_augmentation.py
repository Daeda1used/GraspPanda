"""Observation augmentation shared by the native RGB-D and local branches."""
from grasppanda.training.options import finite


LIMITS = {'brightness': 1., 'contrast': 1., 'saturation': 1., 'hue': .5,
          'grayscale_probability': 1., 'blur_probability': 1.,
          'blur_sigma_min': 10., 'blur_sigma_max': 10.,
          'depth_noise_std': .01, 'depth_noise_clip': .1, 'depth_dropout': .3}


def validate(options):
    if set(options) - {'mode', 'blur_kernel', *LIMITS}:
        raise ValueError('RGB-D augmentation accepts photometric and depth-observation controls; see the image augmentation guide')
    mode = options.get('mode', 'custom')
    if mode not in ('native', 'none', 'custom'):
        raise ValueError('augmentation.mode must be native, none or custom')
    if mode != 'custom' and set(options) - {'mode'}:
        raise ValueError('Custom augmentation parameters require mode: custom')
    for key, maximum in LIMITS.items():
        if key in options and not finite(options[key], .001 if key.startswith('blur_sigma') else 0, maximum):
            raise ValueError('Invalid RGB-D augmentation parameter: ' + key)
    kernel = options.get('blur_kernel', 5)
    if type(kernel) is not int or not 3 <= kernel <= 31 or kernel % 2 == 0:
        raise ValueError('blur_kernel must be an odd integer from 3 to 31')
    if options.get('blur_sigma_min', .1) > options.get('blur_sigma_max', 2.):
        raise ValueError('blur_sigma_min must not exceed blur_sigma_max')


def configure_dataset(dataset, config):
    options = config.augmentation
    if not options or options.get('mode') in ('native', 'none'): return dataset
    from torchvision.transforms import ColorJitter, Compose, GaussianBlur, RandomApply, RandomGrayscale
    photo = []
    if any(options.get(k, 0) for k in ('brightness', 'contrast', 'saturation', 'hue')):
        photo.append(ColorJitter(*(options.get(k, 0) for k in ('brightness', 'contrast', 'saturation', 'hue'))))
    if options.get('grayscale_probability', 0):
        photo.append(RandomGrayscale(options['grayscale_probability']))
    if options.get('blur_probability', 0):
        photo.append(RandomApply([GaussianBlur(options.get('blur_kernel', 5),
            (options.get('blur_sigma_min', .1), options.get('blur_sigma_max', 2.)))], p=options['blur_probability']))
    # Native get_rgb saves the same augmented full-resolution RGB for local
    # features before resizing it for the anchor network.
    dataset.aug = Compose(photo) if photo else None
    if options.get('depth_noise_std', 0) or options.get('depth_dropout', 0):
        from functools import partial
        dataset.get_depth = partial(depth_observation, dataset, options)
    return dataset


def depth_observation(dataset, options, index, rot=0, zoom=1.):
    import numpy as np
    from PIL import Image
    if rot != 0 or zoom != 1:
        raise ValueError('Image-space geometry requires transformed camera and local grasp labels')
    with Image.open(dataset.depthpath[index]) as image:
        depth = np.asarray(image, dtype=np.float32).copy()
    if dataset.trainning:
        valid = depth > 0
        std = options.get('depth_noise_std', 0)
        if std:
            z = depth[valid] / 1000.
            noise = np.clip(np.random.normal(0, std*z**2), -options.get('depth_noise_clip', .01), options.get('depth_noise_clip', .01))
            depth[valid] = np.maximum(z+noise, 1e-6) * 1000.
        dropout = options.get('depth_dropout', 0)
        if dropout: depth[valid & (np.random.random(depth.shape) < dropout)] = 0
    dataset.cur_depth = depth.T.copy()
    reduced = np.asarray(Image.fromarray(depth).resize(dataset.output_size), dtype=np.float32) / 1000.
    return np.clip(reduced-reduced.mean(), -1, 1).T
