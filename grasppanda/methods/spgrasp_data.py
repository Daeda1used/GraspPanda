"""RGB sequence and continuous planar targets for the SPGrasp adapter."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Letterbox:
    height: int
    width: int
    resolution: int = 512

    def __post_init__(self):
        if any(type(value) is not int for value in (self.height, self.width, self.resolution)) or min(self.height, self.width) < 2 or self.resolution not in (256, 512, 768, 1024):
            raise ValueError('Invalid image dimensions or SPGrasp resolution')

    @property
    def resized(self):
        scale = self.resolution / max(self.height, self.width)
        return max(1, int(self.height * scale)), max(1, int(self.width * scale))

    @property
    def padding(self):
        h, w = self.resized
        left, top = (self.resolution - w) // 2, (self.resolution - h) // 2
        return left, top, self.resolution - w - left, self.resolution - h - top

    def image(self, image):
        if image.size != (self.width, self.height):
            raise ValueError('Image dimensions change within the selected sequence')
        image = TF.resize(image, self.resized, antialias=True)
        return TF.pad(image, self.padding, fill=0)

    def targets(self, targets):
        if targets.shape[-2:] != (self.height, self.width):
            raise ValueError('Target geometry does not match the RGB frame')
        result = TF.resize(targets.float(), self.resized, interpolation=TF.InterpolationMode.NEAREST)
        return TF.pad(result, self.padding, fill=0)

    def points(self, xy, inverse=False):
        """Transform pixel centers using the image resize convention."""
        xy = torch.as_tensor(xy, dtype=torch.float32)
        h, w = self.resized
        scale = xy.new_tensor([w / self.width, h / self.height])
        offset = xy.new_tensor(self.padding[:2])
        return (xy - offset + .5) / scale - .5 if inverse else (xy + .5) * scale - .5 + offset


def frame_paths(dataset_root, scene, camera, start, count):
    if camera not in ('kinect', 'realsense') or type(scene) is not int or not 0 <= scene < 190:
        raise ValueError('Invalid GraspNet scene or camera')
    if any(type(v) is not int for v in (start, count)) or count < 1 or start < 0 or start + count > 256:
        raise ValueError('A temporal sequence must stay within one GraspNet scene')
    directory = Path(dataset_root) / 'scenes' / f'scene_{scene:04d}' / camera
    paths = [directory / 'rgb' / f'{i:04d}.png' for i in range(start, start + count)]
    missing = next((p for p in paths if not p.is_file()), None)
    if missing:
        raise FileNotFoundError(f'RGB sequence frame missing: {missing}')
    return paths


def read_rgb_sequence(dataset_root, scene, camera, start, count, resolution=512):
    """Inference reads RGB files only; no annotation or depth dependency."""
    paths = frame_paths(dataset_root, scene, camera, start, count)
    images, geometry = [], None
    for path in paths:
        with Image.open(path) as source:
            rgb = source.convert('RGB')
        if geometry is None:
            geometry = Letterbox(rgb.height, rgb.width, resolution)
        image = TF.to_tensor(geometry.image(rgb))
        images.append(TF.normalize(image, [.485, .456, .406], [.229, .224, .225]))
    return torch.stack(images), geometry, paths


def prepare_prompts(objects, geometry):
    """Convert explicit first-frame object prompts to padded native point tensors.

    Each object has an integer id and a box [x0,y0,x1,y1], points [[x,y,label], ...],
    or both. Coordinates are pixel centers in the original RGB image.
    """
    if not isinstance(objects, list) or not 1 <= len(objects) <= 8:
        raise ValueError('Provide one to eight object prompts')
    rows, labels, ids = [], [], []
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) - {'id', 'box', 'points'}:
            raise ValueError('Each object prompt accepts id, box and points')
        oid = obj.get('id')
        if type(oid) is not int or oid < 0 or oid in ids:
            raise ValueError('Prompt object IDs must be unique nonnegative integers')
        xy, tags = [], []
        if 'box' in obj:
            box = obj['box']
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError('A prompt box is [x0,y0,x1,y1]')
            xy.extend([box[:2], box[2:]]); tags.extend([2, 3])
        points = obj.get('points', [])
        if not isinstance(points, list) or len(points) > 64:
            raise ValueError('Each object accepts at most 64 labelled points')
        for point in points:
            if not isinstance(point, list) or len(point) != 3 or type(point[2]) is not int or point[2] not in (0, 1):
                raise ValueError('Each point is [x,y,label] with integer label 0 or 1')
            xy.append(point[:2]); tags.append(point[2])
        if not xy or any(type(value) not in (int, float) or not np.isfinite(value) for pair in xy for value in pair):
            raise ValueError('Prompts require finite pixel coordinates')
        if any(not 0 <= x < geometry.width or not 0 <= y < geometry.height for x, y in xy):
            raise ValueError('Prompt coordinates lie outside the RGB frame')
        if 'box' in obj and not (xy[0][0] < xy[1][0] and xy[0][1] < xy[1][1]):
            raise ValueError('Prompt boxes require x0 < x1 and y0 < y1')
        ids.append(oid); rows.append(geometry.points(xy)); labels.append(torch.tensor(tags, dtype=torch.int32))
    length = max(len(row) for row in rows)
    coords = torch.zeros(len(rows), length, 2)
    point_labels = torch.full((len(rows), length), -1, dtype=torch.int32)
    for index, (row, tags) in enumerate(zip(rows, labels)):
        coords[index, :len(row)] = row
        point_labels[index, :len(tags)] = tags
    return ids, {'point_coords': coords, 'point_labels': point_labels}


def rectangle_directory(dataset_root, scene, camera, label_root=''):
    """Support both published rectangle-archive layouts without moving data."""
    relative = Path(f'scene_{scene:04d}') / camera
    choices = [Path(label_root) / relative] if label_root else [
        Path(dataset_root) / 'rect_labels' / relative,
        Path(dataset_root) / 'scenes' / relative / 'rect',
    ]
    for path in choices:
        if path.is_dir():
            return path
    raise FileNotFoundError('Rectangle labels missing; set label_root to the directory containing scene_XXXX/CAMERA')


def native_targets(dataset_root, scene, camera, frame, width_scale, object_ids=None, label_root=''):
    """Retain the author rasterizer and angle convention with continuous targets.

    Width is measured in original RGB pixels and divided by an explicit scale.
    Visible instances retain semantic supervision even when no rectangle is labelled.
    """
    from training.dataset.vos_segment_loader import (
        _compute_grasp_rectangles_from_graspnet_format,
        _generate_global_grasp_map_from_rectangles,
    )
    from training.grasp_dataset.grasp_OCID import GraspMat
    if type(width_scale) not in (int, float) or not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError('Width scale must be positive and finite')
    directory = Path(dataset_root) / 'scenes' / f'scene_{scene:04d}' / camera
    with Image.open(directory / 'label' / f'{frame:04d}.png') as image:
        semantic = np.asarray(image).copy()
    if semantic.ndim != 2 or semantic.dtype.kind not in 'ui' or (semantic > 88).any():
        raise ValueError('GraspNet instance labels must be a two-dimensional image')
    rectangles = np.load(rectangle_directory(dataset_root, scene, camera, label_root) / f'{frame:04d}.npy', allow_pickle=False)
    if rectangles.ndim != 2 or rectangles.shape[1] != 7 or not np.isfinite(rectangles).all() or (rectangles[:, 4] < 0).any():
        raise ValueError('Invalid GraspNet rectangle annotations; expected finite [N,7] rows')
    widths = 2 * np.linalg.norm(rectangles[:, 2:4] - rectangles[:, :2], axis=1)
    if (widths > width_scale).any():
        raise ValueError('Rectangle width exceeds width_scale; increase the explicit scale, without clipping labels')
    corners = _compute_grasp_rectangles_from_graspnet_format(rectangles)
    raster = _generate_global_grasp_map_from_rectangles(corners, semantic.shape)
    ids = sorted(int(value) for value in np.unique(semantic) if value != 0) if object_ids is None else list(object_ids)
    if len(ids) != len(set(ids)) or any(type(oid) is not int or not 1 <= oid <= 88 for oid in ids):
        raise ValueError('Training instance IDs must be unique integers from 1 to 88')
    result = {}
    for oid in ids:
        mask = semantic == oid
        encoded = GraspMat.decode(np.where(mask[None], raster, 0)).astype(np.float32)
        encoded[3] *= 100.0 / width_scale
        targets = np.concatenate([encoded, mask[None].astype(np.float32)])
        if not np.isfinite(targets).all():
            raise ValueError('Non-finite planar training target')
        result[oid] = torch.from_numpy(targets)
    return result


def decode_maps(logits, geometry, object_ids, width_scale, score_threshold=.5,
                semantic_threshold=.5, max_grasps=10, min_distance=20., width_positive_weight=10.):
    """Decode four-DoF openings in original image coordinates, without 3D claims."""
    from scipy.ndimage import maximum_filter
    if logits.shape != (len(object_ids), 5, geometry.resolution, geometry.resolution) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite object-major five-channel planar logits')
    if not 0 <= score_threshold <= 1 or not 0 <= semantic_threshold <= 1:
        raise ValueError('Planar score thresholds must lie in [0,1]')
    if type(max_grasps) is not int or not 1 <= max_grasps <= 100 or not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError('Invalid planar output limits')
    if not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError('Width scale must be positive and finite')
    if not np.isfinite(width_positive_weight) or width_positive_weight <= 0:
        raise ValueError('Width positive-class weight must be positive and finite')
    probabilities = logits[:, [0, 3, 4]].float().sigmoid().detach().cpu().numpy()
    # For continuous targets, weighted BCE has optimum logit(t) + log(alpha).
    # Undo that shift before interpreting the width channel in physical pixels.
    # Alpha is saved from training, never estimated from inference annotations.
    probabilities[:, 1] = (logits[:, 3].float() - np.log(width_positive_weight)).sigmoid().detach().cpu().numpy()
    angles = logits[:, 1:3].float().tanh().detach().cpu().numpy()
    left, top, _, _ = geometry.padding
    height, width = geometry.resized
    results = []
    for index, oid in enumerate(object_ids):
        quality, widths, semantic = probabilities[index]
        valid = np.zeros_like(quality, dtype=bool)
        valid[top:top + height, left:left + width] = True
        valid &= (quality >= score_threshold) & (semantic >= semantic_threshold)
        peaks = valid & (quality == maximum_filter(quality, size=3, mode='nearest'))
        y, x = np.where(peaks)
        order = np.argsort(-quality[y, x], kind='stable')
        chosen = []
        for i in order:
            center = geometry.points([float(x[i]), float(y[i])], inverse=True).tolist()
            if not (0 <= center[0] < geometry.width and 0 <= center[1] < geometry.height):
                continue
            if any(np.linalg.norm(np.array(center) - old) < min_distance for old in chosen):
                continue
            # The author encodes the rectangle's height edge. Add pi/2 to recover
            # the opening direction represented by GraspNet's center-to-right vector.
            angle = .5 * np.arctan2(angles[index, 1, y[i], x[i]], angles[index, 0, y[i], x[i]])
            angle = (angle + np.pi) % np.pi - np.pi / 2
            results.append({'object_id': oid, 'center': center, 'angle_radians': float(angle),
                            'width_pixels': float(widths[y[i], x[i]] * width_scale),
                            'score': float(quality[y[i], x[i]]),
                            'semantic_score': float(semantic[y[i], x[i]])})
            chosen.append(center)
            if len(chosen) == max_grasps:
                break
    return results


def draw_preview(rgb_path, grasps, destination):
    """Draw planar openings; the short endpoint ticks are display glyphs only."""
    from PIL import ImageDraw
    with Image.open(rgb_path) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    for grasp in sorted(grasps, key=lambda row: -row['score'])[:30]:
        center = np.asarray(grasp['center'], dtype=np.float64)
        theta = grasp['angle_radians']
        direction = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-direction[1], direction[0]])
        half = direction * grasp['width_pixels'] / 2
        left, right = center - half, center + half
        draw.line([tuple(left), tuple(right)], fill=(61, 225, 161), width=2)
        for end in (left, right):
            draw.line([tuple(end - normal * 8), tuple(end + normal * 8)], fill=(61, 225, 161), width=3)
        draw.ellipse([*(center - 3), *(center + 3)], fill=(255, 214, 95))
    draw.rectangle((0, 0, 340, 24), fill=(20, 30, 30))
    draw.text((8, 6), 'Planar openings (pixels); no 3D collision check', fill='white')
    image.save(destination)
