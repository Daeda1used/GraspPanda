"""Deterministic configuration grids with method-aware validation."""
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import json
import math
import re

from .config import Experiment


@dataclass(frozen=True)
class Sweep:
    base: dict
    grid: dict

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {'base', 'grid'}:
            raise ValueError('A sweep requires exactly base and grid mappings')
        sweep = cls(deepcopy(data['base']), deepcopy(data['grid']))
        sweep.expand()
        return sweep

    def to_dict(self):
        return dict(base=deepcopy(self.base), grid=deepcopy(self.grid))

    def expand(self):
        if not isinstance(self.base, dict) or not isinstance(self.grid, dict) or not self.grid:
            raise ValueError('Provide a base configuration and a nonempty parameter grid')
        for path in self.grid:
            if not isinstance(path, str) or not re.fullmatch(r'[a-z_][a-z_0-9]*(\.[a-z_][a-z_0-9]*)*', path):
                raise ValueError('Grid keys must be dotted configuration paths')
        paths = sorted(self.grid)
        for path in paths:
            values = self.grid[path]
            if not isinstance(values, list) or not values:
                raise ValueError(f'{path}: provide a nonempty list of values')
            if any(other.startswith(path + '.') for other in paths):
                raise ValueError(f'Overlapping grid paths: {path}')
        if math.prod(len(self.grid[path]) for path in paths) > 128:
            raise ValueError('A sweep supports at most 128 combinations; split larger grids')
        configs, seen = [], set()
        for values in product(*(self.grid[path] for path in paths)):
            data = deepcopy(self.base)
            for path, value in zip(paths, values):
                keys = path.split('.')
                node = data
                for key in keys[:-1]:
                    # Component names can use the compact string notation.
                    if key in ('backbone', 'crop', 'head') and isinstance(node.get(key), str):
                        node[key] = {'type': node[key]}
                    node = node.setdefault(key, {})
                    if not isinstance(node, dict):
                        raise ValueError(f'{path}: parent must be a configuration mapping')
                node[keys[-1]] = deepcopy(value)
            config = Experiment.from_dict(data)
            canonical = json.dumps(config.to_dict(), sort_keys=True, allow_nan=False)
            if canonical in seen:
                raise ValueError('Grid produces duplicate configurations; remove repeated values')
            seen.add(canonical)
            configs.append(config)
        return configs

    def preview(self):
        return [config.to_dict() for config in self.expand()]
