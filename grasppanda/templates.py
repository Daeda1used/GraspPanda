"""Create editable local configurations from method presets and curated examples."""
from copy import deepcopy
from pathlib import Path

from .config import ROOT, Experiment, default_dataset


def examples():
    import yaml
    return yaml.safe_load((ROOT / 'grasppanda/resources/examples.yaml').read_text())


def configuration(*, method=None, example=None, dataset_root=None):
    if method and example:
        raise ValueError('Choose a method preset or an example')
    if example:
        entries = examples()
        if example not in entries:
            raise ValueError('Unknown example; use ./panda init --list')
        data = deepcopy(entries[example]['config'])
        target = data.get('base', data)
        if dataset_root is not None:
            target['dataset_root'] = dataset_root
    else:
        from .recipes import NO_DATA, preset
        from .weights import records
        method = method or 'graspnet_baseline'
        root = dataset_root if dataset_root is not None else default_dataset()
        root = root or ('' if method in NO_DATA else '/data/GraspNet-1B')
        config = preset(method, root)
        defaults = Experiment().to_dict()
        required = {'method', 'action', 'dataset_root', 'camera', 'checkpoint'}
        data = {key: value for key, value in config.to_dict().items()
                if key in required or value != defaults[key]}
        # Write the registered destination even before the weights are downloaded.
        if config.action == 'infer':
            data['checkpoint'] = next((r['path'] for r in records(method, config.camera)
                                       if r.get('role', 'primary') == 'primary'), '')
    if 'base' in data:
        from .sweeps import Sweep
        Sweep.from_dict(data)
    else:
        Experiment.from_dict(data)
    return data


def write_configuration(data, output):
    import yaml
    destination = Path(output)
    if destination.suffix not in ('.yaml', '.yml'):
        raise ValueError('Use a .yaml or .yml output file')
    payload = '# Local experiment. Set your data paths before running.\n'
    payload += yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    # Exclusive creation also protects existing files and symlinks.
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return destination
