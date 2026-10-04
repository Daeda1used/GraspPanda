"""Read-only method descriptions built from the executable registries."""
from dataclasses import asdict
import json

from .config import ROOT, capabilities, catalogue
from .datasets import datasets, get_dataset


SOURCE_LABELS = {
    'author_source': 'Author source',
    'third_party_port': 'Third-party port',
    'source_mirror': 'Source mirror',
    'toolbox_reconstruction': 'Toolbox reconstruction',
    'author_binary_sdk': 'Author binary SDK',
    'unverified': 'Authorship not verified',
}


def method_description(method, dataset=None):
    """Describe provenance and contracts without importing a model or CUDA."""
    from .components import slots
    from .module_options import schema
    from .weights import records
    items = catalogue()
    if method not in items:
        raise ValueError('Unknown method: '+method)
    if dataset is not None:
        get_dataset(dataset)
    item = items[method]
    supported = {key: capabilities(method, key) for key in datasets()
                 if capabilities(method, key)}
    if dataset is not None and dataset not in supported:
        raise ValueError(f'{method} has no operations on {dataset}')
    pins = json.loads((ROOT/'grasppanda/resources/upstreams.lock.json').read_text())
    pin = next((row for row in pins if row['id'] == method), {})
    source = dict(item.get('provenance', {}))
    source['kind'] = source.get('kind', 'unverified')
    source['label'] = SOURCE_LABELS[source['kind']]
    source['repository'] = item['repository']
    source['commit'] = pin.get('pinned_commit', pin.get('commit'))
    source['revision_url'] = (item['repository']+'/tree/'+source['commit']
                              if source['commit'] else None)
    components = []
    for slot in slots(method):
        component = asdict(slot)
        component['configuration_path'] = 'modules.'+slot.name
        component['parameters'] = {choice: schema(method, slot.name, choice)
                                   for choice in slot.choices}
        components.append(component)
    weights = {}
    for key in supported if dataset is None else (dataset,):
        from .recipes import checkpoint_camera
        for camera in get_dataset(key).cameras:
            for record in records(method, checkpoint_camera(method, camera, key)):
                weights[record['path']] = record
    preset_dataset = dataset or next(iter(supported), None)
    return {
        'id': method, 'name': item.get('name', method), 'paper_title': item['paper_title'],
        'source': source, 'datasets': supported, 'selected_dataset': dataset,
        'protocol': get_dataset(dataset).protocol_note if dataset else '',
        'input_note': item.get('dataset_notes', {}).get(dataset, item.get('input_note', '')),
        'components': components, 'checkpoints': list(weights.values()),
        'configuration': {
            'preset': f'./panda init --method {method} --dataset {preset_dataset}' if preset_dataset else None,
            'modules': 'Compose modules → selectors and Component parameters by slot',
            'training': 'Training settings → Trainer parameters; JSON/YAML trainer mapping',
            'full_config': 'Configuration editor → Generate from form → edit JSON',
            'reference': item.get('configuration_reference','docs/REFERENCE.md'),
            'parameters': item.get('configuration_parameters',{}),
        },
    }


def format_method(description):
    """Render the same source and configuration information in the UI and CLI."""
    source = description['source']
    lines = [f"### {description['name']}", description['paper_title'], '',
             f"**Implementation: {source['label']}** · {source.get('publication') or 'Publication not verified'}",
             f"[Source repository]({source['repository']})"]
    if source.get('paper_url'):
        lines[-1] += f" · [Paper]({source['paper_url']})"
    if source.get('revision_url'):
        lines[-1] += f" · [Pinned revision `{source['commit'][:12]}`]({source['revision_url']})"
    if source.get('adapter_note'):
        lines += ['', '**Toolbox adaptation.** '+source['adapter_note']]
    if description['input_note']:
        lines += ['', description['input_note']]
    if description['protocol']:
        lines += ['', description['protocol']]
    lines += ['', '| Dataset | Available operations |', '|---|---|']
    for dataset, actions in description['datasets'].items():
        lines.append(f"| {get_dataset(dataset).title} | {', '.join(actions)} |")
    lines += ['', '**Component configuration**']
    for slot in description['components']:
        lines.append(f"- `{slot['configuration_path']}`: "+', '.join(f'`{c}`' for c in slot['choices']))
    if not description['components']:
        lines.append('This adapter retains its native architecture; no interchangeable slots are registered.')
    parameters = description['configuration']['parameters']
    if parameters:
        lines += ['', '| Parameter | Default | Where to configure |', '|---|---|---|']
        for name,rule in parameters.items():
            lines.append(f"| `{name}` | `{rule['default']}` | {rule['ui']} |")
        lines += ['', '[Parameter guide](https://github.com/Daeda1used/GraspPanda/blob/main/'+description['configuration']['reference']+')']
    lines += ['', 'Use **Compose modules** for compatible choices and accepted parameter names. '
              '**Configuration editor** stores the full experiment as JSON. '
              '**Guide → Modules / Usage** explains checkpoint transfer and training.',
              '', (f"CLI: `{description['configuration']['preset']}`. " if description['configuration']['preset'] else '')+
              f"`./panda describe {description['id']} --json` also exposes slot contracts, "
              'parameter rules and registered checkpoint sources.']
    return '\n'.join(lines)
