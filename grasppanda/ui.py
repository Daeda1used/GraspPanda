"""Browser interface for reproducible visual grasping experiments."""
from functools import partial
import json
import os
from pathlib import Path
import zipfile
import subprocess
import sys

import gradio as gr

from .config import ROOT, Experiment, capabilities, catalogue, default_dataset
from .jobs import JobManager
from .recipes import RECIPES, CHECKPOINT_RECIPES, preset
from .weights import primary, fetch, records
from .components import slots
from .datasets import datasets

CSS = """.gradio-container {max-width: 1480px !important;}
#panda-hero {padding:28px 32px; background:linear-gradient(110deg,#102d2a,#245d50);border-radius:18px;color:#fff;margin-bottom:18px;}
#panda-hero h1 {font-size:36px;color:#fff;margin:0 0 8px;letter-spacing:-1px;}
#panda-hero p {color:#d0e6dc;font-size:16px;margin:0;}
.panda-note {padding:12px 16px;border-left:3px solid #40aa84;background:#edf8f2;border-radius:8px;}
"""


ACTION_LABELS = {
    'infer': 'Predict grasps', 'evaluate': 'Evaluate predictions',
    'recipe': 'Run native recipe', 'train': 'Train across epochs',
    'train_short': 'Short training run',
}


def action_choices(method):
    available = capabilities(method)
    return [(label, a) for a, label in ACTION_LABELS.items() if a in available]


def method_card(method):
    item = catalogue()[method]
    note = item.get('input_note', '')
    recipe = RECIPES.get(method, ('', ''))[1] if 'infer' not in capabilities(method) else ''
    return (f"### {item.get('name', method)}\n{item['paper_title']}\n\n"
            f"[Implementation]({item['repository']})\n\n{recipe or note}")


def component_parameters(method, backbone, crop, head='upstream', memory='upstream', sampling='upstream'):
    from .module_options import schema
    rows = []
    available = {slot.name: slot for slot in slots(method)}
    for slot, choice in (('backbone', backbone), ('crop', crop), ('head', head), ('memory', memory), ('sampling', sampling)):
        if slot not in available or choice not in available[slot].choices:
            continue
        for key, rule in schema(method, slot, choice).items():
            if rule[0] == 'pointtpa':
                description = 'Mapping with type: pointtpa; optional blocks selects flat encoder block indices. bottleneck_channels, experts, group_size, group_mode (num/length), dynamic_down, dynamic_up and scale accept one value or one per selected block. See Guide → Modules for pretrained and freezing behavior.'
            elif rule[0] in ('sampler', 'samplers'):
                description = ('One output-seed policy' if rule[0] == 'sampler' else 'Four policies, one per downsampling stage') + '; name or {type, ...}: upstream, uniform, fps, pointsp_wrs, pointsp_ffps. Density policies accept neighbors and density_quantile; FFPS adds keep_ratio; FPS/FFPS accept start: first or random. Use {train: POLICY, eval: POLICY} for mode-specific sampling.'
            elif rule[0] == 'flash3d_pooling':
                description = 'mean, sum, min or max; one name for all transitions, or a list with one name between each pair of hierarchy levels'
            elif rule[0] == 'mscq_branches':
                description = 'Four branch policies in increasing native radius order. Each accepts type and radius_scale; upstream accepts nsample. Replacements accept their Baseline crop parameters. Native fusion, gate and heads remain in place.'
            elif rule[0] == 'choice':
                description = ', '.join(rule[1])
            elif rule[0] in ('int', 'float'):
                description = f'{rule[0]}: {rule[1]} to {rule[2]}'
            elif choice == 'sonata_ptv3' and key == 'stride':
                description = '4 pooling strides, each 1, 2, 4 or 8'
            elif rule[0] == 'int_list':
                description = f'{rule[1]} integers, each {rule[2]} to {rule[3]}'
            elif rule[0] == 'int_sequence':
                description = f'{rule[1]}–{rule[2]} integers, each {rule[3]} to {rule[4]}'
            elif rule[0] == 'per_stage':
                scalar = rule[2]
                values = 'true or false' if scalar[0] == 'bool' else f'number: {scalar[1]} to {scalar[2]}'
                description = values + ('; one value for all attention layers, or one per layer' if slot == 'head' else '; one value for all stages, or a list with one value per stage')
            elif rule[0] == 'float_matrix':
                description = f'{rule[1]}–{rule[2]} stage lists; each has {rule[3]}–{rule[4]} grid sizes from {rule[5]} to {rule[6]} stage lattice cells'
            elif rule[0] == 'float_list':
                description = f'{rule[1]}–{rule[2]} numbers, each {rule[3]} to {rule[4]}'
            elif rule[0] == 'per_block':
                scalar = rule[2]
                values = 'true or false' if scalar[0] == 'bool' else f'integer: {scalar[1]} to {scalar[2]}'
                description = values + ('; one value for all attention layers, or one per layer' if slot == 'head' else '; one value for all scan blocks, or a list with one value per active block')
            elif rule[0] == 'choice_list':
                description = f'{rule[1]}–{rule[2]} values: ' + ', '.join(rule[3])
            elif rule[0] == 'checkpoint_path':
                description = 'Path to a matching DSN segmentation checkpoint; omit to use the registered RealSense weights'
            elif rule[0] == 'bool':
                description = 'true or false'
            else:
                description = {'channels': '1–8 layer widths, each 8–2048',
                               'radii': '1–8 radius factors, each 0.1–4',
                               'blocks': '5 stage depths, each 1–12'}[rule[0]]
            if choice == 'sp2t':
                if key.startswith('block_'):
                    description += '; encoder fine to coarse, then decoder coarse to fine; one entry per block; overrides shared settings'
                elif key.startswith(('enc_', 'dec_')):
                    description += '; stages listed fine to coarse; decoder has one fewer stage'
                elif key == 'serialization_depth':
                    description += '; fixed lattice bit depth keeps Hilbert order independent of other scenes in the batch'
                elif key in ('proxy_start_stage', 'proxy_end_stage'):
                    description += '; inclusive stage interval; applies to both encoder and decoder'
            if choice == 'quality_residual' and key == 'initial_probability':
                description += '; sets the residual logit bias, not the final probability; the native score also contributes'
            if choice == 'swin3d':
                if key.startswith('block_'):
                    description += '; full vector: encoder fine to coarse, then decoder coarse to fine; length sum(depths) + sum(decoder_depths); overrides the corresponding stage/shared setting'
                elif key in ('strides', 'downsample', 'knn_neighbors', 'decoder_depths', 'up_neighbors'):
                    description += '; one per transition/decoder target level, listed fine to coarse; length len(channels) - 1'
                elif key in ('depths', 'heads', 'window_sizes', 'quant_sizes'):
                    description += '; one per stage, matching len(channels)'
                if key in ('heads', 'block_heads'):
                    description += '; even heads, with 8, 16 or 32 channels per head'
                elif key == 'decoder_depths':
                    description += '; zero retains interpolation and removes attention'
                elif key == 'rpe_features':
                    description += '; xyz_normals requires FineGrasp with native normals enabled'
            if choice == 'pointrwkv_released' and key.startswith('block_'):
                description += '; one value per encoder block, in stage order; length equals sum(depths); overrides the corresponding stage or shared setting'
            if choice == 'pointrwkv_released' and key in ('decoder_channels', 'decoder_depths'):
                description += '; coarse to fine propagation stages, ending at the original input points'
            if choice == 'pointhr' and key in ('dec_channels', 'dec_depths', 'dec_groups', 'dec_neighbours'):
                description += '; finest original-point resolution to coarsest decoded resolution'
            if choice == 'fastvit':
                if rule[0] in ('per_stage','per_block'):
                    scalar = rule[2]
                    description = 'true or false' if scalar[0] == 'bool' else f'{scalar[0]}: {scalar[1]} to {scalar[2]}'
                if key == 'block_mixers':
                    description = 'repmixer or attention; one entry per block, in fine-to-coarse stage order'
                elif key.startswith('block_'):
                    description += '; scalar or one entry per block, in fine-to-coarse stage order'
                elif key.startswith('attention_'):
                    description += '; scalar or one entry per attention block, in stage order'
                elif key == 'repmixer_kernels':
                    description += '; scalar or one odd kernel per RepMixer block, in stage order'
                elif key == 'parameterization':
                    description += '; selects branch or fused convolution weights; FastViTHD pretraining requires fused'
                elif key == 'trainable_stages':
                    description += '; last N stages, including their downsamplers; 0 freezes the RGB encoder'
            if choice == 'efficientvit':
                if key in ('attention_dims', 'attention_heads', 'attention_bias'):
                    description += '; scalar or one value per attention block, in fine-to-coarse stage order'
                elif key == 'attention_scales':
                    description += '; one list of odd kernel sizes per attention block, or one list broadcast to all blocks; [[]] disables extra aggregation scales'
                elif key == 'stage_depths':
                    description += '; five native stage counts; B-family stages 1/2 include downsampling, other stages count blocks after the stem/downsampler'
                elif key == 'trainable_stages':
                    description += '; last N RGB stages; 0 freezes the encoder; depth and grasp projections remain trainable'
                elif key == 'pretrained':
                    description += '; ImageNet initialization; structural edits require false; L0 has no registered checkpoint'
            if choice == 'mambavision':
                if key.startswith('block_'):
                    if rule[0] == 'choice_list':
                        description = ', '.join(rule[3]) + '; one entry per block'
                    else:
                        scalar = rule[2]
                        description = ('true or false' if scalar[0] == 'bool' else f'{scalar[0]}: {scalar[1]} to {scalar[2]}') + '; scalar or one entry per block'
                    description += '; stage 3 followed by stage 4'
                    if key in ('block_state_dims', 'block_conv_sizes', 'block_expansions', 'block_dt_ranks'):
                        description += '; Mamba blocks only'
                    elif key in ('block_heads', 'block_qkv_bias', 'block_qk_norm'):
                        description += '; attention blocks only'
                elif key in ('stage_heads', 'window_sizes'):
                    description += '; stages 3 and 4 only; stages 1 and 2 use convolution'
                elif key == 'pretrained':
                    description += '; author ImageNet initialization; structural edits require false; source and weights use NVIDIA non-commercial research terms'
                elif key == 'trainable_stages':
                    description += '; last N encoder stages; 0 freezes the RGB encoder, 4 also trains its patch embedding; depth and grasp projections remain trainable'
                elif key == 'gradient_checkpointing':
                    description += '; recompute hybrid blocks in stages 3 and 4 during backward'
            if choice == 'rala' and rule[0] == 'choice_list':
                description = 'One rala or softmax attention choice per encoder block; list length equals sum(stage_depths), in fine-to-coarse stage order'
            if choice == 'pointcnnpp':
                if key == 'block_kernel_sizes': description = '1, 3 or 5; one value for all residual blocks, or one per block in encoder then decoder order'
                elif key == 'block_radius_scalers': description = '0.1 to 8; one value for all residual blocks, or one per block; scales the neighborhood sphere volume'
                elif key == 'block_activations': description = 'One relu, gelu or silu per residual block; list length equals the sum of depths'
            if choice == 'flash3d':
                if rule[0] in ('per_block', 'per_stage'):
                    scalar = rule[2]
                    values = 'true or false' if scalar[0] == 'bool' else f'{scalar[0]}: {scalar[1]} to {scalar[2]}'
                    description = values + '; scalar for all encoder/decoder blocks, or one value per block in fine-to-coarse level order'
                elif rule[0] == 'choice_list':
                    description = 'Repeating block pattern: ' + ', '.join(rule[3]) + '; follows encoder/decoder blocks in fine-to-coarse level order'
            if choice == 'kpconvx_cylinder':
                if rule[0] in ('per_block', 'per_stage'):
                    description = values + '; one value for all kernel blocks, or one per block'
                if key in ('kernel_radius', 'kernel_sigma'):
                    description += '; in units of the query radius'
                elif key == 'channels':
                    description += '; embedding width followed by one output width per block'
                elif key == 'chunk_size':
                    description += '; complete cylinders per memory chunk; 0 processes all'
                elif key == 'normalization':
                    description += '; batch requires chunk_size: 0; group statistics are per cylinder'
                elif key == 'checkpoint':
                    description += '; recompute activations during backward to reduce memory'
            rows.append(f'| `{slot}.{key}` | {description} |')
    return ('| Parameter | Accepted values |\n|---|---|\n'+'\n'.join(rows)+'\n\nOmitted parameters use component defaults. Cross-stage constraints are checked when generating or running the configuration.') if rows else 'The selected components use their native settings.'


def prime_preset(method, action, policy, current):
    try:
        if method not in ('hggd', 'region_normalized_grasp') or action not in ('train', 'train_short'):
            raise ValueError('PRIME photometric augmentation requires HGGD or RNG training')
        options = json.loads(current or '{}')
        if not isinstance(options, dict): raise ValueError('Augmentation configuration must be a mapping')
        policies = {'color': ['color'], 'filter': ['filter'], 'color_filter': ['color', 'filter']}
        if policy == 'none': options.pop('prime', None)
        elif policy in policies:
            previous = options.get('prime', {})
            if not isinstance(previous, dict): raise ValueError('augmentation.prime must be a mapping')
            options['mode'] = 'custom'
            options['prime'] = {**previous, 'primitives': policies[policy]}
        else: raise ValueError('Unknown PRIME photometric policy')
        from .training.image_augmentation import validate
        validate(options)
        return json.dumps(options, indent=2)
    except (ValueError, TypeError) as error:
        raise gr.Error(str(error)) from error


def loss_parameters(method):
    from grasppanda.training.options import LOSS_TERMS
    from grasppanda.training.losses import PARAMETERS, is_classification, parameter_schema
    if method == 'spgrasp':
        return 'Planar training accepts loss.weights for position (default 2), angle, width and semantic (default 1 each). Native loss formulations are retained. Photometric augmentation accepts native, none, or custom brightness, contrast, saturation, grayscale and consistent. See Guide → Modules for units and prompt supervision.'
    terms = LOSS_TERMS.get(method, {})
    if not terms:
        return 'This method uses its native objective and augmentation. Custom controls are not registered.'
    from grasppanda.training.losses import choices
    available = {kind for term in terms for kind in choices(term, method)}
    def parameter_description(key, rule):
        if rule[0] == 'formulation':
            return f'`{key}`: name or {{type, parameters}}; ' + ', '.join(rule[1:])
        if rule[0] == 'choice':
            return f'`{key}`: ' + ', '.join(rule[1:])
        if key == 'norm_order':
            return '`norm_order`: 1 to 8, or "inf" for maximum absolute logit'
        return f'`{key}`: {rule[0]} to {rule[1]}'
    rows = [f'| `{name}` | ' + (', '.join(parameter_description(key, rule) for key, rule in options.items()) or 'No parameters') + ' |'
            for name in PARAMETERS if name != 'upstream' and name in available for options in [parameter_schema(method, name)]]
    semantics = ('HGGD/RNG use independent sigmoid labels: cross_entropy means binary cross-entropy, and ASL uses the multi-label formulation. Native positive thresholds, class balancing and positive-count reductions remain in place. Focal alpha applies to each classification term. '
                 if method in ('hggd', 'region_normalized_grasp') else 'Focal alpha applies only to binary objectness. ')
    if method in ('hggd', 'region_normalized_grasp'):
        semantics += ('RGB-D augmentation accepts `prime: {"primitives": ["color", "filter"], "mixture_width": 3}` in custom mode. '
                      'PRIME photometric transforms share the same full-resolution RGB between anchor and local branches. '
                      'See the RGB-D augmentation reference for strength, chain depth and probability controls. ')
    if method == 'gtg2':
        semantics = 'Graph score regression uses one scalar per candidate. Graph augmentation supports native, none, or custom half_turn_probability and point_dropout. '
    if method == 'scale_balanced_grasp':
        semantics = 'graspable uses the native robust graspability target; focal alpha applies to this binary term. View and grasp losses retain the native scale prior and weighted denominators, including the score mask shared across depths. '
    if method in ('graspnet_baseline', 'pointnet2_upgrade', 'scale_balanced_grasp', 'graspness', 'economicgrasp', 'finegrasp'):
        semantics += ('LogitNorm, MbLS and LogitClip affect only softmax classification losses during training. '
                      'LogitClip defaults to the released reciprocal scale (1 / threshold); set scale equal to threshold for norm clipping. '
                      'Its optional base selects a classification formulation with its own parameters. ')
        semantics += ('Point augmentation accepts `resampling: {"type": "pointsp_wrs", "keep_ratio": [0.5, 1.0], "neighbors": 20}` '
                      'in custom mode. Alternatives are `uniform` and `pointsp_lgd`; the latter accepts `global_fraction` '
                      '(0: local removal, 1: global removal, "random": random range). Point labels follow the same selected rows. ')
    if method in ('graspnet_baseline', 'pointnet2_upgrade', 'scale_balanced_grasp', 'graspness'):
        semantics += ('Quality BCE, Varifocal and MAL apply only to score and require the quality_residual head. '
                      'Native supervision retains the original score mask; all_angles includes zero-quality bins on object seeds. '
                      'See the quality-head guide for score mappings and normalization. ')
    classification = ', '.join(f'`{term}`' for term in terms if is_classification(method, term)) or 'none'
    return ('Loss terms: ' + ', '.join(f'`{term}`' for term in terms) + '. Classification terms: ' + classification + '; remaining terms use regression losses.\n\n'
            '| Formulation | Parameters |\n|---|---|\n' + '\n'.join(rows) +
            '\n\nUse `loss.functions.TERM: {"type": "NAME", ...}` in experiment JSON. ' + semantics +
            'See [Training controls](https://github.com/Daeda1used/GraspPanda/blob/main/docs/REFERENCE.md#training-controls) for defaults, target units and augmentation parameters.')


def loss_preset(method, classification, regression, current, head='upstream', quality='upstream'):
    from grasppanda.training.options import LOSS_TERMS
    from grasppanda.training.losses import is_classification
    if method not in LOSS_TERMS:
        raise gr.Error('This method uses its native objective; loss overrides are not registered.')
    try:
        value = json.loads(current or '{}')
        if not isinstance(value, dict): raise ValueError('Loss configuration must be a mapping')
        value['functions'] = {term: classification if is_classification(method, term) else regression
                              for term in LOSS_TERMS[method]}
        if quality != 'upstream':
            from .training.quality import METHODS, LOSSES
            if method not in METHODS or quality not in LOSSES:
                raise ValueError('Choose a registered quality formulation for a compatible method')
            value['functions']['score'] = quality
        from grasppanda.training.options import validate_training_options
        validate_training_options(Experiment(method=method, action='train' if method == 'gtg2' else 'train_short',
            loss=value, modules={'head': head} if head != 'upstream' else {}))
    except (ValueError, TypeError) as error:
        raise gr.Error(str(error)) from error
    return json.dumps(value, indent=2)


def sampling_preset(method, action, kind, minimum, maximum, current):
    if method not in ('graspnet_baseline', 'pointnet2_upgrade', 'scale_balanced_grasp', 'graspness', 'economicgrasp', 'finegrasp') or action not in ('train', 'train_short'):
        raise gr.Error('Observation sampling requires a registered point training operation.')
    try:
        value = json.loads(current or '{}')
        if not isinstance(value, dict): raise ValueError('Augmentation configuration must be a mapping')
        previous = value.pop('resampling', {})
        if not isinstance(previous, dict): raise ValueError('Existing resampling configuration must be a mapping')
        if kind != 'none':
            # Preserve advanced settings for the same rule; carry only shared
            # controls across rules so incompatible parameters cannot linger.
            options = previous if previous.get('type', 'uniform') == kind else {k: v for k, v in previous.items() if k == 'probability'}
            options.update(type=kind, keep_ratio=minimum if minimum == maximum else [minimum, maximum])
            value.update(mode='custom', resampling=options)
        from grasppanda.training.options import validate_training_options
        validate_training_options(Experiment(method=method, action=action, augmentation=value))
    except (ValueError, TypeError) as error:
        raise gr.Error(str(error)) from error
    return json.dumps(value, indent=2)


def documentation(name):
    """Render repository guides with usable links inside the browser app."""
    import re
    import posixpath
    text = (ROOT/'docs'/name).read_text()
    def link(match):
        target = match.group(2)
        if '://' in target:
            return match.group(0)
        if target.startswith('#'):
            target = name + target
        target = posixpath.normpath(posixpath.join('docs', target))
        return f"[{match.group(1)}](https://github.com/Daeda1used/GraspPanda/blob/main/{target})"
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)


def prompt_frame(dataset, camera, scene, frame):
    from PIL import Image
    from .methods.spgrasp_data import frame_paths
    try:
        path = frame_paths(str(ROOT / Path(dataset).expanduser()), int(scene), camera, int(frame), 1)[0]
        with Image.open(path) as image:
            return image.convert('RGB'), '[]'
    except (OSError, ValueError, TypeError) as error:
        raise gr.Error(str(error)) from error


def add_prompt_point(current, object_id, label, canvas, evt: gr.SelectData):
    from PIL import ImageDraw
    try:
        if canvas is None:
            raise ValueError('Load the selected RGB frame first')
        objects = json.loads(current or '[]')
        if not isinstance(objects, list) or type(object_id) not in (int, float) or int(object_id) != object_id or object_id < 0:
            raise ValueError('Use an object list and a nonnegative integer object ID')
        oid = int(object_id)
        x, y = (int(value) for value in evt.index)
        if not 0 <= x < canvas.width or not 0 <= y < canvas.height:
            raise ValueError('Prompt point is outside the image')
        obj = next((obj for obj in objects if isinstance(obj, dict) and obj.get('id') == oid), None)
        if obj is None:
            obj = dict(id=oid, points=[])
            objects.append(obj)
        obj.setdefault('points', []).append([x, y, 1 if label == 'Foreground' else 0])
        from .methods.spgrasp_data import Letterbox, prepare_prompts
        prepare_prompts(objects, Letterbox(canvas.height, canvas.width))
        preview = canvas.copy()
        draw = ImageDraw.Draw(preview)
        color = '#39d98a' if label == 'Foreground' else '#ff6767'
        draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline='white', width=2)
        draw.text((x+8, y-8), str(oid), fill=color)
        return json.dumps(objects, indent=2), preview
    except (ValueError, TypeError, AttributeError) as error:
        raise gr.Error(str(error)) from error


def create_app(manager=None):
    manager = manager or JobManager()
    items = {mid: item for mid, item in catalogue().items() if capabilities(mid)}
    groups = ["All"] + sorted({m["group"] for m in items.values()})
    first = Experiment(method="graspnet_baseline",action="infer",dataset_root=default_dataset()).to_dict()
    cp = ROOT / "checkpoints/graspnet_baseline/checkpoint-rs.tar"
    if cp.exists():
        first["checkpoint"] = str(cp)

    def filter_methods(group):
        ids = [m for m, v in items.items() if group == "All" or v["group"] == group]
        return gr.Dropdown(choices=[(items[mid].get('name', mid), mid) for mid in ids], value=ids[0])

    def checkpoint_for(method,camera):
        return primary(method,camera)

    def select_method(method,camera):
        actions = capabilities(method)
        selected=preset(method,default_dataset()) if actions else None
        return method_card(method), gr.Dropdown(choices=action_choices(method), value=selected.action if selected else None), gr.Button(interactive=bool(actions)), selected.checkpoint if selected else '', selected.camera if selected else camera, selected.workspace if selected else 'official_gt_workspace', selected.num_points if selected else 15000

    def download_checkpoint(method,camera,progress=gr.Progress()):
        if method == 'gtg2':
            return '', 'Train an ensemble after preparing graph inputs. This reconstruction has no registered pretrained weights; see Guide → Methods & papers.'
        try:
            progress(0,desc='Downloading / verifying author weights…')
            value=fetch(method,camera,lambda message: progress(.5,desc=message))
            progress(1,desc='Verified')
            return value, ('SAM2 initialization downloaded. Train SPGrasp first; this file is not a trained grasp checkpoint.' if method == 'spgrasp' else 'Weights downloaded and verified. Set your inputs, then run the experiment.')
        except Exception as error:
            raise gr.Error(str(error)) from error

    def download_component_weights(method, backbone, parameters, progress=gr.Progress()):
        from .components import validate_selection
        from .weights import fetch_component
        try:
            if backbone not in ('dinov2','dinov3','utonia','concerto','mambavision','efficientvit','fastvit'):
                return 'Select a registered pretrained encoder to prepare its weights.'
            parameters = json.loads(parameters or '{}')
            if not isinstance(parameters, dict):
                raise ValueError('Component parameters must be a mapping keyed by slot')
            options = parameters.get('backbone', {})
            if not isinstance(options, dict) or 'type' in options:
                raise ValueError('Use the encoder selector for type and provide its parameters as a mapping')
            validate_selection(method, {'backbone': dict(type=backbone, **options)})
            if backbone == 'fastvit':
                from .modules.fastvit_options import resolve
                options = resolve(options)
            if backbone == 'efficientvit':
                from .modules.efficientvit_options import resolve
                options = resolve(options)
            if not options.get('pretrained', True): return 'This configuration uses random encoder initialization.'
            defaults = {'concerto': 'base', 'mambavision': 'tiny', 'efficientvit': 'b0'}
            name = 'utonia' if backbone == 'utonia' else backbone+'_'+options.get('variant', defaults.get(backbone, 'small'))
            if backbone == 'fastvit':
                from .modules.fastvit_options import weight_id
                name = weight_id(options)
            fetch_component(name, lambda message: progress(.5, desc=message))
            return 'Pretrained encoder weights verified. New projection layers still require grasp training.'
        except Exception as error:
            raise gr.Error(str(error)) from error

    def apply_preset(method,dataset):
        config=preset(method,(dataset or '').strip())
        if not capabilities(method): raise gr.Error('This source has no runnable adapter. See the method card.')
        # Apply the complete preset in one response, so delayed reset callbacks
        # cannot overwrite a composition edited after the preset has loaded.
        return (gr.update(choices=action_choices(method), value=config.action), config.camera, config.checkpoint, config.workspace, config.num_points, config.split, config.scene, config.frame, config.frames, config.seed, config.epochs, config.batch_size, config.learning_rate, config.label_root, config.timeout_minutes, json.dumps(config.to_dict(),indent=2), config.collision_thresh,
                'upstream', gr.update(value='upstream', visible=any(s.name=='crop' for s in slots(method))),
                'strict', '{}', '{}', '{}', 'upstream', '{}', 'upstream', '{}',
                '{}', 'upstream', 'upstream', '[]', '{}', 0, config.training_steps,
                'initialize', 0, 0, 0, None, 'Preset ready. Configure your experiment and run.', 'upstream', 'none', '{}')

    def setup_check(dataset):
        import torch
        root=Path((dataset or '').strip()) if (dataset or '').strip() else None
        cameras={c:bool(root and (root/'scenes/scene_0100'/c/'depth/0000.png').exists()) for c in ('realsense','kinect')}
        return {'python':sys.version.split()[0], 'torch':torch.__version__, 'cuda_available':torch.cuda.is_available(),
                'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'scene_0100_depth':cameras, 'next_step':'Select method → Load preset → Download weights → Run current form.'}

    def compose(method, action, dataset, checkpoint, camera, split, scene, frame, count, points, seed, workspace, collision, epochs, batch, lr, predictions, gpu, dataset_key="graspnet1b", backbone="upstream", crop="upstream", checkpoint_policy="strict", training_steps=3, label_root="", train_checkpoint_mode='initialize', train_batch_limit=0, eval_batch_limit=0, data_workers=0, component_options='{}', loss_options='{}', augmentation_options='{}', optimizer_kind='upstream', optimizer_options='{}', scheduler_kind='upstream', scheduler_options='{}', proposal_warmup_steps=0, trainer_options='{}', timeout_minutes=60, head='upstream', memory='upstream', prompt_options='[]', planar_options='{}', sampling='upstream', refinement_kind='none', refinement_options='{}'):
        def mapping(text, label):
            try:
                value=json.loads(text or '{}')
            except (ValueError, TypeError) as error:
                raise gr.Error(f'{label} must contain valid JSON') from error
            if not isinstance(value,dict):raise gr.Error(f'{label} must be a mapping')
            return value
        refinement=mapping(refinement_options,'Refinement parameters')
        if 'type' in refinement:raise gr.Error('Choose the refinement type with its selector')
        if refinement_kind=='none' and refinement:raise gr.Error('Select contact-score refinement before setting its parameters')
        refinement={'type':refinement_kind,**refinement} if refinement_kind!='none' else {}
        parameters=mapping(component_options,'Component parameters')
        loss=mapping(loss_options,'Loss configuration')
        augmentation=mapping(augmentation_options,'Augmentation configuration')
        optimizer=mapping(optimizer_options,'Optimizer parameters')
        scheduler=mapping(scheduler_options,'Scheduler parameters')
        trainer=mapping(trainer_options,'Trainer parameters')
        planar=mapping(planar_options,'Planar settings')
        try:
            prompts=json.loads(prompt_options or '[]')
        except (ValueError, TypeError) as error:
            raise gr.Error('First-frame prompts must contain valid JSON') from error
        if not isinstance(prompts,list): raise gr.Error('First-frame prompts must be an object list')
        for kind, options, name in ((optimizer_kind, optimizer, 'optimizer'),(scheduler_kind, scheduler, 'scheduler')):
            if 'type' in options:raise gr.Error(f'Choose the {name} type with its selector')
            if kind=='upstream' and options:raise gr.Error(f'Select a custom {name} before setting its parameters')
        optimizer={'type':optimizer_kind,**optimizer} if optimizer_kind!='upstream' else {}
        scheduler={'type':scheduler_kind,**scheduler} if scheduler_kind!='upstream' else {}
        if action == 'recipe':
            if refinement or parameters or loss or augmentation or optimizer or scheduler or trainer or proposal_warmup_steps or backbone!='upstream' or crop!='upstream' or head!='upstream' or memory!='upstream' or sampling!='upstream' or prompts or planar:
                raise gr.Error('The native recipe uses fixed components and settings. Load its preset to reset overrides.')
            from dataclasses import replace
            config=replace(preset(method,(dataset or '').strip()),gpu=int(gpu),timeout_minutes=int(timeout_minutes))
            if method in CHECKPOINT_RECIPES:config=replace(config,checkpoint=(checkpoint or '').strip())
            return json.dumps(config.to_dict(),indent=2)
        selected={s.name:{"backbone":backbone,"crop":crop,"head":head,"memory":memory,"sampling":sampling}[s.name] for s in slots(method)}
        selection={name:choice for name,choice in selected.items() if choice != 'upstream'}
        for name,values in parameters.items():
            if name not in selected:raise gr.Error(f'This method has no configurable {name} slot')
            if not isinstance(values,dict) or 'type' in values:raise gr.Error('Use the component selector for type; supply only its parameters here')
            selection[name]={'type':selected[name],**values}
        config = Experiment(refinement=refinement,prompts=prompts,planar=planar,timeout_minutes=int(timeout_minutes),trainer=trainer,proposal_warmup_steps=int(proposal_warmup_steps),dataset=dataset_key,modules=selection,loss=loss,augmentation=augmentation,optimizer=optimizer,scheduler=scheduler,checkpoint_policy=checkpoint_policy,training_steps=int(training_steps),label_root=(label_root or '').strip(),method=method, action=action or "infer", dataset_root=(dataset or '').strip(), checkpoint=(checkpoint or '').strip(),
                            camera=camera, split=split, scene=int(scene), frame=int(frame), frames=int(count),
                            num_points=int(points), seed=int(seed), workspace=workspace, collision_thresh=collision,
                            epochs=int(epochs), batch_size=int(batch), learning_rate=lr,
                            prediction_dir=(predictions or '').strip(), gpu=int(gpu),train_checkpoint_mode=train_checkpoint_mode,
                            train_batch_limit=int(train_batch_limit),eval_batch_limit=int(eval_batch_limit),data_workers=int(data_workers))
        try:config.validate()
        except ValueError as error:raise gr.Error(str(error)) from error
        return json.dumps(config.to_dict(), indent=2)

    def preflight(text):
        try:
            config = Experiment.from_dict(json.loads(text)).preflight()
            return f"Configuration accepted: **{config.method} / {config.action}**. Ready to queue."
        except Exception as error:
            return f"Configuration blocked: {error}"

    def submit(text):
        try:
            config = Experiment.from_dict(json.loads(text))
            job_id = manager.submit(config)
            return job_id, f"Queued **{job_id}** · {config.method} / {config.action}"
        except Exception as error:
            raise gr.Error(str(error)) from error

    def submit_form(*values):
        text=compose(*values)
        job,message=submit(text)
        return job,message,text

    def sweep_config(text, grid):
        from .sweeps import Sweep
        try:
            return Sweep.from_dict(dict(base=json.loads(text), grid=json.loads(grid)))
        except (ValueError, TypeError) as error:
            raise gr.Error(str(error)) from error

    def preview_sweep(text, grid):
        return sweep_config(text, grid).preview()

    def run_sweep(text, grid):
        try:
            specification = sweep_config(text, grid)
            ids = manager.submit_sweep(specification.to_dict())
            return ids[0], 'Queued sweep. Inspect each experiment in **Runs & results**.\n\n' + '\n'.join(f'- `{job}`' for job in ids)
        except Exception as error:
            raise gr.Error(str(error)) from error

    def job_rows():
        return [[j["id"], j["state"], j["config"]["method"], j["config"]["action"], j["config"]["camera"], j["config"]["split"], j["detail"]] for j in manager.list()]

    def inspect(job_id):
        row = manager.get(job_id)
        if not row:
            return "Select or submit an experiment.", {}, None, []
        directory = manager.root / row["id"]
        log = directory / "experiment.log"
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 32000))
            text = stream.read().decode(errors="replace")
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"state": row["state"], "detail": row["detail"]}
        preview = directory / "preview.png"
        files = [str(p) for p in directory.iterdir() if p.suffix in (".json", ".png", ".log")]
        return text, result, str(preview) if preview.exists() else None, files

    def training_curve(job_id):
        import pandas as pd
        row=manager.get(job_id)
        path=manager.root/row['id']/'result.json' if row else None
        result=json.loads(path.read_text()) if path and path.exists() else {}
        values=result.get('losses',[])
        stages=[v.get('stage','Training') for v in values]
        counts={};batches=[]
        for stage in stages:
            counts[stage]=counts.get(stage,0)+1;batches.append(counts[stage])
        return pd.DataFrame({'Batch':batches,'Loss':[v['total'] for v in values],'Stage':stages})

    def reuse_checkpoint(job_id):
        from dataclasses import replace
        row=manager.get(job_id)
        if not row or row['state']!='succeeded':raise gr.Error('Choose a completed training run first.')
        method=row['config']['method']
        if 'infer' not in capabilities(method) and method not in CHECKPOINT_RECIPES:
            raise gr.Error('This method does not accept a trained checkpoint through the inference form. See Guide → Methods & papers for its supported recipe.')
        path=manager.root/row['id']/'checkpoint.pt'
        if not path.exists():raise gr.Error('This run has no saved training checkpoint.')
        config=replace(Experiment.from_dict(row['config']),action='infer',checkpoint=str(path),
                       checkpoint_policy='strict',split='test_seen',scene=100,frame=0,frames=1,loss={},augmentation={},optimizer={},scheduler={},trainer={},proposal_warmup_steps=0,train_checkpoint_mode='initialize')
        if method=='finegrasp': config=replace(config,workspace='native_demo')
        if method=='generalizing_grasp':config=replace(config,workspace='fused_scene',frame=0,frames=1)
        if method=='spgrasp': config=replace(config,frames=row['config']['frames'],prompts=[])
        if 'infer' not in capabilities(method):
            config=replace(preset(method,row['config']['dataset_root']),checkpoint=str(path),gpu=row['config']['gpu'])
        config.validate()
        note='The fixed native input recipe is retained.' if config.action=='recipe' else 'Review the selected test frame.'
        if method=='spgrasp': note='Set first-frame prompts in the generated JSON before running; use the Planar sequence panel to inspect your selected frame and create prompts.'
        return json.dumps(config.to_dict(),indent=2), 'Inference JSON prepared in Experiments. '+note+' Click Run edited JSON.'

    def cancel(job_id):
        try:
            return manager.cancel(job_id)
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def export(job_id):
        row = manager.get(job_id)
        if not row:
            raise gr.Error("Choose an existing experiment")
        directory = manager.root / row["id"]
        destination = manager.root / (row["id"] + ".zip")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in directory.rglob("*"):
                if file.is_file() and not file.is_symlink():
                    archive.write(file, file.relative_to(directory))
        return str(destination)

    def compare():
        rows = []
        for job in manager.list():
            path = manager.root / job["id"] / "result.json"
            if job["state"] != "succeeded" or not path.exists():
                continue
            data, config = json.loads(path.read_text()), job["config"]
            losses = data.get('losses', [])
            last_loss = losses[-1].get('total') if isinstance(losses, list) and losses and isinstance(losses[-1], dict) else None
            settings = {key: config.get(key) for key in ('learning_rate', 'optimizer', 'scheduler', 'loss', 'augmentation', 'trainer', 'proposal_warmup_steps')}
            rows.append([job["id"], config.get('dataset','graspnet1b'),config["method"],json.dumps(config.get('modules',{})), json.dumps(config.get('refinement',{})), data.get("stage"), config["camera"], config["split"],
                         config["workspace"], config["seed"], json.dumps(settings), last_loss, (len(data["frames"]) if isinstance(data.get("frames"),list) else data.get("frames", "recipe")), json.dumps(data.get("ap"))])
        return rows

    with gr.Blocks(title="GraspPanda · Modular visual grasping") as app:
        gr.HTML('<div id="panda-hero"><h1>🐼 GraspPanda</h1><p>Visual grasping experiments, from method presets to custom components.</p></div>')
        with gr.Tab("Experiments"):
            with gr.Row():
                with gr.Column(scale=4):
                    dataset_key=gr.Dropdown(choices=[(s.title,key) for key,s in datasets().items()],value="graspnet1b",label="Dataset")
                    group = gr.Dropdown(groups, value="All", label="Protocol & family")
                    method = gr.Dropdown([(v.get("name", m), m) for m, v in items.items()], value="graspnet_baseline", label="Method")
                    with gr.Accordion("Method details & input requirements", open=False):
                        card = gr.Markdown(method_card("graspnet_baseline"))
                    action = gr.Dropdown(action_choices("graspnet_baseline"), value="infer", label="Operation")
                    preset_button=gr.Button("Load preset",variant="primary")
                    preset_status=gr.Markdown()
                with gr.Column(scale=6):
                    dataset = gr.Textbox(first["dataset_root"], label="Dataset root on server", placeholder="/data/GraspNet-1B")
                    checkpoint = gr.Textbox(first["checkpoint"], label="Checkpoint on server")
                    download = gr.Button('Download registered weights')
                    download_message = gr.Markdown()
                    with gr.Row():
                        camera = gr.Dropdown(["realsense", "kinect"], value="realsense", label="Camera")
                        split = gr.Dropdown(["test_seen", "test_similar", "test_novel", "train"], value="test_seen", label="Split")
                        gpu = gr.Number(0, precision=0, label="GPU index", minimum=0)
                    with gr.Accordion("Input & preprocessing", open=False):
                        with gr.Row():
                            scene = gr.Number(100, precision=0, label="First scene", minimum=0, maximum=189)
                            frame = gr.Number(0, precision=0, label="First frame", minimum=0, maximum=255)
                            count = gr.Number(1, precision=0, label="Frame count", minimum=1)
                        with gr.Row():
                            points = gr.Number(15000, precision=0, label="Sampled points")
                            seed = gr.Number(0, precision=0, label="Seed")
                        workspace = gr.Dropdown(["official_gt_workspace", "depth_only", "native_demo", "fused_gt_workspace", "fused_scene"], value="official_gt_workspace", label="Workspace policy")
                        gr.Markdown("Official workspace uses dataset segmentation + poses. HGGD / RNG use native_demo. Load the preset to select the correct protocol.", elem_classes="panda-note")
                        collision = gr.Number(0.01, label="Collision threshold (0 disables)")
                    with gr.Accordion('Planar sequence', open=False, visible=False) as planar_panel:
                        gr.Markdown('SPGrasp predicts planar grasps from RGB sequences. Train with the SAM2 initializer, then select your trained checkpoint for prediction. Settings and units are explained in Guide → Modules.')
                        planar_options = gr.Code('{}', language='json', label='Planar settings', lines=4)
                        with gr.Column(visible=False) as prompt_panel:
                            gr.Markdown('Load the first frame and click foreground/background points for each object ID. Coordinates use original RGB pixels. You can also enter boxes as `{"id": 1, "box": [x0, y0, x1, y1]}`. Reloading or changing the frame clears prompts.')
                            load_prompt_frame = gr.Button('Load first RGB frame / clear points')
                            prompt_image = gr.Image(type='pil', interactive=False, label='Click to add a prompt point', height=350, buttons=['download','fullscreen'])
                            with gr.Row():
                                prompt_object = gr.Number(1, precision=0, minimum=0, label='Prompt object ID')
                                prompt_label = gr.Radio(['Foreground', 'Background'], value='Foreground', label='Point label')
                            prompt_options = gr.Code('[]', language='json', label='First-frame object prompts', lines=6)
                    with gr.Accordion("Compose modules",open=False) as composition_panel:
                        composition_intro = gr.Markdown("Select compatible building blocks. **reuse_unchanged** initializes replaced components and retains only unchanged checkpoint modules. Train the replaced components before using their predictions.")
                        initial_components={slot.name:list(slot.choices) for slot in slots('graspnet_baseline')}
                        backbone=gr.Dropdown(initial_components['backbone'],value='upstream',label='Point encoder')
                        with gr.Column() as crop_panel:
                            crop=gr.Dropdown(initial_components['crop'],value='upstream',label='Local cylindrical grouping')
                        head=gr.Dropdown(initial_components.get('head', ['upstream']),value='upstream',label='Grasp prediction head',visible='head' in initial_components)
                        memory=gr.Dropdown(['upstream'],value='upstream',label='Temporal memory',visible=False)
                        sampling=gr.Dropdown(['upstream','object_balanced'],value='upstream',label='Grasp seed sampling',visible=False)
                        with gr.Accordion('Object-balanced sampling inputs', open=False, visible=False) as obs_panel:
                            gr.Markdown('OBS predicts objects with an independent DSN network, then allocates grasp seeds per predicted object. Configure sampling parameters in the component editor. The registered segmentation weights use RealSense; a custom checkpoint is required for Kinect. See Guide → Modules.')
                            obs_download=gr.Button('Prepare OBS segmentation weights')
                            obs_message=gr.Markdown()
                        checkpoint_policy=gr.Dropdown(['strict','reuse_unchanged'],value='strict',label='Checkpoint policy')
                        component_contract=gr.Markdown('Baseline: 256-channel seed features, original point indices, four depth bins.')
                        component_options=gr.Code('{}',language='json',label='Component parameters by slot',lines=5)
                        composition_hint = gr.Markdown('Enter parameters keyed by slot, for example `{"backbone": {"embed_dim": 32}}` for PointMLP. For compatible methods, `{"crop": {"seed_interaction": "gaussian"}}` adds seed interaction to the selected grouping, including `upstream`. The selectors supply each component type.')
                        with gr.Accordion('Pretrained encoder weights', open=False, visible=False) as pretraining_panel:
                            gr.Markdown('Registered image and point encoders use author pretraining by default; `pretrained: false` selects random weights. The selected encoder’s parameter guide explains freezing and fine-tuning controls. Initial training prepares missing weights locally; strict grasp-checkpoint loading does not fetch or reapply pretraining.')
                            component_download = gr.Button('Prepare selected component weights')
                            component_download_message = gr.Markdown()
                        with gr.Accordion('Available component parameters', open=False):
                            parameter_help = gr.Markdown(component_parameters('graspnet_baseline', 'upstream', 'upstream'))
                    with gr.Accordion('Refine grasps', open=False) as refinement_panel:
                        gr.Markdown('Optimize predicted poses with frozen contact and score networks. Instances come from visual DSN predictions. Table-frame refinement uses camera calibration. [Parameters and protocol](https://github.com/Daeda1used/GraspPanda/blob/main/docs/REFERENCE.md#contact-score-refinement).')
                        refinement_kind=gr.Dropdown(['none','contact_score'],value='none',label='Pose refinement')
                        refinement_options=gr.Code('{}',language='json',label='Refinement parameters',lines=5,interactive=False)
                        refinement_download=gr.Button('Prepare refinement networks')
                        refinement_message=gr.Markdown()
                    with gr.Accordion("Training settings", open=False, visible=False) as training_panel:
                        training_steps=gr.Number(3,precision=0,minimum=1,maximum=1000,visible=False,label='Optimizer steps (short training)')
                        proposal_warmup_steps=gr.Number(0,precision=0,minimum=0,maximum=10000,interactive=False,visible=False,label='Proposal warmup updates',info='Optional real-label seed training (Graspness, EconomicGrasp, FineGrasp) or anchor training (RNG) before grasp training. Added to short-training updates; 0 preserves the native preset.')
                        with gr.Accordion('Choose loss formulations', open=False):
                            from grasppanda.training.losses import CLASSIFICATION, REGRESSION
                            with gr.Row():
                                classification_loss=gr.Dropdown(['upstream', *CLASSIFICATION],value='upstream',label='Classification loss',interactive=False)
                                regression_loss=gr.Dropdown(['upstream', *REGRESSION],value='upstream',label='Regression loss',interactive=False)
                            from .training.quality import LOSSES as QUALITY_LOSSES
                            quality_loss=gr.Dropdown(['upstream', *QUALITY_LOSSES],value='upstream',label='Quality score objective',visible=False,
                                info='Overrides score only. Upstream uses the regression selection. Requires the quality_residual head.')
                            apply_loss=gr.Button('Apply loss choices',interactive=False)
                            gr.Markdown('Apply writes the selected formulation to each matching term below and keeps your coefficients. Edit individual terms and parameters in Loss configuration. Training operations only.')
                            with gr.Accordion('Loss parameters & augmentation guide', open=False):
                                loss_help=gr.Markdown(loss_parameters('graspnet_baseline'))
                        loss_options=gr.Code('{}',language='json',label='Loss configuration',lines=5)
                        with gr.Accordion('Choose observation sampling', open=False, visible=False) as sampling_panel:
                            sampling_rule=gr.Dropdown([('Uniform retained rows','uniform'),('Density-weighted (PointSP)','pointsp_wrs'),('Local/global removal (PointSP)','pointsp_lgd'),('Remove sampling override','none')],value='pointsp_wrs',label='Sampling rule')
                            with gr.Row():
                                sampling_min=gr.Number(.5,minimum=.1,maximum=1,label='Minimum keep ratio')
                                sampling_max=gr.Number(1.,minimum=.1,maximum=1,label='Maximum keep ratio')
                            apply_sampling=gr.Button('Apply sampling choices')
                            gr.Markdown('Equal limits give a fixed ratio; otherwise each sample draws a log-uniform ratio. At least 1,024 rows remain. Apply switches to custom point augmentation and writes the rule below, preserving other transforms. Advanced parameters stay editable in JSON. Removing the override keeps the selected augmentation mode.')
                        with gr.Accordion('Choose RGB photometric policy', open=False, visible=False) as prime_panel:
                            prime_policy=gr.Dropdown([('PRIME color + filter','color_filter'),('PRIME smooth color','color'),('PRIME random filter','filter'),('Remove PRIME override','none')],value='color_filter',label='Photometric policy')
                            apply_prime=gr.Button('Apply photometric policy')
                            gr.Markdown('Apply adds PRIME photometric augmentation to the JSON below and preserves other transforms. Edit mixture_width, mixture_depth, probability, color_temperature and filter_sigma in that mapping. The policy changes RGB appearance with fixed camera geometry and grasp labels; it requires no extra weights.')
                        augmentation_options=gr.Code('{}',language='json',label='Augmentation configuration',lines=3)
                        with gr.Accordion('Method training stages', open=False, visible=False) as trainer_panel:
                            trainer_options=gr.Code('{}',language='json',label='Trainer parameters',lines=4,interactive=False)
                            trainer_help = gr.Markdown('Configure HGGD stages, gradient accumulation and local sampling. See [HGGD epoch training](https://github.com/Daeda1used/GraspPanda/blob/main/docs/REFERENCE.md#hggd-epoch-training) for parameters and defaults.')
                        with gr.Accordion('Optimizer & learning-rate schedule', open=False):
                            optimizer_kind=gr.Dropdown(['upstream'],value='upstream',label='Optimizer',interactive=False)
                            optimizer_options=gr.Code('{}',language='json',label='Optimizer parameters',lines=3)
                            scheduler_kind=gr.Dropdown(['upstream'],value='upstream',label='Learning-rate schedule',interactive=False)
                            scheduler_options=gr.Code('{}',language='json',label='Scheduler parameters',lines=3)
                            gr.Markdown('The learning-rate field below sets the base rate. Scheduler warmup and milestones count **optimizer updates**, not epochs. Examples: optimizer `{"weight_decay": 0.01}`; cosine schedule `{"warmup_steps": 1, "min_lr_ratio": 0.01}`. Omit parameters to use the selected implementation defaults; see Guide → Modules for supported settings.')
                        label_root=gr.Textbox(label='Prepared targets / cache root (optional)',placeholder='HGGD/RNG: preprocessed labels. FineGrasp: writable derived-input cache.')
                        gr.Markdown('Short training repeats a labelled sample with the selected method’s batch rules. See Guide → Modules for sampling and training controls.')
                        with gr.Row():
                            epochs = gr.Number(1, precision=0, visible=False, label="Final epoch (must exceed resume epoch)")
                            batch = gr.Number(2, precision=0, label="Batch size")
                            lr = gr.Number(0.001, label="Learning rate")
                        train_checkpoint_mode=gr.Dropdown(['initialize','resume'],value='initialize',label='Epoch training checkpoint mode',interactive=False,visible=False,info='Initialize loads model weights with a fresh optimizer. Resume restores model, optimizer and epoch; requires strict loading.')
                        with gr.Row(visible=False) as epoch_panel:
                            train_batch_limit=gr.Number(0,precision=0,minimum=0,label='Training batches per epoch (0 = full split)')
                            eval_batch_limit=gr.Number(0,precision=0,minimum=0,label='Validation batches per epoch (0 = native range)')
                            data_workers=gr.Number(0,precision=0,minimum=0,maximum=32,label='Data loader workers')
                        epoch_help = gr.Markdown('Set batch limits to 0 for the native data range. Validation follows the selected method’s protocol; see Guide → Modules.', visible=False)
                    predictions = gr.Textbox(label="Complete predictions directory for evaluation", visible=False)
                    with gr.Accordion('Run settings', open=False):
                        timeout = gr.Number(60, precision=0, minimum=1, maximum=43200, label='Run time limit (minutes)')
            run_form=gr.Button('Run current form',variant='primary')
            with gr.Accordion("Configuration editor", open=False):
                gr.Markdown("Generate JSON from the form, edit it, then use **Run edited JSON** to submit that exact configuration.")
                generate = gr.Button("Generate configuration")
                config_text = gr.Code(json.dumps(first, indent=2), language="json", label="Experiment configuration (editable)", lines=14)
                with gr.Row():
                    check = gr.Button("Validate JSON")
                    run = gr.Button("Run edited JSON", variant="primary")
                with gr.Accordion('Configuration sweep', open=False):
                    gr.Markdown('Use the **Experiment configuration** above as the base. Enter lists of values by dotted path, such as `seed`, `learning_rate` or `modules.backbone.embed_dim`. Every combination is validated before queueing. The sweep uses the editor, so generate the configuration after changing form fields.')
                    sweep_grid = gr.Code('{"seed": [0, 1]}', language='json', label='Parameter grid', lines=5)
                    with gr.Row():
                        sweep_preview_button = gr.Button('Preview sweep')
                        sweep_run_button = gr.Button('Run sweep')
                    sweep_preview = gr.JSON(label='Exact experiment configurations')
            message = gr.Markdown()
        with gr.Tab("Runs & results"):
            refresh = gr.Button("Refresh runs")
            table = gr.Dataframe(headers=["ID", "Status", "Method", "Action", "Camera", "Split", "Details"], value=job_rows(), interactive=False, type="array")
            job_id = gr.Textbox(label="Experiment ID (click a table row, or paste ID)")
            with gr.Row():
                inspect_button = gr.Button("Load experiment")
                stop = gr.Button("Cancel experiment", variant="stop")
                export_button = gr.Button("Export experiment ZIP")
                reuse_button = gr.Button("Prepare inference from checkpoint")
            job_message = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    loss_plot=gr.LinePlot(x='Batch',y='Loss',color='Stage',title='Training loss by stage',label='Training objective')
                with gr.Column():
                    preview = gr.Image(label="Predicted gripper projection (first frame)", interactive=False)
                    bundle = gr.File(label="Exported experiment")
            with gr.Accordion("Run details & logs", open=False):
                logs = gr.Textbox(label="Live log (last 32 KB)", lines=16, interactive=False)
                result = gr.JSON(label="Result")
                artifacts = gr.File(label="Configuration, provenance & logs", file_count="multiple")
        with gr.Tab("Compare"):
            gr.Markdown("Compare completed experiments and their saved results.")
            with gr.Accordion("Metric definitions & comparison settings", open=False):
                gr.Markdown("Compare AP under identical dataset, camera, split, workspace, training data and postprocessing. `null` AP means not evaluated. Compare losses only when objectives, coefficients and sampled data match.")
            compare_button = gr.Button("Refresh comparison")
            comparisons = gr.Dataframe(headers=["ID", "Dataset", "Method", "Modules", "Refinement", "Stage", "Camera", "Split", "Workspace", "Seed", "Training settings", "Final loss", "Frames", "AP"], interactive=False)
        with gr.Tab("Guide"):
            gr.Markdown("""### Start an experiment
1. Choose a method in **Experiments**, then **Load preset**.
2. Set the dataset root containing `scenes/`. **Download registered weights** prepares the required networks.
3. Choose an operation and **Run current form**. View predictions, checkpoints and logs in **Runs & results**.

**No dataset yet?** Select ASGrasp, load its preset and download its weights to run the author's stereo sample.

For component experiments, expand **Compose modules**. Full configuration editing is under **Configuration editor**.
""")
            with gr.Tabs():
                for title, filename in (('Install', 'INSTALL.md'), ('Downloads', 'DOWNLOADS.md'),
                                        ('Usage', 'USAGE.md'), ('Modules', 'MODULES.md'),
                                        ('Methods & papers', 'METHODS.md')):
                    with gr.Tab(title) as guide_tab:
                        guide_text = gr.Markdown()
                        guide_tab.select(partial(documentation, filename), outputs=guide_text,
                                         api_name=False, queue=False)
                        if title == 'Install':
                            app.load(partial(documentation, filename), outputs=guide_text,
                                          api_name=False, queue=False)
                        if title == 'Modules':
                            with gr.Accordion('Detailed component and training reference', open=False) as reference:
                                reference_text = gr.Markdown()
                            reference.expand(partial(documentation, 'REFERENCE.md'), outputs=reference_text,
                                             api_name=False, queue=False)
            with gr.Accordion("Check data and GPU", open=False):
                setup_root=gr.Textbox(default_dataset(),label='Dataset root')
                setup_button=gr.Button('Check data & GPU')
                setup_result=gr.JSON(label='Readiness')
                setup_button.click(setup_check,setup_root,setup_result,api_name='readiness')
        # Draft form changes may carry choices from the previous method.
        # Skip dropdown preprocessing for render callbacks; submissions still validate.
        def select_components(method):
            enabled=bool(slots(method))
            contract='\n\n'.join(f"**{s.name}**: {s.input_contract} → {s.output_contract}" for s in slots(method)) if enabled else 'This method currently retains its native components. No interchangeable slots are registered.'
            choices={s.name:list(s.choices) for s in slots(method)}
            return gr.update(choices=choices.get('backbone',['upstream']),value='upstream',interactive='backbone' in choices,label='Image encoder' if method in ('hggd','region_normalized_grasp','spgrasp') else 'Graph encoder' if method == 'gtg2' else 'Point encoder'),gr.update(choices=choices.get('crop',['upstream']),value='upstream',interactive='crop' in choices,visible='crop' in choices),contract,gr.update(choices=choices.get('head',['upstream']),value='upstream',visible='head' in choices,interactive='head' in choices),gr.update(choices=choices.get('memory',['upstream']),value='upstream',visible='memory' in choices,interactive='memory' in choices),gr.update(value='upstream')
        method.input(select_components,method,[backbone,crop,component_contract,head,memory,sampling],api_name='select_components', preprocess=False, queue=False).then(
            lambda: ('{}','{}','{}','strict'),outputs=[component_options,loss_options,augmentation_options,checkpoint_policy],api_name=False, queue=False)
        method.change(lambda m: gr.update(choices=['strict'] if m=='spgrasp' else ['strict','reuse_unchanged'], value='strict'),
            method, checkpoint_policy, api_name=False, preprocess=False)
        method.change(lambda m: 'Configure the Hiera encoder and temporal memory, then train from the SAM2 initializer. A trained SPGrasp checkpoint loads strictly and carries its architecture and width units.' if m=='spgrasp' else 'Select compatible building blocks. **reuse_unchanged** initializes replaced components and retains only unchanged checkpoint modules. Train the replaced components before using their predictions.',
            method, composition_intro, api_name=False, preprocess=False)
        method.change(lambda m: 'Select **hiera** and **temporal** to edit their parameters. For example: `{"backbone": {"resolution": 256}, "memory": {"frames": 3, "layers": 2}}`.' if m=='spgrasp' else 'Enter parameters keyed by slot, for example `{"backbone": {"embed_dim": 32}}` for PointMLP. For compatible methods, `{"crop": {"seed_interaction": "gaussian"}}` adds seed interaction to the selected grouping, including `upstream`. The selectors supply each component type.',
            method, composition_hint, api_name=False, preprocess=False)
        apply_loss.click(loss_preset,[method,classification_loss,regression_loss,loss_options,head,quality_loss],loss_options,api_name='apply_loss_choices')
        def quality_controls(method, action, head):
            from .training.quality import METHODS
            return gr.update(value='upstream',visible=method in METHODS and action in ('train','train_short') and head=='quality_residual')
        gr.on([method.change,action.change,head.change],quality_controls,[method,action,head],quality_loss,api_name=False,preprocess=False,queue=False,trigger_mode='always_last')
        apply_sampling.click(sampling_preset,[method,action,sampling_rule,sampling_min,sampling_max,augmentation_options],augmentation_options,api_name='apply_sampling_choices')
        apply_prime.click(prime_preset,[method,action,prime_policy,augmentation_options],augmentation_options,api_name='apply_photometric_policy')
        gr.on([method.change,action.change],
              lambda m,a: gr.update(visible=m in ('hggd','region_normalized_grasp') and a in ('train','train_short')),
              [method,action],prime_panel,api_name=False,preprocess=False)
        for selector in (method, action):
            selector.change(lambda m,a: gr.update(visible=m in ('graspnet_baseline','pointnet2_upgrade','scale_balanced_grasp','graspness','economicgrasp','finegrasp') and a in ('train','train_short')),
                [method,action],sampling_panel,api_name=False,preprocess=False)
        method.change(loss_parameters,method,loss_help,api_name='loss_parameters', preprocess=False)
        def loss_controls(method, action):
            from grasppanda.training.options import METHODS
            from grasppanda.training.losses import classification_choices
            enabled=method in METHODS and action in ('train','train_short')
            return gr.update(choices=['upstream', *classification_choices(method)],value='upstream',interactive=enabled and method != 'gtg2'),gr.update(value='upstream',interactive=enabled),gr.update(interactive=enabled)
        for selector in (method, action):
            selector.change(loss_controls,[method,action],[classification_loss,regression_loss,apply_loss],api_name=False, preprocess=False)
        for selector in (method, backbone, crop, head, memory, sampling):
            selector.change(component_parameters,[method,backbone,crop,head,memory,sampling],parameter_help,api_name=False, preprocess=False)
        def optimization_choices(method, action, backbone):
            from grasppanda.training.optimization import METHODS,MUON_METHODS,SPARSE_BACKBONES
            enabled=method in METHODS and action in ('train','train_short')
            optimizers=['upstream','adam','adamw','sgd','lion']+(['muon'] if method in MUON_METHODS and backbone not in SPARSE_BACKBONES else [])
            return gr.update(choices=optimizers if enabled else ['upstream'],value='upstream',interactive=enabled),gr.update(choices=['upstream','constant','cosine','multistep'] if enabled else ['upstream'],value='upstream',interactive=enabled),'{}','{}'
        for selector in (method, action, backbone):
            selector.change(optimization_choices,[method,action,backbone],[optimizer_kind,scheduler_kind,optimizer_options,scheduler_options],api_name='optimization_choices' if selector is action else False, preprocess=False)
        def operation_layout(method, action, backbone):
            training = action in ('train', 'train_short')
            epoch = action == 'train'
            return (gr.update(visible=bool(slots(method)) and action != 'recipe'),
                    gr.update(visible=training), gr.update(visible=action == 'evaluate'),
                    gr.update(visible=epoch), gr.update(visible=epoch), gr.update(visible=epoch),
                    gr.update(visible=epoch and method not in ('graspness', 'finegrasp', 'economicgrasp')),
                    gr.update(visible=backbone in ('dinov2', 'dinov3', 'utonia', 'concerto', 'mambavision', 'efficientvit', 'fastvit')),
                    gr.update(visible=any(slot.name=='crop' for slot in slots(method))))
        gr.on([method.change, action.change, backbone.change], operation_layout, [method, action, backbone],
            [composition_panel, training_panel, predictions, epochs, epoch_panel, epoch_help, eval_batch_limit, pretraining_panel, crop_panel],
            api_name=False, preprocess=False, queue=False, trigger_mode='always_last')
        def reset_inactive_training(action):
            training = action in ('train', 'train_short')
            return ([gr.update() if training else gr.update(value='{}') for _ in range(2)] +
                    [gr.update() if action == 'train' else gr.update(value=0) for _ in range(3)])
        action.change(reset_inactive_training, action,
            [loss_options, augmentation_options, train_batch_limit, eval_batch_limit, data_workers], api_name=False, preprocess=False)
        def action_defaults(a,m):
            training=a in ('train_short','train')
            workspace_policy='native_demo' if m in ('hggd','region_normalized_grasp','spgrasp') or (m=='finegrasp' and not training) else (('fused_gt_workspace' if a=='train_short' else 'fused_scene') if m=='generalizing_grasp' else 'official_gt_workspace')
            return ('train',0,5e-6 if m=='spgrasp' else 2e-6 if m=='gfla' else .01 if m == 'gtg2' else 1e-4,workspace_policy) if training else ('test_seen',100,.001,workspace_policy)
        for selector in (method, action):
            selector.change(lambda m,a:gr.update(value='{}',interactive=(m in ('hggd','gtg2') and a=='train' or m=='spgrasp' and a=='train_short' or m=='scale_balanced_grasp' and a in ('train','train_short'))),[method,action],trainer_options,api_name=False, preprocess=False)
            selector.change(lambda m,a:gr.update(visible=(m in ('hggd','gtg2') and a=='train' or m=='spgrasp' and a=='train_short' or m=='scale_balanced_grasp' and a in ('train','train_short'))),[method,action],trainer_panel,api_name=False, preprocess=False)
            selector.change(lambda m,a: gr.update(value=0,interactive=m not in ('graspness','finegrasp','economicgrasp') and a=='train'),[method,action],eval_batch_limit,api_name=False, preprocess=False)
        action.change(lambda a:gr.update(visible=a=='train_short'),action,training_steps,api_name=False, preprocess=False)
        action.input(action_defaults,[action,method],[split,scene,lr,workspace],api_name='action_defaults', preprocess=False, queue=False)
        action.change(lambda a: gr.update(value='initialize',interactive=a=='train',visible=a=='train'),action,train_checkpoint_mode,api_name=False, preprocess=False)
        def training_checkpoint(a,m,c,path):
            from .weights import primary
            return primary(m,c) if a in ('train_short','train') and not path else path
        action.change(training_checkpoint,[action,method,camera,checkpoint],checkpoint,api_name=False, preprocess=False)
        group.input(filter_methods, group, method, api_name="filter_methods", preprocess=False, queue=False).then(
            select_method, [method,camera], [card,action,run,checkpoint,camera,workspace,points], api_name=False, preprocess=False, queue=False).then(
            select_components, method, [backbone,crop,component_contract,head,memory,sampling], api_name=False, preprocess=False, queue=False)
        load_prompt_frame.click(prompt_frame, [dataset,camera,scene,frame], [prompt_image,prompt_options], api_name='load_prompt_frame')
        prompt_image.select(add_prompt_point, [prompt_options,prompt_object,prompt_label,prompt_image], [prompt_options,prompt_image], api_name=False)
        gr.on([dataset.change,camera.change,scene.change,frame.change,method.change],
            lambda: (None,'[]'), outputs=[prompt_image,prompt_options], api_name=False,
            queue=False, trigger_mode='always_last')
        method.change(lambda: '{}', outputs=planar_options, api_name=False)
        method.change(lambda m: gr.update(value=0 if m=='spgrasp' else .01), method, collision, api_name=False, preprocess=False)
        gr.on([method.change, action.change],
            lambda m,a: (gr.update(visible=m=='spgrasp'),gr.update(visible=m=='spgrasp' and a=='infer')),
            [method,action], [planar_panel,prompt_panel], api_name=False, preprocess=False, queue=False, trigger_mode='always_last')
        action.change(lambda a: gr.update() if a=='infer' else gr.update(value='[]'), action, prompt_options, api_name=False, preprocess=False)
        # Only a user's selection resets method defaults. Programmatic change
        # notifications can arrive again after a preset or form edit.
        method.input(select_method, [method,camera], [card, action, run, checkpoint,camera,workspace,points], api_name="select_method", preprocess=False, queue=False)
        camera.input(checkpoint_for,[method,camera],checkpoint,api_name=False, preprocess=False, queue=False)
        component_download.click(download_component_weights,[method,backbone,component_options],component_download_message,api_name='download_component_weights',concurrency_limit=1)
        download.click(download_checkpoint,[method,camera],[checkpoint,download_message],api_name='download_checkpoint',concurrency_limit=1)
        for selector in (action, method):
            selector.change(lambda a,m: [gr.update(interactive=a!='recipe' and not (m=='gtg2' and (i in (5,7) or a=='train' and i in (2,4)) or m=='spgrasp' and i in (5,7,8) or m=='generalizing_grasp' and a=='infer' and i in (3,4,7))) for i in range(13)],
                [action,method],[camera,split,scene,frame,count,points,seed,workspace,collision,epochs,batch,lr,predictions],api_name=False, preprocess=False)
        method.change(lambda m: ('Configure objects (1–8), box_probability (0–1), correction_clicks (0–7), conditioning_frames and correction_frames (1–4). Training simulates prompts from instance labels. conditioning_frames <= correction_frames <= frame count.' if m=='spgrasp' else 'Prepare graphs with `./panda prepare-gtg2 --config YOUR.local.yaml`. Set scene IDs and held-out folds in Trainer parameters. [GtG2 guide](https://github.com/Daeda1used/GraspPanda/blob/main/docs/REFERENCE.md#candidate-graph-experiments)' if m == 'gtg2' else 'Enter `{"noisy_clean": true, "clean_probability": 0.25}` in Trainer parameters. Prepare the CAD cache with `./panda prepare-clean-scenes` and set Prepared targets / cache root to its output. [Scale-Balanced-Grasp guide](https://github.com/Daeda1used/GraspPanda/blob/main/docs/REFERENCE.md#scale-balanced-grasp-components)' if m == 'scale_balanced_grasp' else 'Configure HGGD stages, accumulation and sampling. [HGGD guide](https://github.com/Daeda1used/GraspPanda/blob/main/docs/REFERENCE.md#hggd-epoch-training)'),method,trainer_help,api_name=False, preprocess=False)
        method.change(lambda m:gr.update(label='Prepared graph root (required for training)' if m == 'gtg2' else 'Prepared targets / cache root (optional)'),method,label_root,api_name=False, preprocess=False)
        action.change(lambda a,m:gr.update(interactive=a!='recipe' or m in CHECKPOINT_RECIPES),[action,method],checkpoint,api_name=False, preprocess=False)
        def seed_warmup_control(method, action):
            enabled = method in ('region_normalized_grasp','economicgrasp','graspness','finegrasp') and action == 'train_short'
            return gr.update(interactive=enabled, visible=enabled, **({} if enabled else {'value': 0}))
        gr.on([method.change, action.change], seed_warmup_control, [method, action], proposal_warmup_steps,
            api_name=False, preprocess=False, queue=False, trigger_mode='always_last')
        gr.on([method.change, action.change], lambda m,a: gr.update(visible=m=='scale_balanced_grasp' and a in ('infer','evaluate'), **({'value':'upstream'} if a not in ('infer','evaluate') or m!='scale_balanced_grasp' else {})), [method,action], sampling, api_name=False, preprocess=False, queue=False)
        gr.on([method.change, action.change, sampling.change], lambda m,a,s: gr.update(visible=m=='scale_balanced_grasp' and a in ('infer','evaluate') and s=='object_balanced'), [method,action,sampling], obs_panel, api_name=False, preprocess=False, queue=False)
        def prepare_obs_weights():
            from .weights import fetch_component
            try: fetch_component('scale_balanced_dsn')
            except (ValueError, OSError) as error: raise gr.Error(str(error)) from error
            return 'RealSense DSN weights are ready.'
        obs_download.click(prepare_obs_weights, outputs=obs_message, api_name='prepare_obs_weights')
        def inactive_sampling_parameters(method, action, current):
            if method != 'scale_balanced_grasp' or action in ('infer', 'evaluate'): return gr.update()
            try: parameters = json.loads(current or '{}')
            except (ValueError, TypeError): return gr.update()
            if not isinstance(parameters, dict) or 'sampling' not in parameters: return gr.update()
            parameters.pop('sampling')
            return json.dumps(parameters, indent=2)
        action.change(inactive_sampling_parameters, [method, action, component_options], component_options,
                      api_name=False, preprocess=False, queue=False)
        from .refinement import METHODS as refinement_methods
        gr.on([method.change,action.change],lambda m,a:gr.update(visible=m in refinement_methods and a in ('infer','evaluate')),
              [method,action],refinement_panel,api_name=False,preprocess=False,queue=False,trigger_mode='always_last')
        gr.on([method.input,action.input],lambda:('none','{}'),outputs=[refinement_kind,refinement_options],api_name=False,queue=False)
        refinement_kind.change(lambda kind:gr.update(interactive=kind!='none',**({'value':'{}'} if kind=='none' else {})),
                               refinement_kind,refinement_options,api_name=False,preprocess=False,queue=False)
        gr.on([method.change,action.change],lambda m,a:(gr.update(),gr.update()) if m in refinement_methods and a in ('infer','evaluate') else ('none','{}'),
              [method,action],[refinement_kind,refinement_options],api_name=False,preprocess=False,queue=False,trigger_mode='always_last')
        def prepare_refinement(camera='realsense',parameters='{}'):
            from .refinement import WEIGHTS
            from .weights import fetch_component
            try:
                options=json.loads(parameters or '{}')
                if not isinstance(options,dict):raise ValueError('Refinement parameters must be a mapping')
                for key,name in WEIGHTS.items():
                    if options.get(key) or (key=='segmentation_checkpoint' and camera!='realsense'):continue
                    fetch_component(name)
            except (ValueError,OSError) as error:raise gr.Error(str(error)) from error
            if camera!='realsense' and not options.get('segmentation_checkpoint'):
                return 'Contact and score weights are ready. Set segmentation_checkpoint to your matching Kinect DSN weights.'
            return 'Registered refinement weights are ready. Custom checkpoint paths are checked when validating the experiment.'
        refinement_download.click(prepare_refinement,[camera,refinement_options],outputs=refinement_message,api_name='prepare_refinement')
        inputs = [method, action, dataset, checkpoint, camera, split, scene, frame, count, points, seed, workspace, collision, epochs, batch, lr, predictions, gpu,dataset_key,backbone,crop,checkpoint_policy,training_steps,label_root,train_checkpoint_mode,train_batch_limit,eval_batch_limit,data_workers,component_options,loss_options,augmentation_options,optimizer_kind,optimizer_options,scheduler_kind,scheduler_options,proposal_warmup_steps,trainer_options,timeout,head,memory,prompt_options,planar_options,sampling,refinement_kind,refinement_options]
        generate.click(compose, inputs, config_text, api_name="compose_config")
        check.click(preflight, config_text, message, api_name="validate_config")
        run.click(submit, config_text, [job_id, message], api_name="submit_experiment")
        run_form.click(submit_form,inputs,[job_id,message,config_text],api_name='submit_form')
        sweep_preview_button.click(preview_sweep,[config_text,sweep_grid],sweep_preview,api_name='preview_sweep')
        sweep_run_button.click(run_sweep,[config_text,sweep_grid],[job_id,message],api_name='run_sweep')
        preset_button.click(apply_preset,[method,dataset],
            [action,camera,checkpoint,workspace,points,split,scene,frame,count,seed,epochs,batch,lr,label_root,timeout,config_text,collision,
             backbone,crop,checkpoint_policy,component_options,loss_options,augmentation_options,optimizer_kind,optimizer_options,scheduler_kind,scheduler_options,
             trainer_options,head,memory,prompt_options,planar_options,proposal_warmup_steps,training_steps,train_checkpoint_mode,train_batch_limit,eval_batch_limit,data_workers,prompt_image,preset_status,sampling,refinement_kind,refinement_options],
            api_name='apply_preset', concurrency_id='method-preset', concurrency_limit=1)
        gr.on([method.input,group.input], lambda: '', outputs=preset_status, api_name=False, queue=False)
        action.change(lambda a: ('**Fixed recipe:** '+ 'The method card specifies its actual input and settings. Single-frame/training fields below are ignored; click Load preset before running.') if a=='recipe' else '',action,download_message,api_name=False, preprocess=False)
        refresh.click(job_rows, outputs=table, api_name="list_runs")
        inspect_button.click(inspect, job_id, [logs, result, preview, artifacts], api_name="inspect_run")
        inspect_button.click(training_curve,job_id,loss_plot,api_name='training_curve')
        reuse_button.click(reuse_checkpoint,job_id,[config_text,job_message],api_name='reuse_checkpoint')
        def selected(evt: gr.SelectData):
            return evt.row_value[0]
        table.select(selected, outputs=job_id, api_name=False)
        stop.click(cancel, job_id, job_message, api_name="cancel_experiment")
        export_button.click(export, job_id, bundle, api_name="export_experiment")
        compare_button.click(compare, outputs=comparisons, api_name="compare_runs")
        timer = gr.Timer(2)
        timer.tick(inspect, job_id, [logs, result, preview, artifacts], api_name=False, show_progress="hidden")
        timer.tick(training_curve,job_id,loss_plot,api_name=False,show_progress='hidden')
        timer.tick(job_rows, outputs=table, api_name=False, show_progress="hidden")
    app.panda_manager = manager
    return app


def launch(host="127.0.0.1", port=7860):
    app = create_app()
    password = os.environ.get("GRASPPANDA_PASSWORD")
    app.queue().launch(server_name=host, server_port=port, share=False,
                       auth=(os.environ.get("GRASPPANDA_USER", "panda"), password) if password else None,
                       allowed_paths=[str(app.panda_manager.root)], css=CSS, theme=gr.themes.Soft(primary_hue="emerald"),
                       footer_links=[], show_error=True)
