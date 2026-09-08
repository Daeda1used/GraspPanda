"""Browser workbench for reproducible visual grasping experiments."""
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
    'pipeline_smoke': 'Run native recipe', 'train': 'Train across epochs',
    'train_check': 'Short training run', 'train_smoke': 'Single training step',
    'probe': 'Check model environment',
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


def component_parameters(method, backbone, crop):
    from .module_options import schema
    rows = []
    available = {slot.name: slot for slot in slots(method)}
    for slot, choice in (('backbone', backbone), ('crop', crop)):
        if slot not in available or choice not in available[slot].choices:
            continue
        for key, rule in schema(method, slot, choice).items():
            if rule[0] == 'choice':
                description = ', '.join(rule[1])
            elif rule[0] in ('int', 'float'):
                description = f'{rule[0]}: {rule[1]} to {rule[2]}'
            elif rule[0] == 'int_list':
                description = f'{rule[1]} integers, each {rule[2]} to {rule[3]}'
            else:
                description = {'channels': '1–8 layer widths, each 8–2048',
                               'radii': '1–8 radius factors, each 0.1–4',
                               'blocks': '5 stage depths, each 1–12'}[rule[0]]
            rows.append(f'| `{slot}.{key}` | {description} |')
    return ('| Parameter | Accepted values |\n|---|---|\n'+'\n'.join(rows)+'\n\nOmitted parameters use component defaults. Cross-stage constraints are checked when generating or running the configuration.') if rows else 'The selected components use their native settings.'


def documentation(name):
    """Render repository guides with usable links inside the browser app."""
    import re
    import posixpath
    text = (ROOT/'docs'/name).read_text()
    def link(match):
        target = match.group(2)
        if '://' in target or target.startswith('#'):
            return match.group(0)
        target = posixpath.normpath(posixpath.join('docs', target))
        return f"[{match.group(1)}](https://github.com/Daeda1used/GraspPanda/blob/main/{target})"
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)


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
        try:
            progress(0,desc='Downloading / verifying author weights…')
            value=fetch(method,camera,lambda message: progress(.5,desc=message))
            progress(1,desc='Verified')
            return value, 'All registered weight roles downloaded and SHA256 verified. Ready to run.'
        except Exception as error:
            raise gr.Error(str(error)) from error

    def apply_preset(method,dataset):
        config=preset(method,(dataset or '').strip())
        if not capabilities(method): raise gr.Error('This source has no runnable adapter. See the method card.')
        return config.action, config.camera, config.checkpoint, config.workspace, config.num_points, config.split, config.scene, config.frame, config.frames, config.seed, json.dumps(config.to_dict(),indent=2)

    def setup_check(dataset):
        import torch
        root=Path((dataset or '').strip()) if (dataset or '').strip() else None
        cameras={c:bool(root and (root/'scenes/scene_0100'/c/'depth/0000.png').exists()) for c in ('realsense','kinect')}
        return {'python':sys.version.split()[0], 'torch':torch.__version__, 'cuda_available':torch.cuda.is_available(),
                'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'scene_0100_depth':cameras, 'next_step':'Select method → Load preset → Download weights → Run current form.'}

    def compose(method, action, dataset, checkpoint, camera, split, scene, frame, count, points, seed, workspace, collision, epochs, batch, lr, predictions, gpu, dataset_key="graspnet1b", backbone="upstream", crop="upstream", checkpoint_policy="strict", training_steps=3, label_root="", train_checkpoint_mode='initialize', train_batch_limit=0, eval_batch_limit=0, data_workers=0, component_options='{}', loss_options='{}', augmentation_options='{}', optimizer_kind='upstream', optimizer_options='{}', scheduler_kind='upstream', scheduler_options='{}', proposal_warmup_steps=0):
        def mapping(text, label):
            try:
                value=json.loads(text or '{}')
            except (ValueError, TypeError) as error:
                raise gr.Error(f'{label} must contain valid JSON') from error
            if not isinstance(value,dict):raise gr.Error(f'{label} must be a mapping')
            return value
        parameters=mapping(component_options,'Component parameters')
        loss=mapping(loss_options,'Loss configuration')
        augmentation=mapping(augmentation_options,'Augmentation configuration')
        optimizer=mapping(optimizer_options,'Optimizer parameters')
        scheduler=mapping(scheduler_options,'Scheduler parameters')
        for kind, options, name in ((optimizer_kind, optimizer, 'optimizer'),(scheduler_kind, scheduler, 'scheduler')):
            if 'type' in options:raise gr.Error(f'Choose the {name} type with its selector')
            if kind=='upstream' and options:raise gr.Error(f'Select a custom {name} before setting its parameters')
        optimizer={'type':optimizer_kind,**optimizer} if optimizer_kind!='upstream' else {}
        scheduler={'type':scheduler_kind,**scheduler} if scheduler_kind!='upstream' else {}
        if action == 'pipeline_smoke':
            if parameters or loss or augmentation or optimizer or scheduler or proposal_warmup_steps or backbone!='upstream' or crop!='upstream':
                raise gr.Error('The native recipe uses fixed components and settings. Load its preset to reset overrides.')
            from dataclasses import replace
            config=replace(preset(method,(dataset or '').strip()),gpu=int(gpu))
            if method in CHECKPOINT_RECIPES:config=replace(config,checkpoint=(checkpoint or '').strip())
            return json.dumps(config.to_dict(),indent=2)
        selection={s.name:{"backbone":backbone,"crop":crop}[s.name] for s in slots(method) if {"backbone":backbone,"crop":crop}[s.name]!="upstream"}
        for name,values in parameters.items():
            if name not in selection:raise gr.Error(f'Select a replacement for {name} before configuring its parameters')
            if not isinstance(values,dict) or 'type' in values:raise gr.Error('Use the component selector for type; supply only its parameters here')
            selection[name]={'type':selection[name],**values}
        config = Experiment(proposal_warmup_steps=int(proposal_warmup_steps),dataset=dataset_key,modules=selection,loss=loss,augmentation=augmentation,optimizer=optimizer,scheduler=scheduler,checkpoint_policy=checkpoint_policy,training_steps=int(training_steps),label_root=(label_root or '').strip(),method=method, action=action or "probe", dataset_root=(dataset or '').strip(), checkpoint=(checkpoint or '').strip(),
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
            raise gr.Error('Checkpoint saved, but this method currently exposes only a fixed inference recipe. Configurable checkpoint inference is not yet integrated; see its method card.')
        path=manager.root/row['id']/'checkpoint.pt'
        if not path.exists():raise gr.Error('This run has no saved training checkpoint.')
        config=replace(Experiment.from_dict(row['config']),action='infer',checkpoint=str(path),
                       checkpoint_policy='strict',split='test_seen',scene=100,frame=0,frames=1,loss={},augmentation={},optimizer={},scheduler={},proposal_warmup_steps=0,train_checkpoint_mode='initialize')
        if 'infer' not in capabilities(method):
            config=replace(preset(method,row['config']['dataset_root']),checkpoint=str(path),gpu=row['config']['gpu'])
        config.validate()
        note='The fixed native input recipe is retained.' if config.action=='pipeline_smoke' else 'Review the selected test frame.'
        return json.dumps(config.to_dict(),indent=2), 'Inference JSON prepared in Experiments. '+note+' Click Run edited JSON. A short training check does not establish grasp quality.'

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
            settings = {key: config.get(key) for key in ('learning_rate', 'optimizer', 'scheduler', 'loss', 'augmentation')}
            rows.append([job["id"], config.get('dataset','graspnet1b'),config["method"],json.dumps(config.get('modules',{})), data.get("stage"), config["camera"], config["split"],
                         config["workspace"], config["seed"], json.dumps(settings), last_loss, (len(data["frames"]) if isinstance(data.get("frames"),list) else data.get("frames", "recipe")), json.dumps(data.get("ap"))])
        return rows

    with gr.Blocks(title="GraspPanda · Modular visual grasping") as app:
        gr.HTML('<div id="panda-hero"><h1>🐼 GraspPanda</h1><p>Compose. Experiment. Grasp. · One runtime. Reproducible research.</p></div>')
        with gr.Tab("Experiments"):
            with gr.Row():
                with gr.Column(scale=4):
                    dataset_key=gr.Dropdown(choices=[(s.title,key) for key,s in datasets().items()],value="graspnet1b",label="Dataset integration")
                    group = gr.Dropdown(groups, value="All", label="Protocol & family")
                    method = gr.Dropdown([(v.get("name", m), m) for m, v in items.items()], value="graspnet_baseline", label="Method")
                    with gr.Accordion("Method details & input requirements", open=False):
                        card = gr.Markdown(method_card("graspnet_baseline"))
                    action = gr.Dropdown(action_choices("graspnet_baseline"), value="infer", label="Operation")
                    preset_button=gr.Button("Load preset",variant="primary")
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
                        workspace = gr.Dropdown(["official_gt_workspace", "depth_only", "native_demo", "fused_gt_workspace"], value="official_gt_workspace", label="Workspace policy")
                        gr.Markdown("Official workspace uses dataset segmentation + poses. HGGD / RNG use native_demo. Load the preset to select the correct protocol.", elem_classes="panda-note")
                        collision = gr.Number(0.01, label="Collision threshold (0 disables)")
                    with gr.Accordion("Compose modules",open=False):
                        gr.Markdown("Select compatible building blocks. **reuse_unchanged** initializes replaced components and retains only unchanged checkpoint modules. Run a training check before using a new composition.")
                        backbone=gr.Dropdown(['upstream','pointnet','pointnext','pointmlp'],value='upstream',label='Point encoder')
                        crop=gr.Dropdown(['upstream','multiscale','cylinder'],value='upstream',label='Local cylindrical grouping')
                        checkpoint_policy=gr.Dropdown(['strict','reuse_unchanged'],value='strict',label='Checkpoint policy')
                        component_contract=gr.Markdown('Baseline: 256-channel seed features, original point indices, four depth bins.')
                        component_options=gr.Code('{}',language='json',label='Component parameters by slot',lines=5)
                        gr.Markdown('Enter parameters keyed by slot, for example `{"backbone": {"embed_dim": 32}}` for PointMLP. The selectors supply each component type.')
                        with gr.Accordion('Available component parameters', open=False):
                            parameter_help = gr.Markdown(component_parameters('graspnet_baseline', 'upstream', 'upstream'))
                    with gr.Accordion("Training & evaluation settings", open=False):
                        training_steps=gr.Number(3,precision=0,minimum=1,maximum=1000,label='Optimizer steps (short training)')
                        proposal_warmup_steps=gr.Number(0,precision=0,minimum=0,maximum=10000,interactive=False,label='RNG anchor warmup updates',info='Optional real-label anchor training before preparing local proposals. Added to short-training updates; 0 preserves the native preset.')
                        loss_options=gr.Code('{}',language='json',label='Loss configuration',lines=3)
                        augmentation_options=gr.Code('{}',language='json',label='Augmentation configuration',lines=3)
                        with gr.Accordion('Optimizer & learning-rate schedule', open=False):
                            optimizer_kind=gr.Dropdown(['upstream'],value='upstream',label='Optimizer',interactive=False)
                            optimizer_options=gr.Code('{}',language='json',label='Optimizer parameters',lines=3)
                            scheduler_kind=gr.Dropdown(['upstream'],value='upstream',label='Learning-rate schedule',interactive=False)
                            scheduler_options=gr.Code('{}',language='json',label='Scheduler parameters',lines=3)
                            gr.Markdown('The learning-rate field below sets the base rate. Scheduler warmup and milestones count **optimizer updates**, not epochs. Examples: optimizer `{"weight_decay": 0.01}`; cosine schedule `{"warmup_steps": 1, "min_lr_ratio": 0.01}`. Omit parameters to use the selected implementation defaults; see Guide → Modules for supported settings.')
                        label_root=gr.Textbox(label='HGGD / RNG preprocessed label root (optional)',placeholder='Leave blank to use HGGD_Preprocessed under the dataset root')
                        gr.Markdown('Training checks use fixed batches: HGGD, GraNet and fusion use batch 2; other point methods use batch 1; RNG uses anchor batch 2 and up to 48 local patches. The controls below apply to native epoch training.')
                        with gr.Row():
                            epochs = gr.Number(1, precision=0, label="Final epoch (must exceed resume epoch)")
                            batch = gr.Number(2, precision=0, label="Batch size")
                            lr = gr.Number(0.001, label="Learning rate")
                        train_checkpoint_mode=gr.Dropdown(['initialize','resume'],value='initialize',label='Epoch training checkpoint mode',info='Initialize loads model weights with a fresh optimizer. Resume restores model, optimizer and epoch; requires strict loading.')
                        with gr.Row():
                            train_batch_limit=gr.Number(0,precision=0,minimum=0,label='Training batches per epoch (0 = full split)')
                            eval_batch_limit=gr.Number(0,precision=0,minimum=0,label='Validation batches per epoch (0 = full split)')
                            data_workers=gr.Number(0,precision=0,minimum=0,maximum=32,label='Data loader workers')
                        gr.Markdown('A batch limit uses consecutive real frames from First scene / First frame, with native shuffling and augmentation. Baseline and SBG validate on test_seen; Graspness has no native validation loop. Full split runs can take many hours: adjust timeout_minutes in JSON.')
                        predictions = gr.Textbox(label="Complete predictions directory for evaluation")
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
                    logs = gr.Textbox(label="Live log (last 32 KB)", lines=16, interactive=False)
                    result = gr.JSON(label="Result")
                    loss_plot=gr.LinePlot(x='Batch',y='Loss',color='Stage',title='Training loss by stage',label='Native objective')
                with gr.Column():
                    preview = gr.Image(label="Predicted gripper projection (first frame)", interactive=False)
                    artifacts = gr.File(label="Configuration, provenance & logs", file_count="multiple")
                    bundle = gr.File(label="Exported experiment")
        with gr.Tab("Compare"):
            gr.Markdown("Only completed runs are listed. Compare AP only under identical dataset, camera, split, workspace, training data and postprocessing. `null` AP means not evaluated; it is never zero AP. Compare losses only when objectives, coefficients and sampled data match.")
            compare_button = gr.Button("Refresh comparison")
            comparisons = gr.Dataframe(headers=["ID", "Dataset", "Method", "Modules", "Stage", "Camera", "Split", "Workspace", "Seed", "Training settings", "Final loss", "Frames", "AP"], interactive=False)
        with gr.Tab("Guide"):
            gr.Markdown("""### Start an experiment
1. Choose a method in **Experiments**, then **Load preset**.
2. Set the dataset root containing `scenes/`. **Download registered weights** prepares the required networks.
3. Choose an operation and **Run current form**. View predictions, checkpoints and logs in **Runs & results**.

**No dataset yet?** Use the baseline's **Check model environment**, or select ASGrasp, load its preset and download its weights to run the author's stereo sample.

For component experiments, expand **Compose modules**. Full configuration editing is under **Configuration editor**. Short training repeats a bounded labelled sample; epoch training uses the native loader and schedule.
""")
            with gr.Accordion("Installation, data and operating instructions", open=False):
                with gr.Tabs():
                    with gr.Tab("Install"):
                        gr.Markdown(documentation('INSTALL.md'))
                    with gr.Tab("Downloads"):
                        gr.Markdown(documentation('DOWNLOADS.md'))
                    with gr.Tab("Usage"):
                        gr.Markdown(documentation('USAGE.md'))
                    with gr.Tab("Modules"):
                        gr.Markdown(documentation('MODULES.md'))
            with gr.Accordion("Check data and GPU", open=False):
                setup_root=gr.Textbox(default_dataset(),label='Dataset root')
                setup_button=gr.Button('Check data & GPU')
                setup_result=gr.JSON(label='Readiness')
                setup_button.click(setup_check,setup_root,setup_result,api_name='readiness')
        with gr.Tab("Methods & papers"):
            gr.Markdown(documentation('METHODS.md'))
        def select_components(method):
            enabled=bool(slots(method))
            contract='\n\n'.join(f"**{s.name}**: {s.input_contract} → {s.output_contract}" for s in slots(method)) if enabled else 'This method currently retains its native components. No interchangeable slots are registered.'
            choices={s.name:list(s.choices) for s in slots(method)}
            return gr.update(choices=choices.get('backbone',['upstream']),value='upstream',interactive='backbone' in choices,label='Image encoder' if method in ('hggd','region_normalized_grasp') else 'Point encoder'),gr.update(choices=choices.get('crop',['upstream']),value='upstream',interactive='crop' in choices),contract
        method.change(select_components,method,[backbone,crop,component_contract],api_name='select_components').then(
            lambda: ('{}','{}','{}','strict'),outputs=[component_options,loss_options,augmentation_options,checkpoint_policy],api_name=False)
        for selector in (method, backbone, crop):
            selector.change(component_parameters,[method,backbone,crop],parameter_help,api_name=False)
        def optimization_choices(method, action):
            from .optimization import METHODS,MUON_METHODS
            enabled=method in METHODS and action in ('train','train_check')
            optimizers=['upstream','adam','adamw','sgd','lion']+(['muon'] if method in MUON_METHODS else [])
            return gr.update(choices=optimizers if enabled else ['upstream'],value='upstream',interactive=enabled),gr.update(choices=['upstream','constant','cosine','multistep'] if enabled else ['upstream'],value='upstream',interactive=enabled),'{}','{}'
        for selector in (method, action):
            selector.change(optimization_choices,[method,action],[optimizer_kind,scheduler_kind,optimizer_options,scheduler_options],api_name=False)
        def action_defaults(a,m):
            training=a in ('train_check','train_smoke','train')
            workspace_policy='native_demo' if m in ('hggd','region_normalized_grasp','finegrasp') else ('fused_gt_workspace' if m=='generalizing_grasp' and a=='train_check' else 'official_gt_workspace')
            return ('train',0,2e-6 if m=='gfla' else 1e-4,workspace_policy) if training else ('test_seen',100,.001,workspace_policy)
        action.change(action_defaults,[action,method],[split,scene,lr,workspace],api_name='action_defaults')
        action.change(lambda a: gr.update(value='initialize',interactive=a=='train'),action,train_checkpoint_mode,api_name=False)
        def training_checkpoint(a,m,c,path):
            from .weights import primary
            return primary(m,c) if a in ('train_check','train_smoke','train') and not path else path
        action.change(training_checkpoint,[action,method,camera,checkpoint],checkpoint,api_name=False)
        group.change(filter_methods, group, method, api_name="filter_methods")
        method.change(select_method, [method,camera], [card, action, run, checkpoint,camera,workspace,points], api_name="select_method")
        camera.change(checkpoint_for,[method,camera],checkpoint,api_name=False)
        download.click(download_checkpoint,[method,camera],[checkpoint,download_message],api_name='download_checkpoint',concurrency_limit=1)
        action.change(lambda a: [gr.update(interactive=a!='pipeline_smoke') for _ in range(13)],action,[camera,split,scene,frame,count,points,seed,workspace,collision,epochs,batch,lr,predictions],api_name=False)
        action.change(lambda a,m:gr.update(interactive=a!='pipeline_smoke' or m in CHECKPOINT_RECIPES),[action,method],checkpoint,api_name=False)
        for selector in (method,action):
            selector.change(lambda m,a: gr.Number(value=0,interactive=m=='region_normalized_grasp' and a=='train_check'),[method,action],proposal_warmup_steps,api_name=False)
        preset_button.click(lambda: 0,outputs=proposal_warmup_steps,api_name=False)
        inputs = [method, action, dataset, checkpoint, camera, split, scene, frame, count, points, seed, workspace, collision, epochs, batch, lr, predictions, gpu,dataset_key,backbone,crop,checkpoint_policy,training_steps,label_root,train_checkpoint_mode,train_batch_limit,eval_batch_limit,data_workers,component_options,loss_options,augmentation_options,optimizer_kind,optimizer_options,scheduler_kind,scheduler_options,proposal_warmup_steps]
        generate.click(compose, inputs, config_text, api_name="compose_config")
        check.click(preflight, config_text, message, api_name="validate_config")
        run.click(submit, config_text, [job_id, message], api_name="submit_experiment")
        run_form.click(submit_form,inputs,[job_id,message,config_text],api_name='submit_form')
        sweep_preview_button.click(preview_sweep,[config_text,sweep_grid],sweep_preview,api_name='preview_sweep')
        sweep_run_button.click(run_sweep,[config_text,sweep_grid],[job_id,message],api_name='run_sweep')
        preset_button.click(apply_preset,[method,dataset],[action,camera,checkpoint,workspace,points,split,scene,frame,count,seed,config_text],api_name='apply_preset').then(
            lambda: ('upstream','upstream','strict','{}','{}','{}','upstream','{}','upstream','{}'),
            outputs=[backbone,crop,checkpoint_policy,component_options,loss_options,augmentation_options,optimizer_kind,optimizer_options,scheduler_kind,scheduler_options],api_name=False)
        action.change(lambda a: ('**Fixed recipe:** '+ 'The method card specifies its actual input and settings. Single-frame/training fields below are ignored; click Load preset before running.') if a=='pipeline_smoke' else '',action,download_message,api_name=False)
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
