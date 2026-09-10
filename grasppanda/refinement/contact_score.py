"""Contact and score objectives from the pinned Generalizing-Grasp networks."""
from pathlib import Path
import ast
import math


def native_functions(source):
    """Load only geometry definitions, without the author's CLI and GT dataset."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    names = {'differentiable_center_to_contact', 'to_matrix', 'calculate_cmap'}
    tree = ast.parse((source/'optimization.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Pinned contact-score geometry definitions are missing')
    namespace = dict(torch=torch, F=F, np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source/'optimization.py'), 'exec'), namespace)
    return namespace


def contact_targets(left, right, points):
    """Equivalent native distances using finite norm derivatives at coincidence."""
    import torch
    import torch.nn.functional as F
    offsets = points[:, None] - torch.stack((left, right))
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    nearest, index = distances.min(-1)
    delta = offsets[torch.arange(len(points), device=points.device), index]
    axis = F.normalize(left-right, dim=0)
    perpendicular = delta - (delta*axis).sum(-1, keepdim=True)*axis
    # sin(acos(cosine)) * distance is the perpendicular distance; no acos singularity.
    projected = 20*torch.linalg.vector_norm(perpendicular, dim=-1)
    cmap = 2*torch.sigmoid(-40*nearest)
    cmap_proj = 2*torch.sigmoid(-4*projected)
    minima = distances.min(0).values
    contact = F.relu(minima-.01).mean() + 5*F.relu(.005-minima).mean()
    return cmap, cmap_proj, contact


def pose_parameters(grasp):
    """Recover the native approach and angle from a complete GraspNet rotation."""
    import numpy as np
    matrix = grasp[4:13].reshape(3, 3)
    if not np.isfinite(grasp).all() or not np.allclose(matrix.T@matrix, np.eye(3), atol=1e-4) or abs(np.linalg.det(matrix)-1)>1e-4:
        raise ValueError('Refinement requires finite grasps with proper rotation matrices')
    approach = matrix[:, 0].copy()
    y = np.array([-approach[1], approach[0], 0.], dtype=np.float32)
    if np.linalg.norm(y) == 0: y[1] = 1
    y /= np.linalg.norm(y)
    z = np.cross(approach, y)
    angle = np.arctan2(np.dot(matrix[:, 1], z), np.dot(matrix[:, 1], y))
    return approach, float(angle)


class ContactScore:
    """Frozen author networks; gradients update pose variables only."""
    def __init__(self, source, contact_checkpoint, score_checkpoint, options=None):
        import torch
        from contactnet import ContactNet
        from scorenet import GraspScoreNet
        self.options = options or {}
        self.geometry = native_functions(Path(source))
        self.contact = ContactNet(is_training=False).cuda().eval()
        self.score = GraspScoreNet(is_training=False).cuda().eval()
        for model, path in ((self.contact, contact_checkpoint), (self.score, score_checkpoint)):
            state = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state.get('model_state_dict', state), strict=True)
            model.requires_grad_(False)

    def objective(self, grasp, points, parameters, seed):
        import numpy as np
        import torch
        import torch.nn.functional as F
        import open3d as o3d
        from graspnetAPI import Grasp
        translation, approach, angle, width, depth = parameters
        rotation = self.geometry['to_matrix'](approach[None], angle).squeeze(0)
        value = torch.cat((grasp[:1], width, grasp[2:3], depth, rotation.flatten(), translation, grasp[16:17]))
        mesh = Grasp(value.detach().cpu().numpy()).to_open3d_geometry()
        count = self.options.get('gripper_points', 64)
        # Common random samples make initial/final objective comparisons repeatable.
        o3d.utility.random.seed(seed)
        hand = torch.as_tensor(np.asarray(mesh.sample_points_uniformly(count).points), device=points.device, dtype=torch.float32)
        left, right = self.geometry['differentiable_center_to_contact'](translation, approach, angle, width, depth)
        cmap, cmap_proj, contact = contact_targets(left, right, points)
        center = points.mean(0)
        normalized = points-center
        dist, proj, _ = self.contact(dict(point_clouds=normalized[None], grasp_points=(hand-center)[None].detach(),
                                         cmap_label=cmap[None], cmap_label_proj=cmap_proj[None]))
        canonical = value.detach().cpu().numpy().copy()
        canonical[4:13] = np.eye(3).flatten(); canonical[13:16] = 0
        o3d.utility.random.seed(seed+1)
        hand = Grasp(canonical).to_open3d_geometry().sample_points_uniformly(count)
        canonical_points = torch.as_tensor(np.asarray(hand.points), device=points.device, dtype=torch.float32)
        score, _ = self.score(dict(point_clouds=normalized[None], grasp_points=canonical_points[None].detach(),
                                  rotation=rotation.flatten()[None], translation=(translation-center)[None]))
        center_distance = F.relu(torch.sqrt(((points-translation)**2).sum(-1)+1e-6)-.001)
        terms = dict(contact_map=dist.squeeze(), projection_map=proj.squeeze(), center=F.relu(center_distance.min()-.001),
                     contact_distance=contact, score=math.log(12)-score.squeeze())
        defaults = dict(contact_map=1., projection_map=.2, center=5., contact_distance=1., score=.1)
        total = sum(terms[k]*self.options.get(k+'_weight', v) for k,v in defaults.items())
        return total, terms, value

    def refine(self, grasp, points, seed=0):
        import numpy as np
        import torch
        approach, angle = pose_parameters(grasp)
        original = np.asarray(grasp, dtype=np.float32).copy()
        points = torch.as_tensor(points, dtype=torch.float32, device='cuda')
        if points.ndim!=2 or points.shape[1]!=3 or not torch.isfinite(points).all() or not len(points):
            raise ValueError('Refinement requires a finite nonempty object cloud in metres')
        original_tensor = torch.as_tensor(original, device='cuda')
        values = (original[13:16], approach, [angle], original[1:2], original[3:4])
        rates = dict(translation=.0002, approach=.002, angle=.002, width=.0001, depth=0.)
        params, groups = [], []
        for (name, default), value in zip(rates.items(), values):
            lr = self.options.get(name+'_lr', default)
            parameter = torch.tensor(value, dtype=torch.float32, device='cuda', requires_grad=lr>0)
            params.append(parameter)
            if lr>0:groups.append(dict(params=[parameter], lr=lr))
        if not groups:raise ValueError('Select at least one pose variable with a positive learning rate')
        optimizer = torch.optim.Adam(groups)
        best = original.copy(); baseline = None; best_total = float('inf'); accepted = 0; updates = 0
        history = []; stop_reason = 'iteration_limit'; best_values = None
        steps = self.options.get('iterations', 300)
        for step in range(steps+1):
            if not all(torch.isfinite(v).all() for v in params):
                stop_reason='non_finite_pose';break
            if torch.linalg.vector_norm(params[1]) < 1e-6:
                stop_reason='degenerate_approach';break
            optimizer.zero_grad(set_to_none=True)
            total, terms, candidate = self.objective(original_tensor, points, params, seed)
            values = {k:float(v.detach()) for k,v in terms.items()}
            if not torch.isfinite(total) or not all(math.isfinite(v) for v in values.values()):
                stop_reason='non_finite_objective';break
            values['total'] = float(total.detach());history.append(values)
            if baseline is None:baseline = values
            contact_loss = values['contact_map']*self.options.get('contact_map_weight', 1.) + self.options.get('projection_map_weight', .2)*values['projection_map']
            initial_contact = baseline['contact_map']*self.options.get('contact_map_weight', 1.) + self.options.get('projection_map_weight', .2)*baseline['projection_map']
            policy = self.options.get('acceptance', 'joint')
            eligible = values['total'] < baseline['total'] if policy == 'total' else contact_loss < initial_contact and values['score'] < baseline['score']
            if step and eligible and values['total'] < best_total:
                best = candidate.detach().cpu().numpy().copy();best_total=values['total'];accepted=step
                best_values=[v.detach().cpu().numpy().copy() for v in params]
            if step==steps:break
            total.backward()
            grads = [v.grad for v in params if v.requires_grad]
            if not any(g is not None for g in grads):
                stop_reason='no_pose_gradient';break
            if any(g is not None and not torch.isfinite(g).all() for g in grads):
                stop_reason='non_finite_gradient';break
            optimizer.step();updates+=1
            with torch.no_grad():
                if params[3].grad is not None:params[3].clamp_(self.options.get('min_width', .001),self.options.get('max_width', .1))
                if params[4].grad is not None:params[4].clamp_(.001, .1)
        best=best.astype(np.asarray(grasp).dtype,copy=False)
        best[[0,2,16]]=np.asarray(grasp)[[0,2,16]]
        if best_values is None:
            best=np.asarray(grasp).copy()
        else:
            initial_values=(original[13:16],approach,[angle],original[1:2],original[3:4])
            unchanged=[np.array_equal(value,np.asarray(initial,dtype=np.float32)) for value,initial in zip(best_values,initial_values)]
            if unchanged[0]:best[13:16]=grasp[13:16]
            if unchanged[1] and unchanged[2]:best[4:13]=grasp[4:13]
            if unchanged[3]:best[1]=grasp[1]
            if unchanged[4]:best[3]=grasp[3]
        final = history[accepted] if accepted else baseline
        return best, dict(updates=updates, accepted_step=accepted, initial=baseline, final=final,
                          stopped_early=stop_reason!='iteration_limit', stop_reason=stop_reason, objectives=history)
