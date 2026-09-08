"""Exact nearest-surface queries without materializing all pairwise distances."""
import ast
import inspect


def nearest_surface(sdf,points):
    import numpy as np
    from scipy.spatial import cKDTree
    surface=sdf.surface_for_signed_val
    cached=getattr(sdf,'_grasppanda_surface_index',None)
    if cached is None:
        cached=(surface.copy(),cKDTree(surface))
        sdf._grasppanda_surface_index=cached
    coordinates,tree=cached
    # Most continuous queries have a unique nearest point. A second-neighbor
    # query identifies possible ties without one Python/tree call per point.
    distances,pairs=tree.query(points,k=2,eps=0,workers=1)
    indices=pairs[:,0].copy()
    possible_ties=np.flatnonzero(np.isclose(distances[:,0],distances[:,1],rtol=1e-14,atol=1e-14))
    for i in possible_ties:
        point=points[i];distance=distances[i,0]
        candidates=tree.query_ball_point(point,np.nextafter(distance,np.inf))
        if len(candidates)>1:
            candidates=np.asarray(sorted(candidates))
            indices[i]=candidates[np.argmin(np.linalg.norm(coordinates[candidates]-point,axis=-1))]
    return coordinates[indices]


def install_gfla_sampler(module):
    """Retain native interpolation; optimize its out-of-bounds surface search."""
    if hasattr(module,"_grasppanda_native_sampler"):
        return module._grasppanda_native_sampler
    native=module.batch_signed_distance
    tree=ast.parse(inspect.getsource(native));function=tree.body[0]
    start=next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='surface')
    end=next(i for i,node in enumerate(function.body[start:],start) if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='sd')
    replacement=ast.parse('''
if np.any(out_of_bounds):
    outside=coords[out_of_bounds]
    closest=nearest_surface(sdf_,outside)
    sd[out_of_bounds]=(np.linalg.norm(np.einsum("ij,...j->...i",sdf_.T_grid_world_.matrix[:3,:3],closest-outside),axis=-1)
        + sdf_.data[closest[...,0],closest[...,1],closest[...,2]])
''').body
    function.body[start:end+1]=replacement
    namespace={**native.__globals__,'nearest_surface':nearest_surface}
    exec(compile(ast.fix_missing_locations(tree),native.__code__.co_filename,'exec'),namespace)
    module._grasppanda_native_sampler=native
    module.batch_signed_distance=namespace['batch_signed_distance']
    return native
