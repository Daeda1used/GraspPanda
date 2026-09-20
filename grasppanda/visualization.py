"""Interactive views of saved geometry without importing a model implementation."""
from pathlib import Path


def hand_scene(path):
    import numpy as np
    import plotly.graph_objects as go
    import trimesh
    scene = trimesh.load(Path(path), force='scene', process=False)
    meshes, clouds = [], []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geometry = scene.geometry[name].copy()
        geometry.apply_transform(transform)
        if isinstance(geometry, trimesh.Trimesh): meshes.append(geometry)
        elif isinstance(geometry, trimesh.points.PointCloud): clouds.append(geometry.vertices)
    if not meshes: raise ValueError('The saved hand scene has no hand mesh')
    hand = trimesh.util.concatenate(meshes)
    figure = go.Figure(go.Mesh3d(x=hand.vertices[:,0],y=hand.vertices[:,1],z=hand.vertices[:,2],
        i=hand.faces[:,0],j=hand.faces[:,1],k=hand.faces[:,2],color='#29b391',
        name='Predicted hand',hoverinfo='skip',lighting=dict(ambient=.65,diffuse=.7,roughness=.6)))
    if clouds:
        points = np.concatenate(clouds)
        # Limit browser payload only. The full exported scene stays unchanged.
        points = points[::max(1,int(np.ceil(len(points)/12000)))]
        figure.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',
            marker=dict(size=1.5,color='#708596',opacity=.6),name='Observed depth',hoverinfo='skip'))
    figure.update_layout(height=480,margin=dict(l=25,r=25,t=5,b=10),showlegend=False,
        scene=dict(aspectmode='data',xaxis_title='Camera X (m)',yaxis_title='Camera Y (m)',
                   zaxis_title='Camera Z (m)',camera=dict(eye=dict(x=1.3,y=-1.6,z=-1.1),up=dict(x=0,y=-1,z=0))),
        uirevision=str(path))
    return figure
