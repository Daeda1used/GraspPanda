"""PointNet-style features on quantized points, preserving the sparse row map."""
import torch
from torch import nn


class SparsePointNet(nn.Module):
    def __init__(self,out_channels=512,voxel_size=.005):
        super().__init__()
        self.voxel_size=voxel_size
        self.local=nn.Sequential(nn.Linear(6,64,bias=False),nn.BatchNorm1d(64),nn.ReLU(),
                                 nn.Linear(64,128,bias=False),nn.BatchNorm1d(128),nn.ReLU())
        self.context=nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(384,256,bias=False),nn.BatchNorm1d(256),nn.ReLU(),nn.Linear(256,out_channels))

    def forward(self,sparse):
        import MinkowskiEngine as ME
        coords=sparse.C.to(device=sparse.F.device)
        xyz=coords[:,1:].to(sparse.F.dtype)*self.voxel_size
        local=self.local(torch.cat([xyz,sparse.F],1))
        context=self.context(local)
        pooled=torch.empty_like(context)
        for batch in coords[:,0].unique():
            mask=coords[:,0]==batch
            pooled[mask]=context[mask].amax(0,keepdim=True)
        features=self.fusion(torch.cat([local,pooled],1))
        return ME.SparseTensor(features,coordinate_map_key=sparse.coordinate_map_key,coordinate_manager=sparse.coordinate_manager)
