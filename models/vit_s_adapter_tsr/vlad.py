import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim


        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class PatchNetVLAD(nn.Module):

    def __init__(self):
        super(PatchNetVLAD, self).__init__()

        num_clusters = 64
        dim = 3
        self.vlad=NetVLAD(num_clusters=num_clusters, dim=dim)
        self.act = nn.GELU()
        self.dim_map = nn.Sequential(
                nn.Linear(num_clusters*dim, 768),
                nn.GELU(),
                nn.LayerNorm(768, eps=1e-6)

        )


    def forward(self, x):
        # x: [B, N, C]
        x_cls = x[:,0:1,:]
        x_patch = x[:,1:,:]

        B, N, C = x_patch.shape # C = 3*16*16
        x_patch = x_patch.reshape(B*N, 3, 16, 16)
        x_vlad = self.vlad(x_patch) # 392~192
        x_vlad = self.act(x_vlad)
        x_vlad = self.dim_map(x_vlad)
        x_vlad = x_vlad.reshape(B, N, -1)
        x_vlad = torch.cat([x_cls, x_vlad], dim=1)

        return x_vlad


#x = torch.rand(2,197,768)

#m = PatchNetVLAD()
#import IPython; IPython.embed()



