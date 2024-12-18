# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
"""
## Revisiting Global Statistics Aggregation for Improving Image Restoration
## Xiaojie Chu, Liangyu Chen, Chengpeng Chen, Xin Lu
"""
import torch
from torch import nn
from torch.nn import functional as F


from .HINet import HINet
from .MPRNet import MPRNet


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=(1,3,256,256)):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride=1, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.fast_imp
        )
           
    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//self.train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//self.train_size[-1]
            
            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//self.train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//self.train_size[-1])

        if self.fast_imp:   # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0)) # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)
    
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out

class LocalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        super().__init__()
        assert not track_running_stats
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.avgpool = AvgPool2d()
        self.eps = eps

    def forward(self, input):
        mean_x = self.avgpool(input) # E(x)
        mean_xx = self.avgpool(torch.mul(input, input)) # E(x^2)
        mean_x2 = torch.mul(mean_x, mean_x) # (E(x))^2
        var_x = mean_xx - mean_x2 # Var(x) = E(x^2) - (E(x))^2
        mean = mean_x
        var = var_x
        input = (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight.view(1,-1, 1, 1) + self.bias.view(1,-1, 1, 1)
        return input
        

def replace_layers(model, base_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, fast_imp, **kwargs)
            
        if isinstance(m, nn.AdaptiveAvgPool2d): 
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, **kwargs)
            assert m.output_size == 1
            setattr(model, n, pool)

        if isinstance(m, nn.InstanceNorm2d):
            norm = LocalInstanceNorm2d(num_features=m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
            norm.avgpool.base_size = base_size # bad code
            norm.avgpool.fast_imp = fast_imp 
            setattr(model, n, norm)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class HINetLocal(Local_Base, HINet):
    def __init__(self, *args, base_size=None, train_size=(1,3,256,256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        HINet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))
        print(f"faster implementation: {fast_imp}")
        print(f"train_size: {train_size}")
        print(f"base_size: {base_size}")
        self.convert(base_size=base_size, fast_imp=fast_imp, train_size=train_size)


class MPRNetLocal(Local_Base, MPRNet):
    def __init__(self, *args, base_size=None, train_size=(1,3,256,256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MPRNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))
        self.convert(base_size=base_size, fast_imp=fast_imp, auto_pad=False, train_size=train_size)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MPRNetLocal().to(device)
    # net = HINetLocal().to(device)
    for size in [31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]:
        img = torch.randn(1, 3, size, size).to(device)
        outputs = net(img)
        print(*[x.shape for x in outputs])