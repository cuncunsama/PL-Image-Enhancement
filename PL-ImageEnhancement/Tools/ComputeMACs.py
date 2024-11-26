from yxq.model.arch import NAFNet, CascadedGaze, KBNet_s, Restomer, HINet, MPRNet, NAFNet_Update
from yxq.model.arch import HINetLocal, MPRNetLocal, HINet_FRN_SAM_Local, NAFNetLocal, HINet_FRN_SAM
from thop import profile
import torch

model = HINet_FRN_SAM()
input = torch.randn(1, 1, 256, 256)
macs, params = profile(model, inputs=(input, ))
print(f'MACs: {macs}, Params: {params}')
