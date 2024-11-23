import torch
import os
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from skimage import img_as_ubyte
from collections import OrderedDict
from ultralytics.data.filter.SRMNet import SRMNet

def load_checkpoint(model):
    checkpoint = torch.load(os.path.join(os.getcwd(),'ultralytics','data','filter','real_denoising_SRMNet.pth'))
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def create_denoise_model():
    # Load corresponding models architecture and weights
    model = SRMNet()
    model.cuda()
    load_checkpoint(model)
    model.eval()
    return model

def denoise(img,model):
    print('restoring images......')
    mul = 16
    input_ = img

    # Pad the input if not_multiple_of 8
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model(input_)
    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    return restored
