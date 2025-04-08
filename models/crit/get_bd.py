import torch
import torch.nn.functional as F


def generate_BD(mask):
    #print(mask.size())
    # img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    # mask = mask.float()
    mask = torch.abs(mask - F.max_pool2d(mask, 3, 1, 1))
    mask = mask.detach()
    
    return mask