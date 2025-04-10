import os
import torch
from timm.models import efficientnet, convnext


def build_backbone(model_name, pretrained):
    model = getattr(Backbones, model_name)(pretrained=pretrained)
    return model


class Backbones(object):
    @staticmethod
    def efficientnet_b3_p(pretrained):
        # channels: 24, 12, 40, 120, 384
        # for test, pretrained can be set to False
        model = efficientnet.efficientnet_b3_pruned(pretrained=pretrained, features_only=True)
        
        '''
        # pre-downloaded weights
        cp_path = os.path.join('checkpoints', 'effnetb3_pruned-59ecf72d.pth')
        state_dict = torch.load(cp_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict=state_dict, strict=False)'''
        return model
    
    
