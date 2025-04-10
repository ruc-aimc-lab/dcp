import torch
import torch.nn as nn
from .optimizer import Optimizer
from .crit import DiceBCE, generate_BD
from collections import OrderedDict

import torch.nn.functional as F


class BasicProcessor(object):
    def __init__(self) -> None:
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
    
    def set_mode(self, mode):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise Exception('Invalid model mode {}'.format(mode))

    def requires_grad_false(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_device(self, device):
        # print(device)
        if isinstance(device, list):
            if len(device) > 1:
                self.model= nn.DataParallel(self.model, device_ids=device)
                _device = 'cuda'
            else:
                _device = 'cuda:{}'.format(device[0])
            self.model.to(_device)
        else:
            self.model.to(device)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location='cpu')

        remove_module = True
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                remove_module = False
                break
        if remove_module:
        # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] #remove 'module'
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)


class Processor(BasicProcessor):
    def __init__(self, model, training_params, training) -> None:
        self.model = model

        if training:
            self.opt = Optimizer([self.model], training_params)
            self.crit = DiceBCE()

    def fit(self, xs, ys, device, **kwargs):
        self.opt.z_grad()

        if len(device) > 1:
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        xs = xs.type(torch.FloatTensor).to(_device)
        ys = ys.type(torch.FloatTensor).to(_device)

        scores = self.model(xs)
        loss = self.crit(scores, ys)

        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        return scores, loss

    def predict(self, x, device, **kwargs):
        if len(device) > 1:
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        x = x.type(torch.FloatTensor).to(_device)
        return self.model(x)
    
    
class DCPProcessor(BasicProcessor):
    def __init__(self, model, training_params, training=True) -> None:
        self.model = model
        if training:
            if 'prompt_lr' in training_params:
                prompt_lr = training_params['prompt_lr']
                self.opt = Optimizer([self.model.encoder, self.model.decoder, self.model.Last_Conv, self.model.att1, self.model.att2, self.model.att3, self.model.att4, self.model.att5], training_params, 
                                     sep_lr=prompt_lr, sep_params=[self.model.cha_promot1, self.model.cha_promot2, self.model.cha_promot3, self.model.cha_promot4, self.model.cha_promot5, self.model.pos_promot1, self.model.pos_promot2, self.model.pos_promot3, self.model.pos_promot4, self.model.pos_promot5])
            else:
                self.opt = Optimizer([self.model], training_params)
            self.crit = DiceBCE()

    def fit(self, xs, ys, device, **kwargs):
        dataset_idx = kwargs['dataset_idx']
        self.opt.z_grad()
        if len(device) > 1:
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        
        xs = xs.type(torch.FloatTensor).to(_device)
        ys = ys.type(torch.FloatTensor).to(_device)
        
        scores = self.model(xs, dataset_idx)
        loss = self.crit(scores, ys)

        loss.backward()
        
        self.opt.g_step()
        self.opt.update_lr()
        
        return scores, loss

    def predict(self, x, device, **kwargs):
        dataset_idx = kwargs['dataset_idx']
        print(dataset_idx)
        if isinstance(device, list):
            if len(device) > 1:
                _device = 'cuda'
            else:
                _device = 'cuda:{}'.format(device[0])
        else:
            _device = device
        
        x = x.type(torch.FloatTensor).to(_device)
    
        return self.model(x, dataset_idx)
   

class JTFNProcessor(BasicProcessor):
    def __init__(self, model, training_params, training=True) -> None:
        # model_params = training_params['model_params']
        # n_class = model_params['n_class']

        self.model = model
        self.steps = training_params['steps']

        if training:
            self.opt = Optimizer([self.model], training_params)
            # self.crit = DiceLoss()
            self.crit = DiceBCE()

    def fit(self, xs, ys, device, **kwargs):
        self.opt.z_grad()

        #num_domains = len(xs)
        batch_size = len(xs)

        if len(device) > 1:
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        #xs = torch.concatenate(xs, dim=0).type(torch.FloatTensor).to(_device)
        #ys = torch.concatenate(ys, dim=0).type(torch.FloatTensor).to(_device)
        xs = xs.type(torch.FloatTensor).to(_device)
        ys = ys.type(torch.FloatTensor).to(_device)
        
        ys_boundary = generate_BD(ys)
        _, _, h, w = ys.size()

        outputs = self.model(xs)
        loss = 0
        for i in range(self.steps):
            pred_seg = outputs['step_{}_seg'.format(i)]
            pred_bou = outputs['step_{}_bou'.format(i)]
            
            for j in range(len(pred_seg)):
                p_seg = F.interpolate(pred_seg[j], (h, w), mode='bilinear', align_corners=True)
                p_bou = F.interpolate(pred_bou[j], (h, w), mode='bilinear', align_corners=True)
                
                loss += self.crit(p_seg, ys) + self.crit(p_bou, ys_boundary)
        loss /= len(pred_seg)
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        
        scores = outputs['output']
        # _, C, H, W = scores.size()
        
        # scores = scores.view(num_domains, batch_size, C, H, W)
        # scores = scores.cpu().numpy()
        return scores, loss

    def predict(self, x, device, **kwargs):
        if len(device) > 1:
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        x = x.type(torch.FloatTensor).to(_device)
        outputs = self.model(x)
        
        return outputs['output']
    

class JTFNDCPProcessor(BasicProcessor):
    def __init__(self, model, training_params, training=True) -> None:
        # model_params = training_params['model_params']
        # n_class = model_params['n_class']

        self.model = model
        self.steps = training_params['steps']

        if training:
            
            self.opt = Optimizer([self.model], training_params)
            # self.crit = DiceLoss()
            self.crit = DiceBCE()

    def fit(self, xs, ys, device, **kwargs):
        dataset_idx = kwargs['dataset_idx']
        self.opt.z_grad()

        if len(device) > 1:
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        xs = xs.type(torch.FloatTensor).to(_device)
        ys = ys.type(torch.FloatTensor).to(_device)
        
        ys_boundary = generate_BD(ys)
        _, _, h, w = ys.size()

        outputs = self.model(xs, dataset_idx)
        loss = 0
        for i in range(self.steps):
            pred_seg = outputs['step_{}_seg'.format(i)]
            pred_bou = outputs['step_{}_bou'.format(i)]
            
            for j in range(len(pred_seg)):
                p_seg = F.interpolate(pred_seg[j], (h, w), mode='bilinear', align_corners=True)
                p_bou = F.interpolate(pred_bou[j], (h, w), mode='bilinear', align_corners=True)
                
                loss += self.crit(p_seg, ys) + self.crit(p_bou, ys_boundary)
        loss /= len(pred_seg)
        loss.backward()
        self.opt.g_step()
        self.opt.update_lr()

        scores = outputs['output']

        return scores, loss

    def predict(self, x, device, **kwargs):
        dataset_idx = kwargs['dataset_idx']
        if len(device) > 1:
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        x = x.type(torch.FloatTensor).to(_device)
        
        outputs = self.model(x, dataset_idx)
        
        return outputs['output']
    
    

