import torch.optim as optim
import torch.nn as nn
import torch
import itertools


def add_full_model_gradient_clipping(optim, clip_norm_val):

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    return FullModelGradientClippingOptimizer


class Optimizer(object):
    def __init__(self, models, training_params, sep_lr=None, sep_params=None, gradient_clip=0):
        
        params = []
        for model in models:
            if isinstance(model, nn.Parameter):
                params += [model]
            else:
                params += list(model.parameters())
        if sep_lr is not None:
            print(sep_lr)
            add_params = []
            for model in sep_params:
                if isinstance(model, nn.Parameter):
                    add_params += [model]
                else:
                    add_params += list(model.parameters())
            params = [{'params': params}, 
                      {'params': add_params, 'lr': sep_lr}]


        self.lr = training_params['lr']
        self.weight_decay = training_params['weight_decay']
        method = training_params['optimizer']
        
    
        if method == 'SGD':
            self.momentum = training_params['momentum']
            if gradient_clip > 0:
                self.optim = add_full_model_gradient_clipping(optim.SGD, gradient_clip)(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            else:
                self.optim = optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif method == 'AdamW':
            self.optim = optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise Exception('{} is not supported'.format(method))

        schedule_name = training_params['lr_schedule']
        schedule_params = training_params['schedule_params']
        if schedule_name == 'CosineAnnealingLR':
            schedule_params['T_max'] = training_params['inter_val'] * 4
        self.lr_schedule = getattr(optim.lr_scheduler, schedule_name)(self.optim, **schedule_params)
        
    def update_lr(self):
        self.lr_schedule.step()
    
    def z_grad(self):
        self.optim.zero_grad()

    def g_step(self):
        self.optim.step()

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']
