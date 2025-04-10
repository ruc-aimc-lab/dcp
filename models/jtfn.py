# ICCV2021, Joint Topology-preserving and Feature-refinement Network for Curvilinear Structure Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet_p import MultiHeadAttention2D_Dual2_2, rand, window_partition, window_unpartition, prompt_partition, OneLayerRes




class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(1, 1, kernel_size=(5, 5), padding=(2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        out = self.activate(out)
        return out


class GAU(nn.Module):
    def __init__(self, in_channels, use_gau=True, reduce_dim=False, out_channels=None):
        super(GAU, self).__init__()
        self.use_gau = use_gau
        self.reduce_dim = reduce_dim

        if self.reduce_dim:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            in_channels = out_channels

        if self.use_gau:

            self.sa = SpatialAttention()
            self.ca = ChannelAttention(in_channels)

            self.reset_gate = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, y):
        if self.reduce_dim:
            x = self.down_conv(x)

        if self.use_gau:
            y = F.interpolate(y, x.shape[-2:], mode='bilinear', align_corners=True)

            comx = x * y
            resx = x * (1 - y) # bs, c, h, w

            x_sa = self.sa(resx) # bs, 1, h, w
            x_ca = self.ca(resx) # bs, c, 1, 1

            O = self.reset_gate(comx)
            M = x_sa * x_ca

            RF = M * x + (1 - M) * O
        else:
            RF = x
        return RF


class FIM(nn.Module):

    def __init__(self, in_channels, out_channels, f_channels, use_topo=True, up=True, bottom=False):
        super(FIM, self).__init__()
        self.use_topo = use_topo
        self.up = up
        self.bottom = bottom

        if self.up:
            self.up_s = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_t = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up_s = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.decoder_s = nn.Sequential(
            nn.Conv2d(out_channels + f_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        '''self.inner_s = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )'''

        if self.bottom:
            self.st = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        if self.use_topo:
            self.decoder_t = nn.Sequential(
                nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.s_to_t = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.t_to_s = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.res_s = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            '''self.inner_t = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )'''

    def forward(self, x_s, x_t, rf):
        if self.use_topo:
            if self.bottom:
                x_t = self.st(x_t)
            #bs, c, h, w = x_s.shape
            x_s = self.up_s(x_s)
            x_t = self.up_t(x_t)

            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x_t = F.pad(x_t, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            rf_s = torch.cat((x_s, rf), dim=1)
            s = self.decoder_s(rf_s)
            s_t = self.s_to_t(s)

            t = torch.cat((x_t, s_t), dim=1)
            x_t = self.decoder_t(t)
            t_s = self.t_to_s(x_t)

            s_res = self.res_s(torch.cat((s, t_s), dim=1))

            x_s = s + s_res
            # t_cls = self.inner_t(x_t)
            # s_cls = self.inner_s(x_s)
        else:
            x_s = self.up_s(x_s)
            #x_b = self.up_b(x_b)
            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            rf_s = torch.cat((x_s, rf), dim=1)
            s = self.decoder_s(rf_s)
            x_s = s
            x_t = x_s
            #t_cls = None
            #s_cls = self.inner_s(x_s)
        return x_s, x_t


class JTFNDecoder(nn.Module):
    def __init__(self, channels, use_topo) -> None:
        super().__init__()
        self.skip_blocks = []
        for i in range(5):
            self.skip_blocks.append(GAU(channels[i], use_gau=True, reduce_dim=False, out_channels=channels[i]))
        self.fims = []
        index = 3
        for i in range(4):
            if i == index:
                self.fims.append(FIM(channels[i+1], channels[i], channels[i], use_topo=use_topo, up=True, bottom=True))
            else:
                self.fims.append(FIM(channels[i+1], channels[i], channels[i], use_topo=use_topo, up=True, bottom=False))
        self.skip_blocks = nn.ModuleList(self.skip_blocks)
        self.fims = nn.ModuleList(self.fims)

    def forward(self, x1, x2, x3, x4, x5, y):
        x1 = self.skip_blocks[0](x1, y)
        x2 = self.skip_blocks[1](x2, y)
        x3 = self.skip_blocks[2](x3, y)
        x4 = self.skip_blocks[3](x4, y)
        x5 = self.skip_blocks[4](x5, y)
        
        x5_seg, x5_bou = x5, x5
        
        x4_seg, x4_bou = self.fims[3](x5_seg, x5_bou, x4)
        x3_seg, x3_bou = self.fims[2](x4_seg, x4_bou, x3)
        x2_seg, x2_bou = self.fims[1](x3_seg, x3_bou, x2)
        x1_seg, x1_bou = self.fims[0](x2_seg, x2_bou, x1)
        
        
        return [x1_seg, x2_seg, x3_seg, x4_seg], [x1_bou, x2_bou, x3_bou, x4_bou]
        

class JTFN(nn.Module):
    def __init__(self, encoder, decoder, channels, num_classes, steps) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.steps = steps
        
        self.conv_seg1_head = nn.Conv2d(channels[0], num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_seg2_head = nn.Conv2d(channels[1], num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_seg3_head = nn.Conv2d(channels[2], num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_seg4_head = nn.Conv2d(channels[3], num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_bou1_head = nn.Conv2d(channels[0], num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_bou2_head = nn.Conv2d(channels[1], num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_bou3_head = nn.Conv2d(channels[2], num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_bou4_head = nn.Conv2d(channels[3], num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape
        y = torch.zeros([B, self.num_classes, H, W], device=x.device)
        
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        outputs = {}
        for i in range(self.steps):
            segs, bous = self.decoder(x1, x2, x3, x4, x5, y)
            x1_seg, x2_seg, x3_seg, x4_seg = segs
            x1_bou, x2_bou, x3_bou, x4_bou = bous
            
            x1_seg = self.conv_seg1_head(x1_seg)
            x2_seg = self.conv_seg2_head(x2_seg)
            x3_seg = self.conv_seg3_head(x3_seg)
            x4_seg = self.conv_seg4_head(x4_seg)
            
            x1_bou = self.conv_bou1_head(x1_bou)
            x2_bou = self.conv_bou2_head(x2_bou)
            x3_bou = self.conv_bou3_head(x3_bou)
            x4_bou = self.conv_bou4_head(x4_bou)
            
            y = x1_seg
            outputs['step_{}_seg'.format(i)] = [x1_seg, x2_seg, x3_seg, x4_seg]
            outputs['step_{}_bou'.format(i)] = [x1_bou, x2_bou, x3_bou, x4_bou]
        y = self.upsample(y)
        outputs['output'] = y
        return outputs

    def encoder_forward(self, x, dataset_idx):
        # efficient net
        x = self.encoder.conv_stem(x)
        x = self.encoder.bn1(x)
        features = []
        if 0 in self.encoder._stage_out_idx:
            features.append(x)  # add stem out
        for i in range(len(self.encoder.blocks)):
            for j, l in enumerate(self.encoder.blocks[i]):
                if j == len(self.encoder.blocks[i]) - 1 and i + 1 in self.encoder._stage_out_idx:
                    x = l(x, dataset_idx)
                else:
                    x = l(x)
            if i + 1 in self.encoder._stage_out_idx:
                features.append(x)
        return features

           

class JTFN_DCP(JTFN):
    def __init__(self, encoder, decoder, channels, num_classes, steps, dataset_idx, 
                 local_window_sizes, encoder_channels, pos_promot_channels, cha_promot_channels, 
                 embed_ratio, strides, att_fusion, use_conv) -> None:
        super().__init__(encoder, decoder, channels, num_classes, steps)
        self.dataset_idx = dataset_idx
        self.local_window_sizes = local_window_sizes

        self.pos_promot_channels = pos_promot_channels
        pos_p1 = rand((1, encoder_channels[0], pos_promot_channels[0], local_window_sizes[0]), val=3. / encoder_channels[0] ** 0.5)
        pos_p2 = rand((1, encoder_channels[1], pos_promot_channels[1], local_window_sizes[1]), val=3. / encoder_channels[1] ** 0.5)
        pos_p3 = rand((1, encoder_channels[2], pos_promot_channels[2], local_window_sizes[2]), val=3. / encoder_channels[2] ** 0.5)
        pos_p4 = rand((1, encoder_channels[3], pos_promot_channels[3], local_window_sizes[3]), val=3. / encoder_channels[3] ** 0.5)
        pos_p5 = rand((1, encoder_channels[4], pos_promot_channels[4], local_window_sizes[4]), val=3. / encoder_channels[4] ** 0.5)
        self.pos_promot1 = nn.ParameterDict({str(k): nn.Parameter(pos_p1.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot2 = nn.ParameterDict({str(k): nn.Parameter(pos_p2.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot3 = nn.ParameterDict({str(k): nn.Parameter(pos_p3.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot4 = nn.ParameterDict({str(k): nn.Parameter(pos_p4.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot5 = nn.ParameterDict({str(k): nn.Parameter(pos_p5.detach().clone(), requires_grad=True) for k in dataset_idx})
        
        self.cha_promot_channels = cha_promot_channels
        cha_p1 = rand((1, cha_promot_channels[0], local_window_sizes[0], local_window_sizes[0]), val=3. / local_window_sizes[0])
        cha_p2 = rand((1, cha_promot_channels[1], local_window_sizes[1], local_window_sizes[1]), val=3. / local_window_sizes[1])
        cha_p3 = rand((1, cha_promot_channels[2], local_window_sizes[2], local_window_sizes[2]), val=3. / local_window_sizes[2])
        cha_p4 = rand((1, cha_promot_channels[3], local_window_sizes[3], local_window_sizes[3]), val=3. / local_window_sizes[3])
        cha_p5 = rand((1, cha_promot_channels[4], local_window_sizes[4], local_window_sizes[4]), val=3. / local_window_sizes[4])
        self.cha_promot1 = nn.ParameterDict({str(k): nn.Parameter(cha_p1.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.cha_promot2 = nn.ParameterDict({str(k): nn.Parameter(cha_p2.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.cha_promot3 = nn.ParameterDict({str(k): nn.Parameter(cha_p3.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.cha_promot4 = nn.ParameterDict({str(k): nn.Parameter(cha_p4.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.cha_promot5 = nn.ParameterDict({str(k): nn.Parameter(cha_p5.detach().clone(), requires_grad=True) for k in dataset_idx})

        self.strides = strides
        
        self.att1 = MultiHeadAttention2D_Dual2_2(dim_pos=encoder_channels[0], dim_cha=encoder_channels[0] + cha_promot_channels[0], embed_dim=encoder_channels[0], att_fusion=att_fusion, num_heads=8, embed_dim_ratio=embed_ratio, stride=strides[0], pos_slide=0, cha_slide=0, use_conv=use_conv)
        self.att2 = MultiHeadAttention2D_Dual2_2(dim_pos=encoder_channels[1], dim_cha=encoder_channels[1] + cha_promot_channels[1], embed_dim=encoder_channels[1], att_fusion=att_fusion, num_heads=8, embed_dim_ratio=embed_ratio, stride=strides[1], pos_slide=0, cha_slide=0, use_conv=use_conv)
        self.att3 = MultiHeadAttention2D_Dual2_2(dim_pos=encoder_channels[2], dim_cha=encoder_channels[2] + cha_promot_channels[2], embed_dim=encoder_channels[2], att_fusion=att_fusion, num_heads=8, embed_dim_ratio=embed_ratio, stride=strides[2], pos_slide=0, cha_slide=0, use_conv=use_conv)
        self.att4 = MultiHeadAttention2D_Dual2_2(dim_pos=encoder_channels[3], dim_cha=encoder_channels[3] + cha_promot_channels[3], embed_dim=encoder_channels[3], att_fusion=att_fusion, num_heads=8, embed_dim_ratio=embed_ratio, stride=strides[3], pos_slide=0, cha_slide=0, use_conv=use_conv)
        self.att5 = MultiHeadAttention2D_Dual2_2(dim_pos=encoder_channels[4], dim_cha=encoder_channels[4] + cha_promot_channels[4], embed_dim=encoder_channels[4], att_fusion=att_fusion, num_heads=8, embed_dim_ratio=embed_ratio, stride=strides[4], pos_slide=0, cha_slide=0, use_conv=use_conv)
        
    def get_cha_prompts(self, dataset_idx, batch_size):
        if len(dataset_idx) != batch_size:
            raise Exception(dataset_idx, self.dataset_idx, batch_size)
        # print(dataset_idx, '***')
        promots1 = torch.concatenate([self.cha_promot1[str(i)] for i in dataset_idx], dim=0)
        promots2 = torch.concatenate([self.cha_promot2[str(i)] for i in dataset_idx], dim=0)
        promots3 = torch.concatenate([self.cha_promot3[str(i)] for i in dataset_idx], dim=0)
        promots4 = torch.concatenate([self.cha_promot4[str(i)] for i in dataset_idx], dim=0)
        promots5 = torch.concatenate([self.cha_promot5[str(i)] for i in dataset_idx], dim=0)
        return promots1, promots2, promots3, promots4, promots5
    
    def get_pos_prompts(self, dataset_idx, batch_size):
        if len(dataset_idx) != batch_size:
            raise Exception(dataset_idx, self.dataset_idx)
        # print(dataset_idx, '***')
        promots1 = torch.concatenate([self.pos_promot1[str(i)] for i in dataset_idx], dim=0)
        promots2 = torch.concatenate([self.pos_promot2[str(i)] for i in dataset_idx], dim=0)
        promots3 = torch.concatenate([self.pos_promot3[str(i)] for i in dataset_idx], dim=0)
        promots4 = torch.concatenate([self.pos_promot4[str(i)] for i in dataset_idx], dim=0)
        promots5 = torch.concatenate([self.pos_promot5[str(i)] for i in dataset_idx], dim=0)
        return promots1, promots2, promots3, promots4, promots5      

    def forward(self, x, dataset_idx, return_features=False):
        if isinstance(dataset_idx, torch.Tensor):
            dataset_idx = list(dataset_idx.cpu().numpy())
        #print(dataset_idx)
        cha_promots1, cha_promots2, cha_promots3, cha_promots4, cha_promots5 = self.get_cha_prompts(dataset_idx=dataset_idx, batch_size=x.size(0))
        pos_promots1, pos_promots2, pos_promots3, pos_promots4, pos_promots5 = self.get_pos_prompts(dataset_idx=dataset_idx, batch_size=x.size(0))
        
        B, C, H, W = x.shape
        y = torch.zeros([B, self.num_classes, H, W], device=x.device)
        
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        if return_features:
            pre_x1, pre_x2, pre_x3, pre_x4, pre_x5 = x1.detach().clone(), x2.detach().clone(), x3.detach().clone(), x4.detach().clone(), x5.detach().clone()
        h1, w1 = x1.size()[2:]
        h2, w2 = x2.size()[2:]
        h3, w3 = x3.size()[2:]
        h4, w4 = x4.size()[2:]
        h5, w5 = x5.size()[2:]
        x1, (Hp1, Wp1), (h_win1, w_win1) = window_partition(x1, self.local_window_sizes[0])
        x2, (Hp2, Wp2), (h_win2, w_win2) = window_partition(x2, self.local_window_sizes[1])
        x3, (Hp3, Wp3), (h_win3, w_win3) = window_partition(x3, self.local_window_sizes[2])
        x4, (Hp4, Wp4), (h_win4, w_win4) = window_partition(x4, self.local_window_sizes[3])
        x5, (Hp5, Wp5), (h_win5, w_win5) = window_partition(x5, self.local_window_sizes[4])

        cha_promots1 = prompt_partition(cha_promots1, h_win1, w_win1)
        cha_promots2 = prompt_partition(cha_promots2, h_win2, w_win2)
        cha_promots3 = prompt_partition(cha_promots3, h_win3, w_win3)
        cha_promots4 = prompt_partition(cha_promots4, h_win4, w_win4)
        cha_promots5 = prompt_partition(cha_promots5, h_win5, w_win5)

        pos_promots1 = prompt_partition(pos_promots1, h_win1, w_win1)
        pos_promots2 = prompt_partition(pos_promots2, h_win2, w_win2)
        pos_promots3 = prompt_partition(pos_promots3, h_win3, w_win3)
        pos_promots4 = prompt_partition(pos_promots4, h_win4, w_win4)
        pos_promots5 = prompt_partition(pos_promots5, h_win5, w_win5)

        #print(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        cha_x1, cha_x2, cha_x3, cha_x4, cha_x5 = torch.cat([x1, cha_promots1], dim=1), torch.cat([x2, cha_promots2], dim=1), torch.cat([x3, cha_promots3], dim=1), torch.cat([x4, cha_promots4], dim=1), torch.cat([x5, cha_promots5], dim=1)
        pos_x1, pos_x2, pos_x3, pos_x4, pos_x5 = torch.cat([pos_promots1, x1], dim=2), torch.cat([pos_promots2, x2], dim=2), torch.cat([pos_promots3, x3], dim=2), torch.cat([pos_promots4, x4], dim=2), torch.cat([pos_promots5, x5], dim=2)
        
        #print(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        x1, x2, x3, x4, x5 = self.att1(pos_x1, cha_x1), self.att2(pos_x2, cha_x2), self.att3(pos_x3, cha_x3), self.att4(pos_x4, cha_x4), self.att5(pos_x5, cha_x5)
        
        x1 = window_unpartition(x1, self.local_window_sizes[0], (Hp1, Wp1), (h1, w1))
        x2 = window_unpartition(x2, self.local_window_sizes[1], (Hp2, Wp2), (h2, w2))
        x3 = window_unpartition(x3, self.local_window_sizes[2], (Hp3, Wp3), (h3, w3))
        x4 = window_unpartition(x4, self.local_window_sizes[3], (Hp4, Wp4), (h4, w4))
        x5 = window_unpartition(x5, self.local_window_sizes[4], (Hp5, Wp5), (h5, w5))
        
        if return_features:
            pro_x1, pro_x2, pro_x3, pro_x4, pro_x5 = x1.detach().clone(), x2.detach().clone(), x3.detach().clone(), x4.detach().clone(), x5.detach().clone()
            
            return (pre_x1, pre_x2, pre_x3, pre_x4, pre_x5), (pro_x1, pro_x2, pro_x3, pro_x4, pro_x5)
        
        outputs = {}
        for i in range(self.steps):
            segs, bous = self.decoder(x1, x2, x3, x4, x5, y)
            x1_seg, x2_seg, x3_seg, x4_seg = segs
            x1_bou, x2_bou, x3_bou, x4_bou = bous
            
            x1_seg = self.conv_seg1_head(x1_seg)
            x2_seg = self.conv_seg2_head(x2_seg)
            x3_seg = self.conv_seg3_head(x3_seg)
            x4_seg = self.conv_seg4_head(x4_seg)
            
            x1_bou = self.conv_bou1_head(x1_bou)
            x2_bou = self.conv_bou2_head(x2_bou)
            x3_bou = self.conv_bou3_head(x3_bou)
            x4_bou = self.conv_bou4_head(x4_bou)
            
            y = x1_seg
            outputs['step_{}_seg'.format(i)] = [x1_seg, x2_seg, x3_seg, x4_seg]
            outputs['step_{}_bou'.format(i)] = [x1_bou, x2_bou, x3_bou, x4_bou]
        y = self.upsample(y)
        outputs['output'] = y
        return outputs


