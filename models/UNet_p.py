import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def rand(size, val=0.01):
    out = torch.zeros(size)
    
    nn.init.uniform_(out, -val, val)
    return out

#  from medsam
def window_partition(x: torch.Tensor, window_size: int):
    B, C, H, W = x.size()
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    
    #          0  1        2                3                4               5
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows, (Hp, Wp), (Hp // window_size, Wp // window_size)

def prompt_partition(prompt: torch.Tensor, h_windows: int, w_windows: int):
    # prompt: B, C, H, W
    B, C, H, W = prompt.size()
    prompt = prompt.view(B, 1, 1, C, H, W)
    prompt = prompt.repeat((1, h_windows, w_windows, 1, 1, 1)).contiguous().view(-1, C, H, W)
    return prompt
    
def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    # windows: B * Hp // window_size * Wp // window_size, C, window_size, window_size
    Hp, Wp = pad_hw
    H, W = hw
    B = (windows.shape[0] * window_size * window_size) // (Hp * Wp)
    #                0          1                    2         3       4            5
    x = windows.view(B, Hp // window_size, Wp // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, Hp, Wp)

    if Hp > H or Wp > W:
        x = x[:, :, :H, :W].contiguous()
    return x


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        cdf = 0.5 * (1 + torch.erf(x / 2**0.5))
        return x * cdf


class OneLayerRes(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x = x + self.weight * self.conv(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, heat=False):
        B, N, C = x.shape
        out = self.norm(x)
        qkv = self.qkv(out).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.drop(out)
        out = x + out
        if heat:
            return out, attn
        return out


class MultiHeadAttention2D_POS(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, embed_dim, num_heads=8, drop_rate=0.2, embed_dim_ratio=4, stride=1, slide=0):
        super().__init__()
        self.stride = stride
        self.num_heads = num_heads

        self.slide = slide

        self.embed_dim_qk = embed_dim // embed_dim_ratio
        
        if self.embed_dim_qk % num_heads != 0:  # 保证可以整除num_heads
            self.embed_dim_qk = (self.embed_dim_qk // num_heads + 1) * num_heads

        self.embed_dim_v = embed_dim
        if self.embed_dim_v % num_heads != 0:
            self.embed_dim_v = (self.embed_dim_v // num_heads + 1) * num_heads
        
        head_dim = self.embed_dim_qk // num_heads
        # self.norm = nn.LayerNorm(embed_dim)

        self.scale = head_dim ** -0.5

        self.conv_q = nn.Conv2d(in_channels=dim_q, out_channels=self.embed_dim_qk, kernel_size=stride, padding=0, stride=stride)
        self.conv_k = nn.Conv2d(in_channels=dim_k, out_channels=self.embed_dim_qk, kernel_size=stride, padding=0, stride=stride)
        self.conv_v = nn.Conv2d(in_channels=dim_v, out_channels=self.embed_dim_v, kernel_size=stride, padding=0, stride=stride)
                
        self.drop = nn.Dropout(drop_rate)
        self.proj_out = nn.Conv2d(in_channels=self.embed_dim_v, out_channels=dim_q, kernel_size=3, padding=1)
        if self.stride > 1:
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.upsample = nn.Identity()

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, q, k, v, heat=False):
        B, _, H_q, W_q = q.size()
        _, _, H_kv, W_kv = k.size()

        H_q = H_q // self.stride
        W_q = W_q // self.stride
        H_kv = H_kv // self.stride
        W_kv = W_kv // self.stride

        proj_q = self.conv_q(q).reshape(B, self.num_heads, self.embed_dim_qk // self.num_heads, H_q * W_q).permute(0, 1, 3, 2).contiguous()
        proj_k = self.conv_k(k).reshape(B, self.num_heads, self.embed_dim_qk // self.num_heads, H_kv * W_kv).permute(0, 1, 3, 2).contiguous()
        proj_v = self.conv_v(v).reshape(B, self.num_heads, self.embed_dim_v // self.num_heads, H_kv * W_kv).permute(0, 1, 3, 2).contiguous()
        # B, self.num_heads, H * W, self.embed_dim // self.num_heads

        attn = (proj_q @ proj_k.transpose(-2, -1)).contiguous() * self.scale  # B, self.num_heads, H_q * W_q, H_kv * W_kv
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ proj_v)  # B, self.num_heads, H_q * W_q, self.embed_dim // self.num_heads
        out = out.transpose(2, 3).contiguous().reshape(B, self.embed_dim_v, H_q, W_q)

        if self.slide > 0:
            #print(out.size(), self.slide // self.stride)
            #print(q.size(), self.slide)
            out = out[:, :,  self.slide // self.stride:]
            q = q[:, :,  self.slide:]
        
        out = self.proj_out(out)
        out = self.upsample(out)
        out = self.drop(out)
        out = q + out * self.gamma
        return out


class MultiHeadAttention2D_CHA(nn.Module):
    def __init__(self, dim_q, dim_kv, stride, num_heads=8, drop_rate=0.2, slide=0):
        super().__init__()
        self.num_heads = num_heads
        self.stride = stride
        # self.scale = head_dim ** -0.5
        self.slide = slide
        self.dim_q_out = dim_q - slide

        self.conv_q = nn.Conv2d(in_channels=dim_q,  out_channels=dim_q * num_heads,  kernel_size=stride, stride=stride, groups=dim_q)
        self.conv_k = nn.Conv2d(in_channels=dim_kv, out_channels=dim_kv * num_heads, kernel_size=stride, stride=stride, groups=dim_kv)
        self.conv_v = nn.Conv2d(in_channels=dim_kv, out_channels=dim_kv * num_heads, kernel_size=stride, stride=stride, groups=dim_kv)
        
                
        self.drop = nn.Dropout(drop_rate)
        self.proj_out = nn.ConvTranspose2d(in_channels=self.dim_q_out * num_heads, out_channels=self.dim_q_out, kernel_size=stride,  stride=stride, groups=self.dim_q_out)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, q, k, v, heat=False):
        B, C_q, H_q, W_q = q.size()
        _, C_kv, H_kv, W_kv = k.size()

        proj_q = self.conv_q(q).reshape(B, self.num_heads, C_q,  -1)     # batch_size * num_heads * dim_q * (H * W)
        proj_k = self.conv_k(k).reshape(B, self.num_heads, C_kv, -1) 
        proj_v = self.conv_v(v).reshape(B, self.num_heads, C_kv, -1)     # batch_size * num_heads * dim_kv * (H * W)

        scale = proj_q.size(3) ** -0.5
        attn = (proj_q @ proj_k.transpose(-2, -1)).contiguous() * scale  # batch_size, num_heads, dim_q, dim_kv
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ proj_v)  #  batch_size, num_heads, dim_q, (H * W)
        if self.slide > 0:  # channel prompt在后头
            out = out[:, :, :-self.slide]
        out = out.reshape(B, self.num_heads * self.dim_q_out, H_q // self.stride, W_q // self.stride)
        
        out = self.proj_out(out)
        out = self.drop(out)
        out = q + out * self.gamma
        return out


class MultiHeadAttention2D_Dual2_2(nn.Module):
    def __init__(self, dim_pos, dim_cha, embed_dim, att_fusion, num_heads=8, drop_rate=0.2, embed_dim_ratio=4, stride=1, cha_slide=0, pos_slide=0, use_conv=True):
        super().__init__()
        self.pos_att = MultiHeadAttention2D_POS(dim_q=dim_pos, dim_k=dim_pos, dim_v=dim_pos, embed_dim=embed_dim, num_heads=num_heads, drop_rate=drop_rate, embed_dim_ratio=embed_dim_ratio, stride=stride, slide=pos_slide)
        self.cha_att = MultiHeadAttention2D_CHA(dim_q=dim_cha, dim_kv=dim_cha, num_heads=num_heads, drop_rate=drop_rate, slide=cha_slide, stride=stride)
        self.att_fusion = att_fusion # concat, add

        if att_fusion == 'concat':
            channel_in  = 2 * (dim_pos - cha_slide)
        if att_fusion == 'add':
            channel_in  = (dim_pos - cha_slide) 
        channel_out = dim_pos - cha_slide
        
        self.use_conv = use_conv
        if use_conv:
            self.conv_out = nn.Sequential(nn.Dropout2d(drop_rate, True), nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1))
        else:
            self.conv_out = nn.Identity()


    def forward(self, qkv_pos, qkv_cha, heat=False):
        # print(q.size())
        if qkv_cha is None:
            qkv_cha = qkv_pos
        out_pos = self.pos_att(qkv_pos, qkv_pos, qkv_pos, heat)
        out_cha = self.cha_att(qkv_cha, qkv_cha, qkv_cha, heat)

        C = out_pos.size(1)
        H = out_cha.size(2)
        
        if self.att_fusion == 'concat':
            out  = torch.cat([out_pos[:, :, -H:], out_cha[:, :C, :]], dim=1)
        if self.att_fusion == 'add':
            out = (out_pos[:, :, -H:] + out_cha[:, :C, :]) / 2 
        
        out = self.conv_out(out)
        return out



class ResMLP(MLP):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.2):
        super().__init__(in_features=in_features, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = out + x
        return out


class MHSABlock(nn.Module):
    def __init__(self, dim, num_heads=8, drop_rate=0.2) -> None:
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
        self.mlp = ResMLP(in_features=dim, hidden_features=dim*4, out_features=dim)

    def forward(self, x, heat=False):
        
        if heat:
            x, attn = self.mhsa(x, heat=True)
        else:
            x = self.mhsa(x)
        x = self.mlp(x)
        if heat:
            return x, attn
        return x


class SelfAttentionBlocks(nn.Module):
    def __init__(self, dim, block_num, num_heads=8, drop_rate=0.2):
        super().__init__()
        self.block_num = block_num
        assert self.block_num >= 1

        self.blocks = nn.ModuleList([MHSABlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
                                     for i in range(self.block_num)])

    def forward(self, x, heat=False):
        attns = []
        for blk in self.blocks:
            if heat:
                x, attn = blk(x, heat=True)
                attns.append(attn)
            else:
                x = blk(x)
        if heat:
            return x, attns
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g, F_l, F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class R2AttUNetDecoder(nn.Module):
    def __init__(self, channels, t=2):
        super(R2AttUNetDecoder,self).__init__()
        
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.Up5 = up_conv(ch_in=channels[4], ch_out=channels[3])
        self.Att5 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=channels[3]//2)
        self.Up_RRCNN5 = RRCNN_block(ch_in=2 * channels[3], ch_out=channels[3], t=t)
        
        self.Up4 = up_conv(ch_in=channels[3], ch_out=channels[2])
        self.Att4 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[2]//2)
        self.Up_RRCNN4 = RRCNN_block(ch_in=2 * channels[2], ch_out=channels[2], t=t)
        
        self.Up3 = up_conv(ch_in=channels[2], ch_out=channels[1])
        self.Att3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2)
        self.Up_RRCNN3 = RRCNN_block(ch_in=2 * channels[1], ch_out=channels[1], t=t)
        
        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[0])
        self.Att2 = Attention_block(F_g=channels[0], F_l=channels[0], F_int=channels[0]//2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=2 * channels[0], ch_out=channels[0], t=t)

    def forward(self, x1, x2, x3, x4, x5):
        
        out = self.Up5(x5)
        x4_att = self.Att5(g=out, x=x4)
        out = torch.cat((x4_att, out),dim=1)
        out = self.Up_RRCNN5(out)
        
        out = self.Up4(out)
        x3_att = self.Att4(g=out, x=x3)
        out = torch.cat((x3_att, out),dim=1)
        out = self.Up_RRCNN4(out)

        out = self.Up3(out)
        x2_att = self.Att3(g=out, x=x2)
        out = torch.cat((x2_att, out),dim=1)
        out = self.Up_RRCNN3(out)

        out = self.Up2(out)
        x1_att = self.Att2(g=out, x=x1)
        out = torch.cat((x1_att, out),dim=1)
        out = self.Up_RRCNN2(out)

        out = self.Upsample(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0, bias=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.activate = nn.LeakyReLU(negative_slope=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate(out)
        return out


class UNetDecoder(nn.Module):
    def __init__(self, channels):
        super(UNetDecoder,self).__init__()
        
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.Up5 = up_conv(ch_in=channels[4], ch_out=channels[3])
        self.conv5 = ConvBlock(ch_in=2 * channels[3], ch_out=channels[3], kernel_size=3, stride=1, padding=1)
        
        self.Up4 = up_conv(ch_in=channels[3], ch_out=channels[2])
        self.conv4 = ConvBlock(ch_in=2 * channels[2], ch_out=channels[2], kernel_size=3, stride=1, padding=1)
        
        self.Up3 = up_conv(ch_in=channels[2], ch_out=channels[1])
        self.conv3 = ConvBlock(ch_in=2 * channels[1], ch_out=channels[1], kernel_size=3, stride=1, padding=1)
        
        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[0])
        self.conv2 = ConvBlock(ch_in=2 * channels[0], ch_out=channels[0], kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4, x5):
        
        out = self.Up5(x5)
        out = torch.cat((x4, out),dim=1)
        out = self.conv5(out)
        
        out = self.Up4(out)
        out = torch.cat((x3, out),dim=1)
        out = self.conv4(out)

        out = self.Up3(out)
        out = torch.cat((x2, out),dim=1)
        out = self.conv3(out)

        out = self.Up2(out)
        out = torch.cat((x1, out),dim=1)
        out = self.conv2(out)

        out = self.Upsample(out)

        return out


class U_Net_P(nn.Module):
    def __init__(self, encoder, decoder, output_ch, num_classes):
        super(U_Net_P, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
        self.Last_Conv = nn.Conv2d(output_ch, num_classes, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # encoding path
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        x = self.Last_Conv(x)

        return x


class Prompt_U_Net_P_DCP(nn.Module):
    def __init__(self, encoder, decoder, output_ch, num_classes, dataset_idx, encoder_channels, prompt_init, pos_promot_channels, cha_promot_channels, embed_ratio, strides, local_window_sizes, att_fusion, use_conv):
        super(Prompt_U_Net_P_DCP, self).__init__()
        self.dataset_idx = dataset_idx
        self.local_window_sizes = local_window_sizes

        self.encoder = encoder
        self.decoder = decoder
    
        self.Last_Conv = nn.Conv2d(output_ch, num_classes, kernel_size=3, stride=1, padding=1)
        if prompt_init == 'zero':
            p_init = torch.zeros
        elif prompt_init == 'one':
            p_init = torch.ones
        elif prompt_init == 'rand':
            p_init = rand        
  
        else:
            raise Exception(prompt_init)
        
        self.pos_promot_channels = pos_promot_channels
        pos_p1 = p_init((1, encoder_channels[0], pos_promot_channels[0], local_window_sizes[0]))
        pos_p2 = p_init((1, encoder_channels[1], pos_promot_channels[1], local_window_sizes[1]))
        pos_p3 = p_init((1, encoder_channels[2], pos_promot_channels[2], local_window_sizes[2]))
        pos_p4 = p_init((1, encoder_channels[3], pos_promot_channels[3], local_window_sizes[3]))
        pos_p5 = p_init((1, encoder_channels[4], pos_promot_channels[4], local_window_sizes[4]))
        self.pos_promot1 = nn.ParameterDict({str(k): nn.Parameter(pos_p1.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot2 = nn.ParameterDict({str(k): nn.Parameter(pos_p2.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot3 = nn.ParameterDict({str(k): nn.Parameter(pos_p3.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot4 = nn.ParameterDict({str(k): nn.Parameter(pos_p4.detach().clone(), requires_grad=True) for k in dataset_idx})
        self.pos_promot5 = nn.ParameterDict({str(k): nn.Parameter(pos_p5.detach().clone(), requires_grad=True) for k in dataset_idx})
        
        self.cha_promot_channels = cha_promot_channels
        cha_p1 = p_init((1, cha_promot_channels[0], local_window_sizes[0], local_window_sizes[0]))
        cha_p2 = p_init((1, cha_promot_channels[1], local_window_sizes[1], local_window_sizes[1]))
        cha_p3 = p_init((1, cha_promot_channels[2], local_window_sizes[2], local_window_sizes[2]))
        cha_p4 = p_init((1, cha_promot_channels[3], local_window_sizes[3], local_window_sizes[3]))
        cha_p5 = p_init((1, cha_promot_channels[4], local_window_sizes[4], local_window_sizes[4]))
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
        cha_promots1, cha_promots2, cha_promots3, cha_promots4, cha_promots5 = self.get_cha_prompts(dataset_idx=dataset_idx, batch_size=x.size(0))
        pos_promots1, pos_promots2, pos_promots3, pos_promots4, pos_promots5 = self.get_pos_prompts(dataset_idx=dataset_idx, batch_size=x.size(0))
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

        x = self.decoder(x1, x2, x3, x4, x5)
        x = self.Last_Conv(x)

        return x

