from .processor import Processor, DCPProcessor, JTFNProcessor, JTFNDCPProcessor
from .UNet_p import U_Net_P, R2AttUNetDecoder, UNetDecoder, Prompt_U_Net_P_DCP
from .jtfn import JTFN, JTFNDecoder, JTFN_DCP
from .backbones import build_backbone


def build_model(model_name, model_params, training, dataset_idx, pretrained):
    model = getattr(Models, model_name)(model_params=model_params, training=training, dataset_idx=dataset_idx, pretrained=pretrained)
    return model


class Models(object):
    @staticmethod
    def effi_b3_p_unet(model_params, training, dataset_idx, pretrained=True):
        n_class = model_params['n_class']
        channels = (24, 12, 40, 120, 384)

        encoder = build_backbone('efficientnet_b3_p', pretrained=pretrained)
        decoder = UNetDecoder(channels=channels)

        seg_net = U_Net_P(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class)
        model = Processor(model=seg_net, training_params=model_params, training=training)
        return model
    
    
    @staticmethod
    def effi_b3_p_r2attunet(model_params, training, dataset_idx, pretrained=True):
        n_class = model_params['n_class']
        channels = (24, 12, 40, 120, 384)

        encoder = build_backbone('efficientnet_b3_p', pretrained=pretrained)
        decoder = R2AttUNetDecoder(channels=channels)

        seg_net = U_Net_P(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class)
        model = Processor(model=seg_net, training_params=model_params, training=training)
        return model
    
    @staticmethod
    def effi_b3_p_jtfn(model_params, training, dataset_idx, pretrained=True):
        n_class = model_params['n_class']
        channels = (24, 12, 40, 120, 384)
        steps = model_params['steps']
        
        encoder = build_backbone('efficientnet_b3_p')
        decoder = JTFNDecoder(channels=channels, use_topo=True)
        
        seg_net = JTFN(encoder=encoder, decoder=decoder, channels=channels, num_classes=n_class, steps=steps)
        model = JTFNProcessor(model=seg_net, training_params=model_params, training=training)
        return model

    
    @staticmethod
    def prompt_effi_b3_p_unet_dcp(model_params, training, dataset_idx, pretrained=True):
        n_class = model_params['n_class']
        channels = [24, 12, 40, 120, 384]

        cha_promot_channels = model_params['cha_promot_channels']
        pos_promot_channels = model_params['pos_promot_channels']
        local_window_sizes = model_params['local_window_sizes']
        att_fusion = model_params['att_fusion']
        prompt_init = model_params.get('prompt_init', 'rand')  # rand, zero, one
        embed_ratio = model_params['embed_ratio']
        strides = model_params['strides']
        use_conv = model_params['use_conv']

        encoder = build_backbone('efficientnet_b3_p', pretrained=pretrained)
        decoder = UNetDecoder(channels=channels)

        seg_net = Prompt_U_Net_P_DCP(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class, 
                                         dataset_idx=dataset_idx, encoder_channels=channels, prompt_init=prompt_init, 
                                         cha_promot_channels=cha_promot_channels, pos_promot_channels=pos_promot_channels,
                                         embed_ratio=embed_ratio, strides=strides, local_window_sizes=local_window_sizes,
                                         att_fusion=att_fusion, use_conv=use_conv)

        model = DCPProcessor(model=seg_net, training_params=model_params, training=training)
        return model
    
    @staticmethod
    def prompt_effi_b3_p_r2attunet_dcp(model_params, training, dataset_idx, pretrained=True):
        n_class = model_params['n_class']
        channels = [24, 12, 40, 120, 384]

        cha_promot_channels = model_params['cha_promot_channels']
        pos_promot_channels = model_params['pos_promot_channels']
        local_window_sizes = model_params['local_window_sizes']
        att_fusion = model_params['att_fusion']
        prompt_init = model_params.get('prompt_init', 'rand')  # rand, zero, one
        embed_ratio = model_params['embed_ratio']
        strides = model_params['strides']
        use_conv = model_params['use_conv']

        encoder = build_backbone('efficientnet_b3_p', pretrained=pretrained)
        decoder = R2AttUNetDecoder(channels=channels)

        seg_net = Prompt_U_Net_P_DCP(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class, 
                                         dataset_idx=dataset_idx, encoder_channels=channels, prompt_init=prompt_init, 
                                         cha_promot_channels=cha_promot_channels, pos_promot_channels=pos_promot_channels,
                                         embed_ratio=embed_ratio, strides=strides, local_window_sizes=local_window_sizes,
                                         att_fusion=att_fusion, use_conv=use_conv)

        model = DCPProcessor(model=seg_net, training_params=model_params, training=training)
        return model

        
    @staticmethod
    def prompt_effi_b3_p_jtfn_dcp(model_params, training, dataset_idx, pretrained=True):
        n_class = model_params['n_class']
        steps = model_params['steps']
        channels = [24, 12, 40, 120, 384]

        cha_promot_channels = model_params['cha_promot_channels']
        pos_promot_channels = model_params['pos_promot_channels']
        local_window_sizes = model_params['local_window_sizes']
        att_fusion = model_params['att_fusion']
        embed_ratio = model_params['embed_ratio']
        strides = model_params['strides']
        use_conv = model_params['use_conv']

        encoder = build_backbone('efficientnet_b3_p', pretrained=pretrained)
        decoder = JTFNDecoder(channels=channels, use_topo=True)
        seg_net = JTFN_DCP(encoder=encoder, decoder=decoder, channels=channels, num_classes=n_class, steps=steps,
                                         dataset_idx=dataset_idx, local_window_sizes=local_window_sizes, 
                                         encoder_channels=channels, 
                                         cha_promot_channels=cha_promot_channels, pos_promot_channels=pos_promot_channels,
                                         embed_ratio=embed_ratio, strides=strides, 
                                         att_fusion=att_fusion, use_conv=use_conv)

        model = JTFNDCPProcessor(model=seg_net, training_params=model_params, training=training)
        return model
    
    