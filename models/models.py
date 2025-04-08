from .processor import Processor, DualPromptProcessor, JTFNProcessor, JTFNDualPromptProcessor
from .UNet_p import U_Net_P, R2AttUNetDecoder, UNetDecoder, Prompt_U_Net_P_DualAtt2_2
from .jtfn import JTFN, JTFNDecoder, JTFN_DualAtt2_2
from .backbones import build_backbone


def build_model(model_name, training_params, training, dataset_idx):
    model = getattr(Models, model_name)(training_params=training_params, training=training, dataset_idx=dataset_idx)
    return model


class Models(object):
    @staticmethod
    def effi_b3_p_unet(training_params, training, dataset_idx):
        n_class = training_params['n_class']
        channels = (24, 12, 40, 120, 384)

        encoder = build_backbone('efficientnet_b3_p')
        decoder = UNetDecoder(channels=channels)

        seg_net = U_Net_P(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class)
        model = Processor(model=seg_net, training_params=training_params, training=training)
        return model
    
    
    @staticmethod
    def effi_b3_p_r2attunet(training_params, training, dataset_idx):
        n_class = training_params['n_class']
        channels = (24, 12, 40, 120, 384)

        encoder = build_backbone('efficientnet_b3_p')
        decoder = R2AttUNetDecoder(channels=channels)

        seg_net = U_Net_P(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class)
        model = Processor(model=seg_net, training_params=training_params, training=training)
        return model
    
    @staticmethod
    def effi_b3_p_jtfn(training_params, training, dataset_idx):
        n_class = training_params['n_class']
        channels = (24, 12, 40, 120, 384)
        steps = training_params['steps']
        
        encoder = build_backbone('efficientnet_b3_p')
        decoder = JTFNDecoder(channels=channels, use_topo=True)
        
        seg_net = JTFN(encoder=encoder, decoder=decoder, channels=channels, num_classes=n_class, steps=steps)
        model = JTFNProcessor(model=seg_net, training_params=training_params, training=training)
        return model

    
    @staticmethod
    def prompt_effi_b3_p_unet_dcp(training_params, training, dataset_idx):
        n_class = training_params['n_class']
        channels = [24, 12, 40, 120, 384]

        cha_promot_channels = training_params['cha_promot_channels']
        pos_promot_channels = training_params['pos_promot_channels']
        local_window_sizes = training_params['local_window_sizes']
        att_fusion = training_params['att_fusion']
        prompt_init = training_params['prompt_init']
        embed_ratio = training_params['embed_ratio']
        strides = training_params['strides']
        use_conv = training_params['use_conv']

        encoder = build_backbone('efficientnet_b3_p')
        decoder = UNetDecoder(channels=channels)

        seg_net = Prompt_U_Net_P_DualAtt2_2(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class, 
                                         dataset_idx=dataset_idx, encoder_channels=channels, prompt_init=prompt_init, 
                                         cha_promot_channels=cha_promot_channels, pos_promot_channels=pos_promot_channels,
                                         embed_ratio=embed_ratio, strides=strides, local_window_sizes=local_window_sizes,
                                         att_fusion=att_fusion, use_conv=use_conv)

        model = DualPromptProcessor(model=seg_net, training_params=training_params, training=training)
        return model
    
    @staticmethod
    def prompt_effi_b3_p_r2attunet_dcp(training_params, training, dataset_idx):
        n_class = training_params['n_class']
        channels = [24, 12, 40, 120, 384]

        cha_promot_channels = training_params['cha_promot_channels']
        pos_promot_channels = training_params['pos_promot_channels']
        local_window_sizes = training_params['local_window_sizes']
        att_fusion = training_params['att_fusion']
        prompt_init = training_params['prompt_init']
        embed_ratio = training_params['embed_ratio']
        strides = training_params['strides']
        use_conv = training_params['use_conv']

        encoder = build_backbone('efficientnet_b3_p')
        decoder = R2AttUNetDecoder(channels=channels)

        seg_net = Prompt_U_Net_P_DualAtt2_2(encoder=encoder, decoder=decoder, output_ch=channels[0], num_classes=n_class, 
                                         dataset_idx=dataset_idx, encoder_channels=channels, prompt_init=prompt_init, 
                                         cha_promot_channels=cha_promot_channels, pos_promot_channels=pos_promot_channels,
                                         embed_ratio=embed_ratio, strides=strides, local_window_sizes=local_window_sizes,
                                         att_fusion=att_fusion, use_conv=use_conv)

        model = DualPromptProcessor(model=seg_net, training_params=training_params, training=training)
        return model

        
    @staticmethod
    def prompt_effi_b3_p_jtfn_dcp(training_params, training, dataset_idx):
        n_class = training_params['n_class']
        steps = training_params['steps']
        channels = [24, 12, 40, 120, 384]

        cha_promot_channels = training_params['cha_promot_channels']
        pos_promot_channels = training_params['pos_promot_channels']
        local_window_sizes = training_params['local_window_sizes']
        att_fusion = training_params['att_fusion']
        prompt_init = training_params['prompt_init']
        embed_ratio = training_params['embed_ratio']
        strides = training_params['strides']
        use_conv = training_params['use_conv']

        encoder = build_backbone('efficientnet_b3_p')
        decoder = JTFNDecoder(channels=channels, use_topo=True)
        seg_net = JTFN_DualAtt2_2(encoder=encoder, decoder=decoder, channels=channels, num_classes=n_class, steps=steps,
                                         dataset_idx=dataset_idx, local_window_sizes=local_window_sizes, 
                                         encoder_channels=channels, 
                                         cha_promot_channels=cha_promot_channels, pos_promot_channels=pos_promot_channels,
                                         embed_ratio=embed_ratio, strides=strides, 
                                         att_fusion=att_fusion, use_conv=use_conv)

        model = JTFNDualPromptProcessor(model=seg_net, training_params=training_params, training=training)
        return model
    
    