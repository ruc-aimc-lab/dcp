# Example code for running inference on a pre-trained model
import os
import json
import numpy as np
import cv2
import torch
from models import build_model


# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def sigmoid(arr):
    return 1. / (1 + np.exp(-arr))

class Inference(object):
    def __init__(self, model_path):
        self.model_path = model_path
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path) as fin:
            params = json.load(fin)
        self.model_params = params['model_params']
        self.modality_mapping = params['modality_mapping']
        self.model = self.load_model()
        
    def inference(self, image, modality):
        assert modality in self.modality_mapping, "Modality '{}' not supported".format(modality)
        
        image = self.load_image(image)
        modality_idx = self.modality_mapping[modality]
        modality_idx = torch.tensor([modality_idx])
        with torch.no_grad():
            output = self.model.predict(x=image, device=device, dataset_idx=modality_idx)
        output = output.data.cpu().numpy()[0][0]
        output = sigmoid(output) * 255
        output = output.astype(np.uint8)
        return output
        
    def load_image(self, image):
        # Load the image and preprocess it
        image = cv2.imread(image)[:, :, [2, 1, 0]]
        image = cv2.resize(image, (self.model_params['size_w'], self.model_params['size_h']))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image)
        return image
    
    def load_model(self):
        print('Loading model from {}'.format(self.model_path))
        model = build_model(model_name=self.model_params['net'], 
                            model_params=self.model_params, 
                            training=False, 
                            dataset_idx=list(self.modality_mapping.values()),
                            pretrained=False)
        model.set_device(device)
        # model.requires_grad_false()
        model.load_model(os.path.join(self.model_path, 'model.pkl'))
        model.set_mode('eval')
        
        return model


if __name__ == '__main__':
    model_path = 'checkpoints/R2AUNet_DCP_512'
    image_path = 'images/CFP.jpg'
    modality = 'CFP'  
    output_path = 'output_image_CFP.png'

    inference = Inference(model_path)
    output = inference.inference(image_path, modality)
    
    # Save the output
    cv2.imwrite(output_path, output)
    print(f"Output saved to {output_path}")