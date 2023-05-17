import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

class Img2Vec():
    """ Class to extract vectors from images using a pretrained CNN
    """
    def __init__(self, model, cuda=True, layer='default', layer_output_size=512, input_size=224):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = model.__class__.__name__

        self.extraction_layer = self._get_layer(layer)


    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if isinstance(img, list) or (isinstance(img, torch.Tensor) and img.dim() == 4):
            if isinstance(img, list):
                images = torch.stack(img).to(self.device)
            else:
                images = img.to(self.device)
                
            if self.model_name in ['AlexNet', 'VGG']:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == 'DenseNet' or 'EfficientNet' in self.model_name:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['AlexNet', 'VGG']:
                    return my_embedding.numpy()[:, :]
                elif self.model_name == 'DenseNet' or 'EfficientNet' in self.model_name:
                    return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = img.to(self.device)

            if self.model_name in ['AlexNet', 'VGG']:
                my_embedding = torch.zeros(1, self.layer_output_size)
            elif self.model_name == 'DenseNet' or 'EfficientNet' in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['AlexNet', 'VGG']:
                    return my_embedding.numpy()[0, :]
                elif self.model_name == 'DenseNet':
                    return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]

    def _get_layer(self, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if self.model_name == 'ResNet':
            if layer == 'default':
                layer = self.model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = self.model._modules.get(layer)
            return layer

        elif self.model_name == 'AlexNet':
            if layer == 'default':
                layer = self.model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]
                
            return layer

        elif self.model_name == 'VGG':
            # VGG-11
            if layer == 'default':
                layer = self.model.classifier[-2]
                self.layer_output_size = self.model.classifier[-1].in_features # should be 4096
            else:
                layer = self.model.classifier[-layer]

            return layer

        elif self.model_name == 'DenseNet':
            # Densenet-121
            if layer == 'default':
                layer = self.model.features[-1]
                self.layer_output_size = self.model.classifier.in_features # should be 1024
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return layer

        elif self.model_name == 'EfficientNet':
            if layer == 'default':
                layer = self.model.features
                self.layer_output_size = 1792
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return layer

        else:
            raise KeyError('Model %s was not found' % model_name)
