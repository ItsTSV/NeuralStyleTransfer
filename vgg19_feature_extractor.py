import torch
from torch import nn
from torchvision import models


class Vgg19FeatureExtractor(nn.Module):
    """Class which extracts features from the VGG19 neural network.

    The extraction is based on a paper by Gatys, Ecker and Bethge (https://arxiv.org/abs/1508.06576). There might
    be some minor differences, but most of the values are taken from the paper.

    Attributes:
        device: torch.device, Device on which the neural network will be loaded.
        content_layer_index: int, index of the layer from which the content features will be extracted (conv4_2).
        style_layer_indexes: list of ints, indexes of the layers from which the style features will
                             be extracted (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1).
        mean: torch.tensor, mean values for normalization in forward pass.
        std: torch.tensor, std values for normalization in forward pass.
        model: torch.nn.Module, VGG19 neural network up to the last style layer.
    """

    def __init__(self, device):
        """Initialize the VGG19FeatureExtractor.

        Loads the pretrained vgg19 model, sets the mean and std values for normalization, defines the content
        and style layer and saves them into a model.
        """
        super().__init__()
        self.device = device

        # Set mean and std normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)

        # Load the VGG19 neural network -- keep only convolutional part, use eval mode
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

        # Define the layer indexes, from which the features will be extracted -- based on the paper
        self.content_layer_index = 21
        self.style_layer_indexes = [0, 5, 10, 19, 28]

        # Define model (vgg network up to the last style layer), transfer it to device
        self.model = vgg19[: max(self.style_layer_indexes) + 1].to(self.device)
        self.model.requires_grad_(False)

    def forward(self, x):
        """Forward pass through the modified VGG19 neural network.

        First, the input tensor is normalized. Then, the tensor is passed through the network, but only the
        wanted content and style features are saved.
        """
        x = (x - self.mean) / self.std

        content_features = {}
        style_features = {}

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == self.content_layer_index:
                content_features["conv4_2"] = x
            if i in self.style_layer_indexes:
                style_features[f"conv{self.style_layer_indexes.index(i) + 1}_1"] = x

        return content_features, style_features
