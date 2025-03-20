import torch.nn as nn
import torchvision.models as models


class Vgg19FeatureExtractor(nn.Module):
    """Class which extracts features from the VGG19 neural network.

    There will be a lot of magically appearing values in the code; I will try to explain them in the docs. They
    are all based on the original Neural Style Transfer paper by Gatys et al. (2015).
    https://arxiv.org/abs/1508.06576

    Attributes:
        device: torch.device, Device on which the neural network will be loaded.
        content_layer_index: int, index of the layer from which the content features will be extracted (conv4_2).
        style_layer_indexes: list of ints, indexes of the layers from which the style features will be extracted (conv1_1,
            conv2_1, conv3_1, conv4_1, conv5_1).
        model: torch.nn.Module, VGG19 neural network up to the last style layer.
    """

    def __init__(self, device):
        """Initialize the VGG19FeatureExtractor.

        First, the VGG19 neural network with weights is loaded. Then, the layers that are used for NST are
        defined (based on the paper). The model only takes convolutional parts of the vgg19, up to the last style
        layer (conv5_1). The model is then sent to device and the parameters are frozen, so the network will not be
        trained.

        Args:
            device: torch.device, Device on which the neural network will be loaded.
        """

        super().__init__()
        self.device = device

        # Load the VGG19 neural network -- keep only convolutional part
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

        # Define the layer indexes, from which the features will be extracted
        self.content_layer_index = 21
        self.style_layer_indexes = [0, 5, 10, 19, 28]

        # Define model (vgg network up to the last style layer), transfer it to device
        self.model = vgg19[: max(self.style_layer_indexes) + 1].to(self.device)
        self.model.requires_grad_(False)

    def forward(self, x):
        """Forward pass through the modified VGG19 neural network.

        The forward pass goes through the modified NN and only saves the features of content layer and style layers.

        Args:
            x: torch.tensor, tensor that represents the input image; provided by the ImageProcessing class.

        Returns:
            content_features: dict, features of the content image, which will be used to calculate the content loss.
            style_features: dict, features of the style image, which will be used to calculate the style loss.
        """
        content_features = {}
        style_features = {}

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == self.content_layer_index:
                content_features["conv4_2"] = x
            if i in self.style_layer_indexes:
                style_features[f"conv{self.style_layer_indexes.index(i) + 1}_1"] = x

        return content_features, style_features
