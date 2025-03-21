import torch
import torch.nn.functional as F


def compute_content_loss(content_tensor, generated_tensor):
    """Computes MSE loss between tensors from content and generated layers.

       Args:
           content_tensor: torch.tensor, output of vgg19 content layer for content image.
           generated_tensor: torch.tensor, output of vgg19 content layer for image that is being generated.
    """
    content_loss = F.mse_loss(content_tensor, generated_tensor)
    return content_loss


def compute_style_loss(style_features, generated_features):
    """Computes MSE loss between gram matrices of style features and generated features.

       Args:
           style_features: dict{layer, tensor}, output of vgg19 style layers for style image.
           generated_features: dict{layer, tensor}, output of vgg19 style layers for image that is being generated.
    """
    style_loss = 0
    for layer in style_features:
        target_gram = _gram_matrix(style_features[layer])
        generated_gram = _gram_matrix(generated_features[layer])
        style_loss += F.mse_loss(target_gram, generated_gram)
    return style_loss


def _gram_matrix(tensor):
    """Compute gram matrix G = A * A^T, which represents the style texture, for given tensor"""
    a, b, c, d = tensor.size()
    features = tensor.view(a * b, c * d)
    gram = torch.mm(features, features.t())
    return gram.div(a * b * c * d)
