import torch
import torch.nn.functional as F


def compute_content_loss(content_tensor, generated_tensor):
    content_loss = F.mse_loss(content_tensor, generated_tensor)
    return content_loss


def compute_style_loss(style_features, generated_features):
    style_loss = 0
    for layer in style_features:
        target_gram = _gram_matrix(style_features[layer])
        generated_gram = _gram_matrix(generated_features[layer])
        style_loss += F.mse_loss(target_gram, generated_gram)
    return style_loss


def _gram_matrix(tensor):
    a, b, c, d = tensor.size()
    features = tensor.view(a * b, c * d)
    gram = torch.mm(features, features.t())
    return gram.div(a * b * c * d)

