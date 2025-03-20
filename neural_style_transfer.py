import torch
import torch.optim as optim
from vgg19_feature_extractor import Vgg19FeatureExtractor
from loss_functions import compute_style_loss, compute_content_loss


class NeuralStyleTransfer:
    """Class for performing Neural Style Transfer using VGG19."""

    def __init__(
        self,
        content_tensor,
        style_tensor,
        device,
        content_weight=1,
        style_weight=1000000,
        num_steps=101,
    ):
        """Initialize the NST model.

        Args:
            content_tensor: torch.tensor, preprocessed content image.
            style_tensor: torch.tensor, preprocessed style image.
            device: torch.device, computation device.
            content_weight: float, weight for content loss.
            style_weight: float, weight for style loss.
            num_steps: int, number of optimization steps.
            lr: float, learning rate for the optimizer.
        """
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.num_steps = num_steps

        self.extractor = Vgg19FeatureExtractor(device)

        # Extract features for content and style images
        self.content_features, _ = self.extractor(content_tensor)
        _, self.style_features = self.extractor(style_tensor)

        # Initialize generated image as a copy of content image
        self.generated_image = content_tensor.clone().requires_grad_(True)
        self.optimizer = optim.LBFGS([self.generated_image])

    def compute_losses(self):
        """Compute content and style loss for the generated image."""
        gen_content_features, gen_style_features = self.extractor(self.generated_image)

        content_loss = compute_content_loss(
            self.content_features["conv4_2"], gen_content_features["conv4_2"]
        )

        style_loss = compute_style_loss(self.style_features, gen_style_features)

        # Total loss
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss

    def train(self):
        """Optimize the generated image to match the content and style targets."""

        def closure():
            self.optimizer.zero_grad()
            loss = self.compute_losses()
            loss.backward()
            return loss

        for step in range(self.num_steps):
            loss = self.optimizer.step(closure)

            if step % 50 == 0:
                print(f"Step {step}/{self.num_steps}, Loss: {loss.item():.4f}")

        return self.generated_image
