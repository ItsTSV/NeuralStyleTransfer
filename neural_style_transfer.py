import torch
from torch import optim
from vgg19_feature_extractor import Vgg19FeatureExtractor
from loss_functions import compute_style_loss, compute_content_loss


class NeuralStyleTransfer:
    """Class for performing Neural Style Transfer using VGG19.

    Attributes:
        device: torch.device, device on which the algorithm will work.
        content_weight: int, how much the content loss will be weighted.
        style_weight: int, how much the style loss will be weighted.
        num_steps: int, the number of iterations NST algorithm performs.
        extractor: Vgg19FeatureExtractor, will be used to get features from the vgg19 model.
        content_features: dict, will be used to save the content features given by the extractor.
        style_features: dict, will be used to save the style features given by the extractor.
        generated_image: torch.tensor, the image that will be optimized throughout the algorithm.
        optimizer: torch.optim, the NST algorithm uses LBFGS.
    """

    def __init__(self, content_tensor, style_tensor, device, args):
        """Initialize the NST model.

        Args:
            args: argparse.Namespace, object that stores all the user settings
        """
        self.device = device
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight
        self.num_steps = args.iterations

        self.extractor = Vgg19FeatureExtractor(device)
        self.extractor.eval()

        # Extract features for content and style images
        self.content_features, _ = self.extractor(content_tensor)
        _, self.style_features = self.extractor(style_tensor)

        # Initialize generated image as a copy of content image
        self.generated_image = content_tensor.clone()
        self.generated_image.requires_grad_(True)
        self.optimizer = optim.LBFGS([self.generated_image])

    def compute_losses(self):
        """Compute content and style loss for the generated image.

        First, the generated image is fed into the customized Vgg19 model. From that, the features are gained.
        These features are then used with features of content and style images to compute the losses. Those
        computed losses are then weighted, so the algorithm output can be customized.
        """
        gen_content_features, gen_style_features = self.extractor(self.generated_image)

        content_loss = compute_content_loss(
            self.content_features["conv4_2"], gen_content_features["conv4_2"]
        )

        style_loss = compute_style_loss(self.style_features, gen_style_features)

        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss

    def train(self):
        """Optimize the generated image to match the content and style targets.

        For given number of iterations, the algorithm performs the NST optimization. In each iteration,
        the generated image is clamped and its gradients are voided. Then, the loss is calculated and propagated
        backwards. The optimizer step needs to be in a closure -- LBFGS algorithm requires it, because it performs
        it multiple times in a single iteration.

        Returns:
            generated_image: torch.tensor, the transferred image, which needs to be deprocessed to view normally.
        """
        for step in range(self.num_steps):

            def closure():
                with torch.no_grad():
                    self.generated_image.clamp_(0, 1)

                self.optimizer.zero_grad()
                loss = self.compute_losses()
                loss.backward()

                if step % 5 == 0:
                    print(f"Step {step}/{self.num_steps}, Loss: {loss.item():.4f}")

                return loss

            self.optimizer.step(closure)

        with torch.no_grad():
            self.generated_image.clamp_(0, 1)

        return self.generated_image
