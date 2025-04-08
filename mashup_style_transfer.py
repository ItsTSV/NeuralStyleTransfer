from neural_style_transfer import NeuralStyleTransfer
from loss_functions import compute_mashup_style_loss, compute_content_loss


class MashupStyleTransfer(NeuralStyleTransfer):
    """Class for performing Style Transfer with two images using VGG19.

    Additional Args:
          mashup_tensor: torch.Tensor, the image that will be used as a mashup.
          w1: float, weight for the first style image.
          w2: float, weight for the second style image.

    See Also:
          NeuralStyleTransfer: The parent class that performs the basic style transfer.
    """

    def __init__(self, content_tensor, style_tensor, mashup_tensor, w1, w2, device, args):
        """Initialize the Mashup Style Transfer model."""
        super().__init__(content_tensor, style_tensor, device, args)

        # Mashup parameters
        _, self.mashup_features = self.extractor(mashup_tensor)
        self.w1 = w1
        self.w2 = w2

    def compute_losses(self):
        """Computes content and mashup style loss for the generated image."""
        gen_content_features, gen_style_features = self.extractor(self.generated_image)

        content_loss = compute_content_loss(
            self.content_features["conv4_2"], gen_content_features["conv4_2"]
        )
        style_loss = compute_mashup_style_loss(
            self.style_features, self.mashup_features, gen_style_features,
            self.w1, self.w2,
        )

        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss
