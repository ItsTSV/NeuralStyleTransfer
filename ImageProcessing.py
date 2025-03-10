import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


class ImageProcessing:
    """Class which handles image loading, preprocessing, saving and previewing.

    Attributes:
        content_path: string, path to an image whose style will be changed.
        style_path: string, path to an image that will provide the style.
        device: torch.device, Device to which the tensors will be sent.
        image_size: int, image size that will be used in the neural network -- the images will be resized to this size.
    """

    def __init__(self, content_path, style_path, device, image_size=512):
        self.image_size = image_size
        self.device = device

        # Set pipeline -- resize both images to same size (w/h must be the same, w and h not!), convert to tensor
        self.pipeline = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.content_tensor = self.load_preprocess_image(content_path)
        self.style_tensor = self.load_preprocess_image(style_path)

    def load_preprocess_image(self, path):
        """Load and preprocess image, so it can be fed to the neural network.

        The preprocessing pipeline consists of multiple steps: loading -> resizing to given size -> converting to tensor
        (changing data type and value range from 0-255 to 0-1) -> adding additional dimension, as required by the VGG19
        neural network -> sending the tensor to selected device.

        Args:
            path: string, path to image file.

        Returns:
            image: tensor, which can be fed to the neural network.
        """
        image = Image.open(path)
        image = self.pipeline(image)
        image = image.unsqueeze(0).to(self.device)
        return image

    def extract_image(self, tensor, size=(1920, 1080)):
        """Extract image from tensor and convert it to PIL Image, which can then be saved.

        The deprocessing pipeline consist of multiple steps: cloning and detaching the tensor -> removing the additional
        dimension, which was required by the VGG19 neural network -> rearranging the dimensions of tensor (channels,
        height, width) to (height, width, channels) -> changing the data type and value frange from 0-1 to 0-255 ->
        building the image from array, resizing it to given size.

        Args:
            tensor: torch.tensor, output of the neural network.
            size: tuple, size of the output image.

        Returns:
            image: PIL Image.
        """
        image = tensor.clone().detach().cpu().squeeze(0)

        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).clip(0, 255).astype("uint8")

        image = Image.fromarray(image)
        image = image.resize(size)
        print(type(image))
        return image

    def preview_image(self, tensor):
        """Preview image from tensor.

        Args:
            tensor: torch.tensor, output of the neural network.
        """
        image = self.extract_image(tensor)
        plt.imshow(image)
        plt.axis("off")
        plt.show()

