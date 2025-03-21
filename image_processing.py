import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def load_preprocess_image(path, image_size, device):
    """Load and preprocess image, so it can be fed to the neural network.

    The preprocessing pipeline contains these steps:
        1. Image is resized to a given size
        2. Image is converted to a tensor, its values are changed to <0, 1> range
        3. Another dimension is added to the tensor, so its shape is accepted by the neural network
        4. Tensor is sent to device (cuda gpu)

    Args:
        path: string, path to the image in a standard format (.jpg, .png...)
        image_size: int, size to which the image will be converted
    """
    image = Image.open(path)

    # Prepare the pipeline, run image through it, convert it to tensor
    pipeline = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    image_tensor = pipeline(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor


def assert_dimensions(content_tensor, style_tensor):
    """Assert the shapes of tensors are the same, so the algorithm can run without problems."""
    assert content_tensor.shape == style_tensor.shape, "The dimensions are not correct!"


def extract_image(image_tensor, size=(1920, 1080)):
    """Converts image tensor back into an image.

    The deprocessing pipeline contains these steps:
        1. The tensor is cloned and detached from computation graph
        2. The tensor is sent back to CPU
        3. The tensor is squeezed, so it has 3 dimensions
        4. It is converted to Numpy array, channels are shuffled, so it is in correct format
        5. Its values are converted to range <0, 255>
        6. Finally, an image is created and resized to a given size

     Args:
         image_tensor: torch.tensor, output of the neural network
         size: (int, int), dimension to which the image will be resized
    """
    image = image_tensor.clone().detach().cpu().squeeze(0)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype("uint8")

    image = Image.fromarray(image)
    image = image.resize(size)
    return image


def preview_image(tensor):
    """Preview image from tensor.

    Args:
        tensor: torch.tensor, output of the neural network.
    """
    image = extract_image(tensor)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
