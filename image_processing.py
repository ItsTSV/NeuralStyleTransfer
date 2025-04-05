import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def load_preprocess_images(content_path, style_path, image_size, device, force_resize):
    """Load and preprocess images, so they can be fed to the neural network.

    The preprocessing pipeline contains these steps:
        1. Images are resized to a given size
        2. Images are converted to a tensor, their values are changed to <0, 1> range
        3. Another dimension is added to the tensors, so their shape is accepted by the neural network
        4. Tensors are sent to device (cuda gpu)
    """
    # Load images
    content_image = Image.open(content_path)
    style_image = Image.open(style_path)

    # If selected, force resize the style image to content image size, possibly deforming it
    if force_resize:
        print("The style image was forcefully resized -- a deformation may occur!")
        style_image = style_image.resize(content_image.size)

    # Prepare the pipeline, run image through it, convert it to tensor
    pipeline = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    # Send images to a pipeline, convert them to tensors
    content_tensor = pipeline(content_image).unsqueeze(0).to(device)
    style_tensor = pipeline(style_image).unsqueeze(0).to(device)
    return content_tensor, style_tensor


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
    """
    image = image_tensor.clone().detach().cpu().squeeze(0)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).clip(0, 255).astype("uint8")

    image = Image.fromarray(image)
    image = image.resize(size)
    return image


def preview_image(tensor):
    """Preview image from tensor."""
    image = extract_image(tensor)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
