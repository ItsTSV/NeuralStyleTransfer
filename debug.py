import torch
import matplotlib.pyplot as plt


def visualize_feature_maps(feature_tensor, max_maps=16, title='Feature maps'):
    """Visualizes feature maps from a given tensor.

        Debugging function only -- this is not used in the main algorithm! The only purpose of this function
        is to visualize features, so I can put the image in the paper.

        Args:
            feature_tensor: torch.Tensor, the tensor containing feature maps to visualize.
            max_maps: int, the maximum number of feature maps to visualize.
            title: str, the title of the plot.
    """
    # Remove batch dimension
    feature_tensor = feature_tensor.squeeze(0)

    # Pick the number of channels to visualize
    num_channels = min(max_maps, feature_tensor.shape[0])

    # Create a grid for the subplots
    cols = int(num_channels ** 0.5)
    rows = (num_channels + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        fmap = feature_tensor[i].detach().cpu().numpy()

        # Normalize for better contrast
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)

        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i}')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
