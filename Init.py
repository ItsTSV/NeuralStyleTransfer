import torch
import argparse
from ImageProcessing import ImageProcessing
from Vgg19FeatureExtractor import Vgg19FeatureExtractor

# Check if CUDA device is present, if not, send a warning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
if device == "cpu":
    print("Warning: CUDA GPU is not present! The computations will be performed on CPU.")

# Parse user input
parser = argparse.ArgumentParser(description="Neural Style Transfer")
parser.add_argument("--content", type=str, help="Path to the content image", default="images/fei.jpg")
parser.add_argument("--style", type=str, help="Path to the style image", default="images/starrynight.jpg")
parser.add_argument("--output", type=str, help="Path to the output image", default="output.jpg")
parser.add_argument("--iterations", type=int, help="Number of iterations", default=500)
parser.add_argument("--img_size", type=int, help="Size of image the NN will be working with", default=512)
args = parser.parse_args()

# Load and preprocess images
image_processing = ImageProcessing(args.content, args.style, device, args.img_size)
content_img = image_processing.content_tensor
style_img = image_processing.style_tensor

# Load the VGG19 Feature Extractor
feature_extractor = Vgg19FeatureExtractor(device)
content_features, style_features = feature_extractor(content_img)

