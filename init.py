import torch
from image_processing import ImageProcessing
from vgg19_feature_extractor import Vgg19FeatureExtractor
from neural_style_transfer import NeuralStyleTransfer
from arg_handler import handle_args, print_args

# Check if CUDA device is present, if not, send a warning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print(
        "Warning: CUDA GPU is not present! The computations will be performed on CPU."
    )

# Handle the arguments
args = handle_args()
print_args(args)

# Load and preprocess images
image_processing = ImageProcessing(args.content, args.style, device, args.img_size)
content_img = image_processing.content_tensor
style_img = image_processing.style_tensor

# Load the VGG19 Feature Extractor
feature_extractor = Vgg19FeatureExtractor(device)
content_features, style_features = feature_extractor(content_img)

# Run nst
nst = NeuralStyleTransfer(content_img, style_img, device)
generated_img = nst.train()
output = image_processing.extract_image(generated_img)
output.show()
