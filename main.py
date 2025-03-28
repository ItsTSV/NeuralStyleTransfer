import torch
from image_processing import load_preprocess_image, assert_dimensions, extract_image
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

# Load and preprocess images
content_tensor = load_preprocess_image(args.content_path, args.image_size, device)
style_tensor = load_preprocess_image(args.style_path, args.image_size, device)
assert_dimensions(content_tensor, style_tensor)

# Load the VGG19 Feature Extractor
feature_extractor = Vgg19FeatureExtractor(device)
content_features, style_features = feature_extractor(content_tensor)

# Run nst
nst = NeuralStyleTransfer(content_tensor, style_tensor, device, args)
generated_img = nst.train()

# Extract image, save and show it
output = extract_image(generated_img)
output.save(args.output_path)
output.show()
