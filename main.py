import torch
from image_processing import load_preprocess_images, assert_dimensions, extract_image
from mashup_style_transfer import MashupStyleTransfer
from neural_style_transfer import NeuralStyleTransfer
from arg_handler import handle_args

# Check if CUDA device is present, if not, send a warning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print(
        "Warning: CUDA GPU is not present! The computations will be performed on CPU."
    )

# Handle the arguments
args = handle_args()

# Load and preprocess images
content_tensor, style_tensor = load_preprocess_images(
    args.content_path, args.style_path, args.image_size, device, args.force_resize
)
assert_dimensions(content_tensor, style_tensor)

# Check if the user wants to use one to one style transfer or mashup style transfer
if args.mashup:
    _, mashup_tensor = load_preprocess_images(
        args.content_path, args.mashup_path, args.image_size, device, args.force_resize
    )
    assert_dimensions(content_tensor, mashup_tensor)
    nst = MashupStyleTransfer(
        content_tensor, style_tensor, mashup_tensor, args.w1, args.w2, device, args
    )
else:
    nst = NeuralStyleTransfer(content_tensor, style_tensor, device, args)

# Run, optimize and save the image
generated_img = nst.train()
output = extract_image(generated_img)
output.save(args.output_path)
output.show()
