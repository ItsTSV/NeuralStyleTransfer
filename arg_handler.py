import argparse


def handle_args():
    """Reads command line arguments, parses and returns them. If no arguments are given, default values are used.

    Returns:
        args: argparse.Namespace, object that contains all the selected values
    """
    parser = argparse.ArgumentParser(description="Neural Style Transfer")

    # Input/Output arguments
    parser.add_argument(
        "--content_path",
        type=str,
        help="Path to the content image",
        default="images/content/fei.jpg",
    )
    parser.add_argument(
        "--style_path",
        type=str,
        help="Path to the style image",
        default="images/style/starrynight.jpg",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output image",
        default="results/transferred.jpg",
    )

    # Hyperparameters
    parser.add_argument(
        "--iterations",
        type=int,
        help="How many iterations will the NST algorithm perform",
        default=101,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Size of image the Neural Network will be working with",
        default=256,
    )
    parser.add_argument(
        "--content_weight",
        type=int,
        help="How much will the content part of the image be preferred",
        default=1e0,
    )
    parser.add_argument(
        "--style_weight",
        type=int,
        help="How much will the style part of the image be preferred",
        default=1e6,
    )
    parser.add_argument(
        "--force_resize",
        type=bool,
        help="Forces the style image to be resized to the content image size",
        default=False,
    )

    # Style mashup arguments
    parser.add_argument(
        "--mashup",
        type=bool,
        help="Whether to use style mashup or not",
        default=False,
    )
    parser.add_argument(
        "--mashup_path",
        type=str,
        help="Path to the second style image for mashup",
        default="images/style/starrynight.jpg",
    )
    parser.add_argument(
        "--w1",
        type=float,
        help="Weight for the first style image in mashup",
        default=0.5,
    )
    parser.add_argument(
        "--w2",
        type=float,
        help="Weight for the second style image in mashup",
        default=0.5,
    )

    # Parse, return
    args = parser.parse_args()
    return args


def print_args(args):
    """Prints all args and their values. Mainly for debugging."""
    for arg in vars(args):
        print(f"Argument {arg} is set to {getattr(args, arg)}")
