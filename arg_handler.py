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
        default="images/fei.jpg",
    )
    parser.add_argument(
        "--style_path",
        type=str,
        help="Path to the style image",
        default="images/starrynight.jpg",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output image",
        default="results/transfered.jpg",
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

    # Parse, return
    args = parser.parse_args()
    return args


def print_args(args):
    """Prints all args and their values. Mainly for debugging."""
    for arg in vars(args):
        print(f"Argument {arg} is set to {getattr(args, arg)}")
