import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from binary_classifier import BinaryImageClassifier


INPUT_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

'''
You have 2 options to run the model:

1) Run the command: python run_model.py <weights_path> <input_path>

where <weights_path> is the path to the .pth file with weights and <input_path> is the path to the image or folder with images

2) Insert the paths in the WEIGHTS_PATH and INPUT_PATH variables below and run the command: python run_model.py
'''
WEIGHTS_PATH = "placeholder" # path to the weights file
INPUT_PATH = "placeholder" # path to the image or folder with images

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])


def collect_image_paths(path: str) -> list[str]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    if os.path.isfile(path):
        suffix = os.path.splitext(path)[1].lower()
        if suffix not in IMAGE_EXTENSIONS:
            print(f"Error: not a supported image file: {path}", file=sys.stderr)
            sys.exit(1)
        return [path]

    if os.path.isdir(path):
        paths = []
        for filename in os.listdir(path):
            suffix = os.path.splitext(filename)[1].lower()
            if suffix in IMAGE_EXTENSIONS:
                paths.append(os.path.join(path, filename))
        paths.sort()
        if not paths:
            print(f"Error: no image files found in directory: {path}", file=sys.stderr)
            sys.exit(1)
        return paths

    print(f"Error: not a file or directory: {path}", file=sys.stderr)
    sys.exit(1)


def load_and_predict(model, transform, device, image_path: str) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = model(tensor).squeeze().item()
    class_name = "dog" if probability >= 0.5 else "cat"
    confidence = probability if class_name == "dog" else 1.0 - probability
    return class_name, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Run trained cat/dog classifier on image(s)."
    )
    parser.add_argument(
        "weights_path",
        nargs="?",
        default=WEIGHTS_PATH,
        help="Path to best_run.pth (or other .pth with BinaryImageClassifier state_dict)",
    )
    parser.add_argument(
        "image_or_folder",
        nargs="?",
        default=INPUT_PATH,
        help="Path to a single image file or a directory of images",
    )
    arguments = parser.parse_args()

    weights_path = os.path.abspath(arguments.weights_path)
    if not os.path.isfile(weights_path):
        print(f"Error: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    transform = get_transform()

    model = BinaryImageClassifier(input_channels=3, input_size=INPUT_SIZE).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    image_paths = collect_image_paths(arguments.image_or_folder)

    for image_path in image_paths:
        class_name, confidence = load_and_predict(model, transform, device, image_path)
        print(f"{image_path}\t{class_name}\t{confidence:.4f}")


if __name__ == "__main__":
    main()
