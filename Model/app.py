"""
Gradio web app: upload one image, get cat/dog prediction and confidence.
Run: python app.py
Then open the URL shown (e.g. http://127.0.0.1:7860).
"""
import os

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from binary_classifier import BinaryImageClassifier


INPUT_SIZE = 224

# Change this path to your best_run.pth (or other .pth with BinaryImageClassifier weights)
WEIGHTS_PATH = "/Users/acaerme/Desktop/best_run.pth"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])


def load_model(weights_path: str, device):
    model = BinaryImageClassifier(input_channels=3, input_size=INPUT_SIZE).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict(image):
    """Run model on uploaded image. Returns string: 'class confidence' e.g. 'dog 0.87'."""
    if image is None:
        return "Upload an image to get a prediction."
    device = predict.device
    transform = predict.transform
    model = predict.model
    image_pil = Image.open(image).convert("RGB")
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = model(tensor).squeeze().item()
    class_name = "dog" if probability >= 0.5 else "cat"
    confidence = probability if class_name == "dog" else 1.0 - probability
    return f"{class_name} {confidence:.4f}"


def main():
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights file not found: {WEIGHTS_PATH}. "
            "Edit WEIGHTS_PATH at the top of app.py to point to your .pth file."
        )
    device = get_device()
    transform = get_transform()
    model = load_model(WEIGHTS_PATH, device)
    predict.device = device
    predict.transform = transform
    predict.model = model

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Upload image"),
        outputs=gr.Textbox(label="Prediction"),
        title="Cat / Dog classifier",
        description="Upload a single image. The model predicts cat or dog and confidence.",
    )
    interface.launch(share=True)


if __name__ == "__main__":
    main()
