import os

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from binary_classifier import BinaryImageClassifier, BinaryImageClassifierV2


INPUT_SIZE = 224

WEIGHTS_PATH_V1 = "placeholder" # path to the .pth file with weights for the original architecture
WEIGHTS_PATH_V2 = "placeholder" # path to the .pth file with weights for the updated architecture

MODEL_CHOICES = ["v1 (original)", "v2 (updated)"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])


def load_model_v1(weights_path: str, device):
    model = BinaryImageClassifier(input_channels=3, input_size=INPUT_SIZE).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_model_v2(weights_path: str, device):
    model = BinaryImageClassifierV2(input_channels=3, input_size=INPUT_SIZE).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict(model_choice, image):
    if image is None:
        return "Upload an image to get a prediction."
    device = predict.device
    transform = predict.transform
    models = predict.models
    model = models[model_choice]
    image_pil = Image.open(image).convert("RGB")
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = model(tensor).squeeze().item()
    class_name = "dog" if probability >= 0.5 else "cat"
    confidence = probability if class_name == "dog" else 1.0 - probability
    return f"{class_name} {confidence:.4f}"


def main():
    if not os.path.isfile(WEIGHTS_PATH_V1):
        raise FileNotFoundError(
            f"v1 weights file not found: {WEIGHTS_PATH_V1}. "
            "Edit WEIGHTS_PATH_V1 at the top of app.py."
        )
    if not os.path.isfile(WEIGHTS_PATH_V2):
        raise FileNotFoundError(
            f"v2 weights file not found: {WEIGHTS_PATH_V2}. "
            "Edit WEIGHTS_PATH_V2 at the top of app.py."
        )
    device = get_device()
    transform = get_transform()
    model_v1 = load_model_v1(WEIGHTS_PATH_V1, device)
    model_v2 = load_model_v2(WEIGHTS_PATH_V2, device)
    predict.device = device
    predict.transform = transform
    predict.models = {choice: model for choice, model in zip(MODEL_CHOICES, [model_v1, model_v2])}

    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model"),
            gr.Image(type="filepath", label="Upload image"),
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="Cat / Dog classifier",
        description="Choose a model (v1 or v2), then upload an image. The model predicts cat or dog and confidence.",
    )
    interface.launch()


if __name__ == "__main__":
    main()
