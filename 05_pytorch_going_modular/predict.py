"""
Predicts the class of an input image using a pretrained PyTorch model.
"""

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import model_builder

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a pretrained model.")
    
    parser.add_argument("--image", type=str, required=True, help="Path to the image file for prediction.")
    parser.add_argument("--model_path", type=str, default="/content/models/tinyvgg_model.pth", help="Path to the saved model file.")
    
    return parser.parse_args()

def load_model(model_path, device):
    model = model_builder.TinyVGG(input_shape=3, hidden_units=128, output_shape=3).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def predict_image(image_path, model, class_names, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.inference_mode():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
    
    return class_names[predicted_label], probabilities[0][predicted_label].item()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["pizza", "steak", "sushi"]
    
    model = load_model(args.model_path, device)
    
    pred_class, pred_prob = predict_image(args.image, model, class_names, device)
    
    print(f"[INFO] Predicted class: {pred_class}, Probability: {pred_prob:.3f}")

if __name__ == "__main__":
    main()
