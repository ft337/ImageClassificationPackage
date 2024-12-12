import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model(model_path, num_classes, device):
    # Recreate the model architecture
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load the saved state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move the model to the specified device
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

# Predict and display for a single image
def predict_single_image(model, image_path, classes, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    prediction = classes[predicted.item()]

    # Display the image and prediction
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Prediction: {prediction}")
    plt.show()

# Main function for CLI
def main():
    parser = argparse.ArgumentParser(description="Single Image Prediction")
    parser.add_argument("--model_path", type=str, default="models/model.pth", help="Path to the saved model (default: model.pth)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the image file for prediction")

    args = parser.parse_args()

    # Define classes
    classes = ["anode", "cathode", "nothing"]

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    num_classes = len(classes)
    model = load_model(args.model_path, num_classes, device)

    if os.path.isfile(args.input_path):
        # Single image
        predict_single_image(model, args.input_path, classes, device)
    else:
        raise ValueError("Invalid input path. Must be an image file.")

if __name__ == "__main__":
    main()
