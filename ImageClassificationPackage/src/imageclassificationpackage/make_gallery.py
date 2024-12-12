import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.patches as patches
import warnings

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

# Predict and display images in a grid
def predict_and_display_grid(image_dirs, model, classes, device, known_labels):
    images = []
    for dir_path in image_dirs:
        label = os.path.basename(dir_path) if known_labels else None
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append((Image.open(img_path).convert("RGB"), label))

    # Check if images exist
    if not images:
        raise ValueError(f"No valid images found in the specified directories: {image_dirs}")

    # Set up the grid
    num_images = len(images)
    grid_cols = 4
    grid_rows = (num_images + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 4))
    axes = axes.flatten()

    # Predict and display each image
    for i, (image, true_label) in enumerate(images):
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = classes[predicted.item()]

        # Resize the image to a consistent size for display
        image_resized = image.resize((224, 224))

        ax = axes[i]
        ax.imshow(image_resized)

        if known_labels:
            ax.set_title(f'True: {true_label}\nPred: {predicted_class}', fontsize=10)
            # Add an outline based on prediction correctness
            color = 'green' if true_label == predicted_class else 'red'
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=color, linewidth=12, fill=False)
            ax.add_patch(rect)
        else:
            ax.set_title(f'Pred: {predicted_class}', fontsize=10)

        ax.axis('off')

    # Remove unused subplots
    for j in range(len(images), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Main function for CLI
def main():
    parser = argparse.ArgumentParser(description="Generate Prediction Gallery")
    parser.add_argument("--model_path", type=str, default="models/model.pth", help="Path to the saved model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory of images")
    parser.add_argument("--known_labels", action="store_true", help="Specify if the labels are known (required for subdirectories)")
    args = parser.parse_args()

    # Define classes
    classes = ["nothing", "cathodes", "anodes"]

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    num_classes = len(classes)
    model = load_model(args.model_path, num_classes, device)

    # Gather subdirectories
    subdirs = [os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]
    if not subdirs:
        raise ValueError(f"No subdirectories found in the directory: {args.input_path}")

    # Call the grid display function
    predict_and_display_grid(subdirs, model, classes, device, args.known_labels)

if __name__ == "__main__":
    main()
