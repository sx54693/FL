import torch
from torchvision import transforms
from PIL import Image
from model_def import PneumoniaCNN

# Set device (use CPU for simplicity)
DEVICE = torch.device("cpu")

# Load the trained model
model = PneumoniaCNN().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# Image transform: Resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Replace this with your image path when running
import sys
if len(sys.argv) < 2:
    print("❌ Please provide the path to an image file.")
    exit()

image_path = sys.argv[1]

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# Perform prediction
with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).item()
    label = "PNEUMONIA" if prediction == 1 else "NORMAL"
    print(f"🩺 Prediction: {label}")

