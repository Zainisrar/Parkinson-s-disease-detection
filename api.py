from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Define the label mapping (must match the training setup)
stage_labels = {0: "Stage 1", 1: "Stage 2", 2: "Stage 3", 3: "Stage 4"}

# Define the model structure (must match the training setup)
def load_model(model_path):
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 4)  # 4 classes
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
    return model

# Load the model
model_path = "best_model.pth"
model = load_model(model_path)

# Define the preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(file):
    # Read image file into memory and preprocess
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def predict_stage(model, file):
    img = preprocess_image(file)
    with torch.no_grad():  # No gradient computation needed for inference
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)  # Get the class index with highest probability
        stage = stage_labels[predicted.item()]  # Map index to stage label
    return stage

# Define the prediction route
@app.route('/', methods=['POST'])
def predict():
    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read file directly in memory
    try:
        stage = predict_stage(model, file.read())  # Pass file as bytes
        return jsonify({'stage': stage})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
