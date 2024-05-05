from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io
from .model import LivenessNet

app = FastAPI(title="Liveness Detection API")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LivenessNet(num_classes=2).to(device)
model.eval()

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.post("/detect_liveness")
async def detect_liveness(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = transform_pipeline(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(tensor)
        prob = torch.softmax(preds, dim=1)
        score = prob[0][1].item() # Assuming index 1 is "Real/Live"
        
    return {"liveness_score": score, "status": "Live" if score > 0.5 else "Spoof"}
