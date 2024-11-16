from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import wandb
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import logging
from typing import Dict, Any
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CNN Inference Service",
    description="API for image classification using CNN model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
INFERENCE_TIME = Histogram(
    'inference_time_seconds', 
    'Time spent processing inference',
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total number of inference requests')
INFERENCE_ERRORS = Counter('inference_errors_total', 'Total number of inference errors')
MODEL_LOADING_TIME = Histogram('model_loading_time_seconds', 'Time spent loading model')

class CustomCNN(nn.Module):
    """
    Custom CNN architecture for image classification
    """
    def __init__(self, num_classes: int):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Force CPU usage
device = torch.device("cpu")
model = None

def load_model():
    """Load the model and initialize W&B"""
    global model
    start_time = time.time()
    
    try:
        logger.info("Loading model on CPU...")
        model = CustomCNN(num_classes=50)
        
        # Check if model file exists
        model_path = 'model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load model with explicit map_location
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        # Initialize W&B only if API key is available
        if os.getenv('WANDB_API_KEY'):
            wandb.init(
                project="yeahiasarkernabil-humufy",
                name="inference_service",
                config={
                    "model_architecture": "CustomCNN",
                    "device": "cpu"
                }
            )
        
        load_time = time.time() - start_time
        MODEL_LOADING_TIME.observe(load_time)
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    if not load_model():
        raise RuntimeError("Failed to load model during startup")

def process_image(image_data: bytes) -> torch.Tensor:
    """Process image data into tensor"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {str(e)}"
        )

@app.post("/predict", response_model=Dict[str, Any])
async def predict(file: UploadFile = File(...)):
    INFERENCE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read and process image
        image_data = await file.read()
        image_tensor = process_image(image_data)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        inference_time = time.time() - start_time
        
        # Log to W&B
        wandb.log({
            'inference_time': inference_time,
            'confidence': confidence,
            'predicted_class': predicted_class
        })
        
        INFERENCE_TIME.observe(inference_time)
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'inference_time': float(inference_time)
        }
        
    except HTTPException as e:
        INFERENCE_ERRORS.inc()
        raise e
    except Exception as e:
        INFERENCE_ERRORS.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Perform a simple inference to ensure model is working
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            model(dummy_input)
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(device),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy"
        )
