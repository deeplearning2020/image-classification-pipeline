
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import shutil

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'fastapi_model_deployment',
    default_args=default_args,
    description='Deploy ML model using FastAPI in Docker',
    schedule_interval='@daily',
    catchup=False
)

def prepare_deployment_files():
    deployment_dir = '/home/yeahia/airflow/deployment'
    model_path = '/home/yeahia/airflow/models/best_model.pth'
    
    # Create deployment directory
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Copy model file
    shutil.copy2(model_path, os.path.join(deployment_dir, 'best_model.pth'))
    
    # Create main.py
    fastapi_code = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI(title="Model Prediction API",
             description="API for making predictions using the PyTorch model",
             version="1.0.0")

# Load the model
try:
    model = torch.load('best_model.pth')
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class PredictionInput(BaseModel):
    features: list

    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0]
            }
        }

class PredictionOutput(BaseModel):
    prediction: list

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Input validation
        if not input_data.features:
            raise HTTPException(status_code=400, detail="No features provided")
            
        # Convert input to tensor
        input_tensor = torch.FloatTensor(input_data.features)
        
        # Reshape if needed
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        return PredictionOutput(prediction=prediction.tolist())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
"""
    
    with open(os.path.join(deployment_dir, 'main.py'), 'w') as f:
        f.write(fastapi_code)
    
    # Create requirements.txt
    requirements = """
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
numpy==1.24.3
pydantic==2.5.2
python-multipart==0.0.6
"""
    
    with open(os.path.join(deployment_dir, 'requirements.txt'), 'w') as f:
        f.write(requirements)
    
    # Create Dockerfile
    dockerfile = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
    
    with open(os.path.join(deployment_dir, 'Dockerfile'), 'w') as f:
        f.write(dockerfile)

    # Create docker-compose.yml
    docker_compose = """
version: '3.8'

services:
  fastapi:
    build: .
    container_name: model_prediction_api
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
"""
    
    with open(os.path.join(deployment_dir, 'docker-compose.yml'), 'w') as f:
        f.write(docker_compose)

# Define tasks
prepare_files = PythonOperator(
    task_id='prepare_deployment_files',
    python_callable=prepare_deployment_files,
    dag=dag
)

# Deploy service
deploy_service = BashOperator(
    task_id='deploy_service',
    bash_command='cd /home/yeahia/airflow/deployment && docker-compose up -d --build',
    dag=dag
)

# Health check
check_service = BashOperator(
    task_id='check_service',
    bash_command='''
        for i in {1..5}; do
            if curl -f http://localhost:8000/health; then
                exit 0
            fi
            sleep 5
        done
        exit 1
    ''',
    dag=dag
)

# Set task dependencies
prepare_files >> deploy_service >> check_service
