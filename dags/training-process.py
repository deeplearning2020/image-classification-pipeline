from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import os
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb
from pymongo import MongoClient
import cv2
import numpy as np
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management class"""
    def __init__(self):
        # Paths
        self.PREPROCESSED_DATA_PATH = os.getenv('PREPROCESSED_DATA_PATH', '/home/yeahia/airflow/dataset_preprocessed')
        self.MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', '/home/yeahia/airflow/models')
        
        # Database
        self.MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        self.DB_NAME = os.getenv('DB_NAME', 'image_pipeline')
        
        # Training
        self.WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'yeahiasarkernabil-humufy')
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
        self.NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '10'))
        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
        self.NUM_CLASSES = int(os.getenv('NUM_CLASSES', '50'))
        self.NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))
        
        # Model checkpointing
        self.CHECKPOINT_FILENAME = 'best_model.pth'
        
        # Input normalization
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225]

class CustomCNN(nn.Module):
    """Custom CNN architecture for image classification"""
    def __init__(self, num_classes: int):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

class CustomImageDataset(Dataset):
    """Custom Dataset class for loading preprocessed images"""
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from directory structure"""
        for class_idx, class_name in enumerate(os.listdir(self.data_path)):
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_path, image_name))
                        self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cpu")
        self.model = CustomCNN(config.NUM_CLASSES).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.best_val_accuracy = 0.0
        self.training_metadata = {
            'epoch_metrics': [],
            'training_start_time': datetime.now().isoformat()
        }

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': self.best_val_accuracy
        }, os.path.join(self.config.MODEL_SAVE_PATH, self.config.CHECKPOINT_FILENAME))

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_accuracy': 100. * correct / total
                })

        return train_loss / len(train_loader), 100. * correct / total

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return val_loss / len(val_loader), 100. * correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training process"""
        try:
            for epoch in range(self.config.NUM_EPOCHS):
                # Training phase
                train_loss, train_accuracy = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_accuracy = self._validate(val_loader)

                # Log metrics
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                })

                # Store epoch metrics
                self.training_metadata['epoch_metrics'].append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'timestamp': datetime.now().isoformat()
                })

                # Save best model
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self._save_checkpoint(epoch)

            return {
                'best_val_accuracy': self.best_val_accuracy,
                'final_train_accuracy': train_accuracy,
                'model_path': os.path.join(self.config.MODEL_SAVE_PATH, 
                                         self.config.CHECKPOINT_FILENAME)
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

class DataManager:
    """Handles data preparation and loading"""
    def __init__(self, config: Config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN,
                              std=config.NORMALIZE_STD)
        ])

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation data loaders"""
        try:
            full_dataset = CustomImageDataset(
                self.config.PREPROCESSED_DATA_PATH, 
                transform=self.transform
            )
            
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS
            )

            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

def train_model(**context) -> None:
    """Main training function"""
    config = Config()
    try:
        # Initialize wandb within the training task
        wandb.init(
            project=config.WANDB_PROJECT,
            config={
                "learning_rate": config.LEARNING_RATE,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.NUM_EPOCHS,
                "model_architecture": "CustomCNN",
                "dataset_path": config.PREPROCESSED_DATA_PATH
            }
        )
        
        # Initialize components
        data_manager = DataManager(config)
        trainer = ModelTrainer(config)
        
        # Prepare data
        train_loader, val_loader = data_manager.prepare_data()
        
        # Train model
        training_summary = trainer.train(train_loader, val_loader)
        
        # Save metadata to MongoDB
        with MongoClient(config.MONGO_URI) as client:
            db = client[config.DB_NAME]
            collection = db['training_metadata']
            trainer.training_metadata['training_end_time'] = datetime.now().isoformat()
            trainer.training_metadata['best_val_accuracy'] = trainer.best_val_accuracy
            collection.insert_one(trainer.training_metadata)
        
        # Push metrics to XCom
        context['task_instance'].xcom_push(
            key='training_summary',
            value=training_summary
        )

    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        raise
    finally:
        wandb.finish()

# DAG definition
default_args = {
    'owner': 'yeahia',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Production pipeline for training CNN model with W&B integration',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['training', 'deep_learning', 'cnn']
)

# DAG tasks
training_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# No need for task dependencies anymore since we only have one task
