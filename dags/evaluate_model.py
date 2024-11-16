import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient, errors as mongo_errors
import wandb
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from airflow.models import Variable

# Configure logging with proper formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/yeahia/airflow/logs/model_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for centralized parameter management"""
    PREPROCESSED_DATA_PATH: str = '/home/yeahia/airflow/dataset_preprocessed'
    MODEL_PATH: str = '/home/yeahia/airflow/models/best_model.pth'
    EVALUATION_OUTPUT_PATH: str = '/home/yeahia/airflow/evaluation'
    MONGO_URI: str = "mongodb://localhost:27017/"
    DB_NAME: str = "image_pipeline"
    COLLECTION_NAME: str = "evaluation_metadata"
    BATCH_SIZE: int = 32
    NUM_CLASSES: int = 50
    WANDB_PROJECT: str = "yeahiasarkernabil-humufy"
    NUM_WORKERS: int = 4
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        return cls(
            PREPROCESSED_DATA_PATH=os.getenv('PREPROCESSED_DATA_PATH', cls.PREPROCESSED_DATA_PATH),
            MODEL_PATH=os.getenv('MODEL_PATH', cls.MODEL_PATH),
            MONGO_URI=os.getenv('MONGO_URI', cls.MONGO_URI),
            WANDB_PROJECT=os.getenv('WANDB_PROJECT', cls.WANDB_PROJECT),
        )

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
        
class ModelEvaluator:
    """Handles model evaluation and metric computation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_paths()
        
    def _setup_paths(self) -> None:
        """Ensure all necessary directories exist"""
        Path(self.config.EVALUATION_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        
    def load_model(self) -> Tuple[nn.Module, Dict]:
        """Load the trained model and its metadata"""
        try:
            model = CustomCNN(self.config.NUM_CLASSES).to(self.device)
            checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info(f"Successfully loaded model from {self.config.MODEL_PATH}")
            return model, checkpoint
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise AirflowException(f"Model loading failed: {str(e)}")

    def prepare_data(self) -> DataLoader:
        """Prepare the test dataset and dataloader"""
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Added consistent sizing
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            test_dataset = ImageFolder(
                self.config.PREPROCESSED_DATA_PATH,
                transform=transform
            )
            
            return DataLoader(
                test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True  # Added for better GPU utilization
            ), test_dataset.classes
        except Exception as e:
            logger.error(f"Failed to prepare data: {str(e)}")
            raise AirflowException(f"Data preparation failed: {str(e)}")

    def create_visualizations(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_scores: List[float],
        classes: List[str]
    ) -> Dict[str, str]:
        """Create and save visualization plots"""
        try:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(self.config.EVALUATION_OUTPUT_PATH, 'confusion_matrix.png')
            plt.savefig(cm_path, bbox_inches='tight', dpi=300)
            plt.close()

            # ROC Curve
            plt.figure(figsize=(12, 10))
            fpr, tpr, roc_auc = {}, {}, {}
            
            for i in range(len(classes)):
                fpr[i], tpr[i], _ = roc_curve(
                    (np.array(y_true) == i).astype(int),
                    [score[i] for score in y_scores]
                )
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                if i < 5:  # Plot only top 5 classes for clarity
                    plt.plot(
                        fpr[i], tpr[i],
                        label=f'{classes[i]} (AUC = {roc_auc[i]:0.2f})'
                    )
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (Top 5 Classes)')
            plt.legend(loc="lower right")
            
            roc_path = os.path.join(self.config.EVALUATION_OUTPUT_PATH, 'roc_curve.png')
            plt.savefig(roc_path, bbox_inches='tight', dpi=300)
            plt.close()

            return {'confusion_matrix': cm_path, 'roc_curve': roc_path}
        except Exception as e:
            logger.error(f"Failed to create visualizations: {str(e)}")
            raise AirflowException(f"Visualization creation failed: {str(e)}")

class MongoDBHandler:
    """Handles MongoDB operations with proper connection management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        
    def __enter__(self):
        try:
            self.client = MongoClient(
                self.config.MONGO_URI,
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )
            self.client.server_info()  # Test connection
            return self
        except mongo_errors.ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            raise AirflowException("Could not connect to MongoDB")
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
            
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save evaluation results to MongoDB"""
        try:
            collection = self.client[self.config.DB_NAME][self.config.COLLECTION_NAME]
            result = collection.insert_one(results)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to save results to MongoDB: {str(e)}")
            raise AirflowException(f"MongoDB save operation failed: {str(e)}")

def evaluate_model(**context) -> None:
    """Main evaluation function for the Airflow task"""
    config = Config.from_env()
    evaluator = ModelEvaluator(config)
    
    try:
        # Initialize W&B
        wandb.init(project=config.WANDB_PROJECT, name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Load model and prepare data
        model, checkpoint = evaluator.load_model()
        test_loader, class_names = evaluator.prepare_data()
        
        # Evaluation loop
        all_predictions = []
        all_targets = []
        all_scores = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                try:
                    inputs, targets = inputs.to(evaluator.device), targets.to(evaluator.device)
                    outputs = model(inputs)
                    scores = torch.nn.functional.softmax(outputs, dim=1)
                    
                    _, predicted = outputs.max(1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Processed {batch_idx * config.BATCH_SIZE} samples")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Create visualizations
        viz_paths = evaluator.create_visualizations(
            all_targets, all_predictions, all_scores, class_names
        )
        
        # Calculate per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = \
            precision_recall_fscore_support(all_targets, all_predictions, average=None)
        
        per_class_metrics = {
            class_name: {
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1)
            }
            for class_name, prec, rec, f1 in zip(
                class_names, per_class_precision, per_class_recall, per_class_f1
            )
        }
        
        # Prepare evaluation results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': config.MODEL_PATH,
            'model_checkpoint_epoch': checkpoint.get('epoch', None),
            'overall_metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
            },
            'per_class_metrics': per_class_metrics,
            'visualization_paths': viz_paths,
            'evaluation_parameters': {
                'batch_size': config.BATCH_SIZE,
                'device': str(evaluator.device),
                'num_test_samples': len(test_loader.dataset)
            }
        }
        
        # Save results to MongoDB
        with MongoDBHandler(config) as mongo:
            mongo.save_results(evaluation_results)
        
        # Log results to W&B
        wandb.log({
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1': f1,
            'confusion_matrix': wandb.Image(viz_paths['confusion_matrix']),
            'roc_curve': wandb.Image(viz_paths['roc_curve'])
        })
        
        # Push results to XCom
        context['task_instance'].xcom_push(
            key='evaluation_results',
            value={
                'overall_metrics': evaluation_results['overall_metrics'],
                'visualization_paths': evaluation_results['visualization_paths']
            }
        )
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise AirflowException(f"Model evaluation failed: {str(e)}")
    finally:
        wandb.finish()

# Create the DAG
default_args = {
    'owner': 'yeahia',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)  # Added timeout
}

dag = DAG(
    'model_evaluation_pipeline',
    default_args=default_args,
    description='Production pipeline for evaluating CNN model performance',
    schedule_interval=None,
    catchup=False,
    tags=['evaluation', 'deep_learning', 'cnn'],
    max_active_runs=1  # Prevent multiple concurrent runs
)

evaluation_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
    provide_context=True,
    retries=2,  # Number of retries if task fails
    retry_delay=timedelta(minutes=5),  # Delay between retries
    retry_exponential_backoff=True,  # Exponential backoff for retries
    max_retry_delay=timedelta(minutes=30),  # Maximum delay between retries
    email_on_retry=True,  # Send email notification on retry
    email_on_failure=True,  # Send email notification on failure
    execution_timeout=timedelta(hours=2),  # Maximum execution time
    sla=timedelta(hours=4),  # Service Level Agreement
    depends_on_past=False,  # Don't depend on past executions
    trigger_rule='all_success',  # Only trigger if all upstream tasks succeed
    pool='model_evaluation_pool',  # Resource pool for the task
)

if __name__ == "__main__":
    dag.cli()
