"""
Image Processing Pipeline DAG
----------------------------
This DAG handles the automated processing of image datasets, including validation,
preprocessing, and metadata storage in MongoDB.

Features:
- Comprehensive error handling and logging
- Performance monitoring and metrics
- Configuration management
- Data validation
- Automated retries for transient failures
- Detailed execution tracking

"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from typing import Dict, List, Optional, Union
import traceback
from dataclasses import dataclass
from enum import Enum
import pymongo
from pymongo.errors import PyMongoError
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
class ImageFormat(Enum):
    """Supported image format extensions"""
    JPG = "*.jpg"
    JPEG = "*.jpeg"
    PNG = "*.png"

@dataclass
class PipelineConfig:
    """Pipeline configuration parameters"""
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_COLLECTION: str
    DATASET_PATH: str
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5

    @classmethod
    def from_airflow_variables(cls) -> 'PipelineConfig':
        """Load configuration from Airflow variables"""
        try:
            return cls(
                MONGODB_URI=Variable.get("MONGODB_URI"),
                MONGODB_DB=Variable.get("MONGODB_DB"),
                MONGODB_COLLECTION=Variable.get("MONGODB_COLLECTION"),
                DATASET_PATH=Variable.get("DATASET_PATH"),
                BATCH_SIZE=int(Variable.get("BATCH_SIZE", 100)),
                MAX_RETRIES=int(Variable.get("MAX_RETRIES", 3)),
                RETRY_DELAY=int(Variable.get("RETRY_DELAY", 5))
            )
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise AirflowException(f"Configuration loading failed: {str(e)}")

class MongoDBHandler:
    """Handles MongoDB connections and operations with proper error handling"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = None
        self.db = None
        self.collection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self) -> None:
        """Establish MongoDB connection with retry logic"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.client = pymongo.MongoClient(self.config.MONGODB_URI)
                self.db = self.client[self.config.MONGODB_DB]
                self.collection = self.db[self.config.MONGODB_COLLECTION]
                # Test connection
                self.client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                return
            except PyMongoError as e:
                logger.warning(f"MongoDB connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.MAX_RETRIES - 1:
                    raise AirflowException(f"Failed to connect to MongoDB after {self.config.MAX_RETRIES} attempts")
                time.sleep(self.config.RETRY_DELAY)

    def store_metadata(self, metadata_list: List[Dict]) -> None:
        """Store metadata in MongoDB with batch processing"""
        try:
            for i in range(0, len(metadata_list), self.config.BATCH_SIZE):
                batch = metadata_list[i:i + self.config.BATCH_SIZE]
                self.collection.insert_many(batch)
                logger.info(f"Stored batch of {len(batch)} records in MongoDB")
        except PyMongoError as e:
            logger.error(f"Failed to store metadata in MongoDB: {str(e)}")
            raise AirflowException(f"MongoDB storage failed: {str(e)}")

    def close(self) -> None:
        """Safely close MongoDB connection"""
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB connection closed")
            except PyMongoError as e:
                logger.warning(f"Error while closing MongoDB connection: {str(e)}")

class DatasetValidator:
    """Validates dataset structure and image files"""

    @staticmethod
    def validate_dataset_structure(config: PipelineConfig) -> None:
        """
        Validates the dataset directory structure and image files.
        
        Args:
            config: Pipeline configuration object
        
        Raises:
            AirflowException: If validation fails
        """
        try:
            dataset_path = Path(config.DATASET_PATH)
            if not dataset_path.exists():
                raise AirflowException(f"Dataset path does not exist: {dataset_path}")

            # Validate directory structure
            image_files = []
            for format_type in ImageFormat:
                image_files.extend(list(dataset_path.rglob(format_type.value)))

            if not image_files:
                raise AirflowException("No valid image files found in dataset")

            # Basic image validation
            for image_path in image_files:
                if image_path.stat().st_size == 0:
                    raise AirflowException(f"Empty image file detected: {image_path}")

            logger.info(f"Dataset validation successful. Found {len(image_files)} valid images.")
            return True

        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            logger.error(f"Stacktrace: {traceback.format_exc()}")
            raise AirflowException(f"Dataset validation failed: {str(e)}")

class ImagePreprocessor:
    """Handles image preprocessing and metadata extraction"""

    @staticmethod
    def preprocess_single_image(image_path: str) -> Dict:
        """
        Preprocesses a single image and extracts metadata.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dict containing image metadata
        """
        try:
            # Implementation would go here
            # This is a placeholder that returns basic metadata
            return {
                "file_path": image_path,
                "file_size": Path(image_path).stat().st_size,
                "processed_at": datetime.now().isoformat(),
                "status": "processed"
            }
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            return {
                "file_path": image_path,
                "error": str(e),
                "processed_at": datetime.now().isoformat(),
                "status": "failed"
            }

def process_dataset(**context) -> None:
    """
    Main function to process the entire dataset and store metadata.
    
    Args:
        context: Airflow context containing task instance information
    
    Raises:
        AirflowException: If processing fails
    """
    config = PipelineConfig.from_airflow_variables()
    start_time = time.time()
    processed_count = 0
    failed_count = 0

    try:
        with MongoDBHandler(config) as db_handler:
            # Get list of all images
            image_files = []
            for format_type in ImageFormat:
                image_files.extend(list(Path(config.DATASET_PATH).rglob(format_type.value)))

            # Process images and collect metadata
            metadata_list = []
            for image_path in image_files:
                metadata = ImagePreprocessor.preprocess_single_image(str(image_path))
                if metadata["status"] == "processed":
                    processed_count += 1
                else:
                    failed_count += 1
                metadata_list.append(metadata)

            # Store metadata in MongoDB
            db_handler.store_metadata(metadata_list)

            # Calculate processing metrics
            processing_time = time.time() - start_time
            success_rate = (processed_count / len(image_files)) * 100

            # Store summary in XCom for downstream tasks
            summary = {
                'total_images': len(image_files),
                'processed_count': processed_count,
                'failed_count': failed_count,
                'success_rate': success_rate,
                'processing_time_seconds': processing_time,
                'preprocessing_completed': datetime.now().isoformat()
            }
            
            context['task_instance'].xcom_push(
                key='preprocessing_summary',
                value=summary
            )

            logger.info(f"Processing complete. Summary: {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Dataset processing failed: {str(e)}")
        logger.error(f"Stacktrace: {traceback.format_exc()}")
        raise AirflowException(f"Dataset processing failed: {str(e)}")

# DAG definition
default_args = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'email': ['data_engineering@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'on_failure_callback': None,  # Add your callback function if needed
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'image_preprocessing_pipeline',
    default_args=default_args,
    description='Production pipeline for preprocessing images and extracting metadata',
    schedule_interval=None,
    catchup=False,
    tags=['preprocessing', 'computer_vision', 'production'],
    max_active_runs=1
)

# Define tasks with proper error handling and monitoring
validation_task = PythonOperator(
    task_id='validate_dataset',
    python_callable=DatasetValidator.validate_dataset_structure,
    op_kwargs={'config': PipelineConfig.from_airflow_variables()},
    dag=dag
)

preprocessing_task = PythonOperator(
    task_id='preprocess_dataset',
    python_callable=process_dataset,
    dag=dag
)

# Set task dependencies
validation_task >> preprocessing_task
