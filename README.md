# CNN Image Pipeline with Airflow

![Python](https://img.shields.io/badge/python-3.10.0-blue)
![Airflow](https://img.shields.io/badge/airflow-2.7.3-red)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-orange)
![FastAPI](https://img.shields.io/badge/fastapi-0.115.5-green)
![License](https://img.shields.io/badge/license-MIT-blue)

A production-grade pipeline for CNN model development, training, and deployment using Apache Airflow.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Pipeline Components](#pipeline-components)
- [Usage](#usage)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Prerequisites

- Python 3.10.0
- Docker 24.0+
- MongoDB 6.0+
- CUDA 11.8+ (for GPU support)
- 16GB RAM (minimum)
- 50GB free disk space

## 🚀 Installation

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Linux/Mac:
source venv/bin/activate
# For Windows:
.\venv\Scripts\activate
```

### 2. Install required packages

```bash
# Update pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Set up Airflow

```bash
# Set Airflow home
export AIRFLOW_HOME=~/airflow

# Initialize Airflow database
airflow db init

# Create Airflow user
airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 4. Configure MongoDB

```bash
# Install MongoDB (Ubuntu)
sudo apt-get update
sudo apt-get install -y mongodb

# Start MongoDB service
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

### 5. Set up Weights & Biases

```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login
```

### 6. Configure Docker

```bash
# Install Docker
sudo apt-get update
sudo apt-get install docker.io

# Add user to docker group
sudo usermod -aG docker $USER

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

## 📁 Project Structure

```
airflow/
├── dags/
│   ├── preprocessing_dag.py
│   ├── training_dag.py
│   ├── evaluation_dag.py
│   └── deployment_dag.py
├── dataset/
│   └── raw/
├── models/
├── docker/
└── logs/
```

## ⚙️ Configuration

### 1. Environment Variables

```bash
# Create .env file
cat << EOF > $AIRFLOW_HOME/.env
AIRFLOW_HOME=~/airflow
MONGO_URI=mongodb://localhost:27017/
WANDB_PROJECT=yeahiasarkernabil-humufy
DATASET_PATH=/home/yeahia/airflow/dataset
MODEL_PATH=/home/yeahia/airflow/models
DOCKER_PATH=/home/yeahia/airflow/docker
EOF

# Load environment variables
source $AIRFLOW_HOME/.env
```

### 2. Airflow Configuration

```bash
# Update airflow.cfg
sed -i 's/load_examples = True/load_examples = False/g' $AIRFLOW_HOME/airflow.cfg
sed -i 's/dag_file_processor_timeout = 50/dag_file_processor_timeout = 600/g' $AIRFLOW_HOME/airflow.cfg
```

## 🔄 Pipeline Components

### 1. Data Preprocessing DAG
- Image preprocessing
- Data augmentation
- Metadata extraction
- MongoDB storage

### 2. Model Training DAG
- CNN model training
- W&B integration
- Checkpoint management
- Performance logging

### 3. Model Evaluation DAG
- Model evaluation
- Metrics calculation
- Result visualization
- Performance analysis

### 4. Model Deployment DAG
- Docker containerization
- FastAPI service
- Prometheus monitoring
- Load balancing

## 💻 Usage

### 1. Start Airflow services

```bash
# Start webserver
airflow webserver --port 8080 -D

# Start scheduler
airflow scheduler -D
```

### 2. Trigger pipelines

```bash
# Trigger preprocessing DAG
airflow dags trigger image_preprocessing_pipeline

# Trigger training DAG
airflow dags trigger model_training_pipeline

# Trigger evaluation DAG
airflow dags trigger model_evaluation_pipeline

# Trigger deployment DAG
airflow dags trigger model_deployment_pipeline
```

### 3. Monitor execution

```bash
# View DAG logs
airflow dags show-log image_preprocessing_pipeline
```

## 📊 Monitoring

### 1. Access Airflow UI
- Open `http://localhost:8080` in your browser
- Login with admin credentials

### 2. View W&B Dashboard
- Open `https://wandb.ai/yeahiasarkernabil-humufy`
- Monitor training metrics

### 3. Check MongoDB Status

```bash
# Connect to MongoDB shell
mongosh

# Check collections
use image_pipeline
show collections
```

## ❗ Troubleshooting

### Common Issues

1. **Permission Errors**
```bash
# Fix directory permissions
sudo chown -R $USER:$USER ~/airflow
```

2. **MongoDB Connection Issues**
```bash
# Check MongoDB status
sudo systemctl status mongodb
```

3. **Docker Build Failures**
```bash
# Clean Docker cache
docker system prune -a

# Check Docker logs
docker logs cnn_inference_service
```
