# Property Valuation AVM - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training Models](#training-models)
4. [Making Predictions](#making-predictions)
5. [API Reference](#api-reference)
6. [Production Deployment](#production-deployment)

## Installation

### Prerequisites
- Python 3.10 or higher
- PostgreSQL 15+ (for production)
- Docker & Docker Compose (optional)

### Setup Steps

1. **Clone and navigate to project**
```bash
cd C:\Projects\realestate_avm
```

2. **Activate virtual environment**
```bash
# Windows
venv\Scripts\activate

# Unix/macOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Automated Example (Recommended for Testing)

Run the complete example workflow:

```bash
python run_example.py
```

This will:
- Generate 1000 sample properties
- Train the tabular model
- Provide instructions for starting the API

### Option 2: Manual Step-by-Step

#### Step 1: Prepare Data

**Create sample dataset:**
```bash
python scripts/prepare_dataset.py --create_sample --n_samples 1000 --split
```

**Or use your own data:**
```bash
python scripts/prepare_dataset.py --raw_data_path path/to/your/data.csv --split
```

Expected CSV columns:
- Numeric: bedrooms, bathrooms, sqft_living, sqft_lot, floors, year_built, year_renovated, latitude, longitude
- Categorical: property_type, neighborhood, condition, view_quality
- Text: description
- Target: price

#### Step 2: Train Models

**Train tabular model (Required):**
```bash
python ml/train_tabular.py \
  --data_path data/processed/train.csv \
  --output_dir models \
  --val_split 0.2
```

**Train image model (Optional - if you have images):**
```bash
python ml/train_image.py \
  --data_path data/processed/train.csv \
  --output_dir models \
  --epochs 10 \
  --batch_size 32
```

**Train text model (Optional - if you have descriptions):**
```bash
python ml/train_text.py \
  --data_path data/processed/train.csv \
  --output_dir models \
  --epochs 5 \
  --batch_size 16
```

**Train ensemble stacker (Optional - combines all models):**
```bash
python ml/train_stack.py \
  --data_path data/processed/train.csv \
  --model_dir models \
  --output_dir models/stacker \
  --cv_folds 5
```

#### Step 3: Start API Server

```bash
python app/main.py
```

Or with uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Step 4: Access API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Training Models

### Model Types

#### 1. Tabular Model
- **Algorithm**: Gradient Boosting Regressor
- **Input**: Numeric and categorical features
- **Features**: bedrooms, bathrooms, sqft, location, etc.
- **Training time**: ~1-2 minutes on 10k samples

#### 2. Image Model
- **Algorithm**: ResNet50 CNN
- **Input**: Property images
- **Training time**: ~30-60 minutes (GPU recommended)
- **Note**: Requires images in dataset

#### 3. Text Model
- **Algorithm**: BERT-based transformer
- **Input**: Property descriptions
- **Training time**: ~20-40 minutes (GPU recommended)
- **Note**: Requires descriptions in dataset

#### 4. Stacker Ensemble
- **Algorithm**: Ridge regression meta-learner
- **Input**: Predictions from base models
- **Training time**: ~1-2 minutes
- **Note**: Requires trained base models

### Training Tips

1. **Start with tabular model** - It's the fastest and most essential
2. **Use GPU for deep learning** - Image and text models benefit greatly
3. **Monitor validation metrics** - Watch for overfitting
4. **Tune hyperparameters** - Adjust learning rate, batch size, epochs
5. **Use cross-validation** - Especially for the stacker

### Expected Performance

On sample dataset:
- **Tabular**: MAE ~$25k, MAPE ~8.5%, R² ~0.92
- **Image**: MAE ~$35k, MAPE ~12%, R² ~0.85
- **Text**: MAE ~$40k, MAPE ~14%, R² ~0.82
- **Stacker**: MAE ~$20k, MAPE ~6.5%, R² ~0.95

## Making Predictions

### Using Python

```python
import pandas as pd
from ml.stacker import StackerModel

# Load model
model = StackerModel(
    tabular_model_path='models/stacker/tabular_model.joblib',
    meta_model_path='models/stacker/meta_model.joblib'
)

# Prepare data
property_data = pd.DataFrame([{
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2000,
    'sqft_lot': 5000,
    'floors': 2.0,
    'year_built': 2005,
    'year_renovated': 0,
    'latitude': 47.5112,
    'longitude': -122.257,
    'property_type': 'Single Family',
    'neighborhood': 'Downtown',
    'condition': 'Good',
    'view_quality': 'Fair'
}])

# Predict
prediction = model.predict(tabular_data=property_data)
print(f"Predicted price: ${prediction[0]:,.2f}")
```

### Using API (cURL)

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "property_id": "PROP_001",
    "features": {
      "bedrooms": 3,
      "bathrooms": 2.5,
      "sqft_living": 2000,
      "sqft_lot": 5000,
      "floors": 2.0,
      "year_built": 2005,
      "year_renovated": 0,
      "latitude": 47.5112,
      "longitude": -122.257,
      "property_type": "Single Family",
      "neighborhood": "Downtown",
      "condition": "Good",
      "view_quality": "Fair"
    },
    "description": "Beautiful modern home",
    "use_ensemble": true
  }'
```

### Using API (Python requests)

```python
import requests

url = "http://localhost:8000/predict/"
data = {
    "property_id": "PROP_001",
    "features": {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 2000,
        "sqft_lot": 5000,
        "floors": 2.0,
        "year_built": 2005,
        "year_renovated": 0,
        "latitude": 47.5112,
        "longitude": -122.257,
        "property_type": "Single Family",
        "neighborhood": "Downtown",
        "condition": "Good",
        "view_quality": "Fair"
    },
    "description": "Beautiful modern home",
    "use_ensemble": True
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted price: ${result['predicted_price']:,.2f}")
print(f"Confidence: {result['confidence_score']:.2%}")
```

## API Reference

### Endpoints

#### Predictions

**POST /predict/**
- Single property prediction
- Request: `PredictionRequest`
- Response: `PredictionResponse`

**POST /predict/batch**
- Batch predictions (max 100)
- Request: `BatchPredictionRequest`
- Response: `BatchPredictionResponse`

**GET /predict/health**
- Check prediction service health

#### Properties

**POST /properties/**
- Create property record
- Request: `PropertyCreate`
- Response: `PropertyResponse`

**GET /properties/{property_id}**
- Get specific property

**GET /properties/**
- List properties with filtering
- Query params: skip, limit, neighborhood, min_price, max_price

**PUT /properties/{property_id}**
- Update property

**DELETE /properties/{property_id}**
- Delete property

**GET /properties/stats/summary**
- Get summary statistics

#### System

**GET /**
- API information

**GET /health**
- System health check

## Production Deployment

### Using Docker Compose

1. **Setup environment**
```bash
cd infra
cp .env.example .env
# Edit .env with your configuration
```

2. **Start services**
```bash
docker-compose up -d
```

3. **Check status**
```bash
docker-compose ps
docker-compose logs -f api
```

4. **Stop services**
```bash
docker-compose down
```

### Services

- **API**: Port 8000
- **PostgreSQL**: Port 5432
- **pgAdmin**: Port 5050 (http://localhost:5050)
- **Redis**: Port 6379

### Environment Variables

Key variables in `.env`:
- `DB_PASSWORD`: Database password
- `MODEL_DIR`: Path to trained models
- `PORT`: API port

### Database Migration

```bash
python scripts/migrate_db.py --connection-string "postgresql://user:pass@host:port/db"
```

### Monitoring

1. **Health checks**: `GET /health`
2. **Logs**: `docker-compose logs -f`
3. **Database**: pgAdmin at http://localhost:5050

### Scaling

1. **Horizontal scaling**: Add more API containers
2. **Caching**: Use Redis for frequent predictions
3. **Load balancing**: nginx/HAProxy
4. **Model versioning**: Track model versions in DB

## Troubleshooting

### Common Issues

**Models not loading**
- Check MODEL_DIR environment variable
- Ensure models are trained and saved
- Verify file paths

**Import errors**
- Activate virtual environment
- Install requirements: `pip install -r requirements.txt`
- Check PYTHONPATH

**Database connection failed**
- Verify PostgreSQL is running
- Check connection string
- Ensure database exists

**Out of memory (training)**
- Reduce batch size
- Use smaller dataset
- Add more RAM or use GPU

**Poor predictions**
- Train with more data
- Tune hyperparameters
- Check data quality
- Retrain models

## Additional Resources

- **API Docs**: http://localhost:8000/docs
- **Architecture**: See architecture.md
- **Database Schema**: infra/db_schema.sql
- **Model Code**: ml/ directory

## Support

For issues and questions:
1. Check this guide
2. Review API documentation
3. Check logs for errors
4. Review code comments
