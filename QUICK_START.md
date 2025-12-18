# Quick Start Guide - 5 Minutes to Running AVM

## Prerequisites
- Python 3.10+ installed
- Virtual environment activated
- Dependencies installed (`pip install -r requirements.txt`)

## Option A: Automated (Fastest)

```bash
# Run the complete example workflow
python run_example.py

# Start the API server
python app/main.py
```

Visit http://localhost:8000/docs to see your API!

## Option B: Manual Steps

### 1. Create Sample Data (30 seconds)
```bash
python scripts/prepare_dataset.py --create_sample --n_samples 1000 --split
```

### 2. Train Tabular Model (1-2 minutes)
```bash
python ml/train_tabular.py --data_path data/processed/train.csv --output_dir models
```

### 3. Start API Server (instant)
```bash
python app/main.py
```

### 4. Make a Prediction
Open http://localhost:8000/docs and try the `/predict/` endpoint with this example:

```json
{
  "property_id": "TEST_001",
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
}
```

## Option C: Docker (For Production)

```bash
cd infra
cp .env.example .env
docker-compose up -d
```

Access:
- API: http://localhost:8000/docs
- pgAdmin: http://localhost:5050
- Database: localhost:5432

## Testing Your API

### Using cURL
```bash
curl http://localhost:8000/health
```

### Using Python
```python
import requests

response = requests.get('http://localhost:8000/health')
print(response.json())
```

### Using Browser
Visit http://localhost:8000/docs for interactive API documentation

## What You Get

✅ **3 ML Models**: Tabular, Image, Text
✅ **Ensemble Stacker**: Combines all models
✅ **REST API**: Full CRUD operations
✅ **Database Schema**: PostgreSQL ready
✅ **Docker Setup**: Production ready
✅ **Documentation**: Complete guides

## Next Steps

1. **Train more models** (optional):
   ```bash
   python ml/train_image.py --data_path data/processed/train.csv --output_dir models
   python ml/train_text.py --data_path data/processed/train.csv --output_dir models
   python ml/train_stack.py --data_path data/processed/train.csv --model_dir models --output_dir models/stacker
   ```

2. **Use your own data**:
   - Replace sample data with your CSV
   - Ensure columns match expected schema
   - Retrain models

3. **Deploy to production**:
   - Use Docker Compose
   - Configure environment variables
   - Set up monitoring

## Troubleshooting

**Import errors?**
```bash
# Make sure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/macOS

# Reinstall dependencies
pip install -r requirements.txt
```

**Models not loading?**
- Make sure you trained at least the tabular model
- Check that models/ directory contains .joblib files
- Verify MODEL_DIR environment variable

**Port already in use?**
```bash
# Change port in app/main.py or use environment variable
PORT=8001 python app/main.py
```

## Resources

- **Full Documentation**: See README.md
- **Usage Guide**: See USAGE_GUIDE.md
- **Project Summary**: See PROJECT_SUMMARY.md
- **API Docs**: http://localhost:8000/docs (when running)

## Success Indicators

✓ No errors during data preparation
✓ Model training completes with metrics printed
✓ API starts without errors
✓ Health check returns "healthy" status
✓ Prediction endpoint returns price estimate

That's it! You now have a complete property valuation system running! 🎉
