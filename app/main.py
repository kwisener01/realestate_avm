"""
FastAPI application for Property Valuation AVM
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import routes_predict, routes_properties, routes_sheets
from app.models.property_models import HealthCheck
from ml.stacker import StackerModel

# Global model instance
stacker_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the application.
    Loads ML models on startup and cleans up on shutdown.
    """
    global stacker_model

    print("Starting Property Valuation AVM...")

    # Load models
    model_dir = os.getenv("MODEL_DIR", "models/stacker")

    if os.path.exists(model_dir):
        try:
            print(f"Loading models from {model_dir}...")
            stacker_model = StackerModel(
                tabular_model_path=os.path.join(model_dir, "tabular_model.joblib"),
                image_model_path=os.path.join(model_dir, "image_model.pth")
                if os.path.exists(os.path.join(model_dir, "image_model.pth")) else None,
                text_model_path=os.path.join(model_dir, "text_model.pth")
                if os.path.exists(os.path.join(model_dir, "text_model.pth")) else None,
                meta_model_path=os.path.join(model_dir, "meta_model.joblib")
            )
            routes_predict.set_model(stacker_model)
            routes_sheets.set_model(stacker_model)
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            print("API will start but predictions will not be available.")
    else:
        print(f"Model directory {model_dir} not found. Starting without models.")

    yield

    # Cleanup on shutdown
    print("Shutting down Property Valuation AVM...")


# Create FastAPI app
app = FastAPI(
    title="Property Valuation AVM API",
    description="""
    Automated Valuation Model (AVM) for real estate properties.

    This API provides machine learning-based property valuations using:
    - **Tabular Model**: Analyzes numeric and categorical features
    - **Image Model**: Evaluates property photos (if available)
    - **Text Model**: Processes property descriptions (if available)
    - **Ensemble Model**: Combines all models for optimal predictions

    ## Features
    - Single and batch predictions
    - Property data management
    - Model performance monitoring
    - RESTful API design
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes_predict.router)
app.include_router(routes_properties.router)
app.include_router(routes_sheets.router)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", tags=["root"])
async def root():
    """Serve the web UI"""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)

    # Fallback to API info if static file doesn't exist
    return {
        "message": "Property Valuation AVM API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and loaded models.
    """
    models_loaded = {
        "tabular": False,
        "image": False,
        "text": False,
        "meta": False
    }

    if stacker_model:
        models_loaded["tabular"] = stacker_model.tabular_model is not None
        models_loaded["image"] = stacker_model.image_model is not None
        models_loaded["text"] = stacker_model.text_model is not None
        models_loaded["meta"] = stacker_model.meta_model is not None

    return HealthCheck(
        status="healthy" if any(models_loaded.values()) else "degraded",
        timestamp=datetime.now(),
        models_loaded=models_loaded
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
