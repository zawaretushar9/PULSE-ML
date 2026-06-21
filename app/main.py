"""
Pulse AutoML Platform
Copyright © 2026 TuViZa. All rights reserved.
Designed & Developed by Tushar Vijay Zaware
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
import shutil
import uuid
from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.core.config import settings
from app.core.ml_engine import MLEngine
import mlflow

app = FastAPI(title=settings.PROJECT_NAME)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all. Change to specific origins for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_engine = MLEngine()

class TrainRequest(BaseModel):
    filename: str
    target_column: Optional[str] = None
    task_type: Optional[str] = "auto"
    premium_mode: Optional[bool] = False
    existing_study_name: Optional[str] = None
    n_trials: Optional[int] = 10

@app.get("/")
async def root():
    return {"message": "Welcome to Pulse AutoML API"}

@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        df = pd.read_csv(file_path)
        analysis = ml_engine.profile_data(df)
        analysis["filename"] = filename
        return analysis
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing CSV: {str(e)}")

@app.post("/api/train-automl")
async def train_automl(request: TrainRequest):
    file_path = os.path.join(settings.UPLOAD_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        print(f"🚀 Starting training for file: {request.filename}")
        df = pd.read_csv(file_path)
        
        target_col = request.target_column
        if request.premium_mode and not target_col:
            print("💎 Premium Mode: Auto-detecting target...")
            target_col = ml_engine.auto_detect_target(df)
        
        if not target_col:
            target_col = ml_engine.profile_data(df)["suggested_target"]

        print(f"🎯 Target Column: {target_col}")
        print(f"📊 Task Type: {request.task_type}")

        results = ml_engine.run_automl(
            df=df,
            target_col=target_col,
            task_type=request.task_type or "auto",
            n_trials=request.n_trials or 10,
            existing_study_name=request.existing_study_name
        )
        results["target_column"] = target_col
        print("✅ Training completed successfully")
        return results
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Training Error: {str(e)}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.get("/api/download-model/{run_id}")
async def download_model(run_id: str):
    try:
        # Look for our directly saved model file (simple and reliable!)
        model_path = os.path.join(settings.UPLOAD_DIR, f"model_{run_id}.pkl")
        
        if os.path.exists(model_path):
            print(f"✅ Found model at: {model_path}")
            return FileResponse(
                path=model_path,
                filename=f"pulse_automl_model.pkl",
                media_type="application/octet-stream"
            )
        
        # Fallback: Try MLflow path if direct save failed
        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id
        artifact_uri = run.info.artifact_uri
        
        if artifact_uri.startswith("file://"):
            artifact_path = artifact_uri.replace("file://", "")
            if os.name == 'nt' and artifact_path.startswith("/"):
                artifact_path = artifact_path[1:]
        else:
            artifact_path = os.path.join(settings.MLFLOW_ARTIFACT_ROOT, experiment_id, run_id, "artifacts")

        # Try multiple possible model locations in MLflow artifacts
        possible_paths = [
            os.path.join(artifact_path, "model", "model.pkl"),
            os.path.join(artifact_path, "model.pkl"),
            os.path.join(artifact_path, "model")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    return FileResponse(
                        path=path,
                        filename=f"model_{run_id}.pkl",
                        media_type="application/octet-stream"
                    )
                elif os.path.isdir(path):
                    # Look for model files inside directory
                    for file in os.listdir(path):
                        if file.endswith('.pkl') or file.endswith('.joblib') or file.endswith('.pickle'):
                            full_path = os.path.join(path, file)
                            return FileResponse(
                                path=full_path,
                                filename=f"model_{run_id}.pkl",
                                media_type="application/octet-stream"
                            )
                
        raise HTTPException(status_code=404, detail=f"Model artifact not found at any expected location")
    except Exception as e:
        import traceback
        print(f"❌ Download error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")
