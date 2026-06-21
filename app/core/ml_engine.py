"""
Pulse AutoML Platform
Copyright © 2026 TuViZa. All rights reserved.
Designed & Developed by Tushar Vijay Zaware
"""

import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional, Union
from app.core.config import settings

# Set MLflow tracking URI
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

class MLEngine:
    def __init__(self):
        self.preprocessor = None
        self.label_encoders = {}
        self.selected_features = []
        self.feature_names = []

    def profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Profile the data and suggest task type and target."""
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Find the best target candidate (prioritize non-null, non-unique columns)
        target_candidates = []
        for col in columns:
            null_pct = df[col].isnull().mean()
            unique_ratio = df[col].nunique() / len(df)
            if null_pct < 0.3 and unique_ratio < 0.9:
                target_candidates.append((col, -null_pct, unique_ratio))
        
        target_candidates.sort(key=lambda x: (x[1], x[2]))
        target_suggestion = target_candidates[0][0] if target_candidates else columns[-1]
        
        # Task type detection
        target_series = df[target_suggestion]
        unique_vals = target_series.nunique()
        
        if pd.api.types.is_numeric_dtype(target_series):
            if unique_vals < 15:
                task_type = "classification"
            else:
                task_type = "regression"
        else:
            task_type = "classification"
            
        return {
            "columns": columns,
            "dtypes": dtypes,
            "suggested_target": target_suggestion,
            "suggested_task": task_type,
            "sample_data": df.head(5).to_dict(orient='records')
        }

    def auto_detect_target(self, df: pd.DataFrame) -> str:
        """Premium feature: Auto-detect the best target column."""
        return self.profile_data(df)["suggested_target"]

    def preprocess_data(self, df: pd.DataFrame, target_col: str, task_type: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """SUPER POWERFUL PREPROCESSING - MAXIMIZE PERFORMANCE!"""
        # Drop rows with missing target
        df = df.dropna(subset=[target_col]).copy()
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Separate features by type
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Encode target if needed
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders[target_col] = le
        
        # Create powerful preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        num_features = numeric_features
        if categorical_features:
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features).tolist()
        else:
            cat_features = []
        self.feature_names = num_features + cat_features
        
        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X_processed, columns=self.feature_names)
        
        # Select ALL features (no aggressive selection!)
        self.selected_features = self.feature_names
        
        return X_df, y, self.feature_names

    def run_automl(self, df: pd.DataFrame, target_col: str, task_type: str, n_trials: int = 10, existing_study_name: Optional[str] = None) -> Dict[str, Any]:
        """Run Optuna study with MLflow tracking - GUARANTEED HIGH PERFORMANCE!"""
        if task_type == "auto":
            task_type = self.profile_data(df)["suggested_task"]

        X, y, feature_names = self.preprocess_data(df, target_col, task_type)
        
        # Use a larger validation split for better generalization
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if task_type == "classification" else None)

        def objective(trial):
            with mlflow.start_run(nested=True):
                if task_type == "classification":
                    model_name = trial.suggest_categorical("model", ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "GradientBoosting"])
                    
                    if model_name == "RandomForest":
                        n_estimators = trial.suggest_int("rf_n_estimators", 200, 1500)
                        max_depth = trial.suggest_int("rf_max_depth", 10, 100)
                        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
                        model = RandomForestClassifier(
                            n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            min_samples_split=min_samples_split,
                            random_state=42,
                            n_jobs=-1,
                            criterion='entropy'
                        )
                    elif model_name == "ExtraTrees":
                        n_estimators = trial.suggest_int("et_n_estimators", 200, 1500)
                        max_depth = trial.suggest_int("et_max_depth", 10, 100)
                        model = ExtraTreesClassifier(
                            n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            random_state=42,
                            n_jobs=-1,
                            criterion='entropy'
                        )
                    elif model_name == "LightGBM":
                        n_estimators = trial.suggest_int("lgb_n_estimators", 200, 1500)
                        learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True)
                        num_leaves = trial.suggest_int("lgb_num_leaves", 20, 250)
                        model = LGBMClassifier(
                            n_estimators=n_estimators, 
                            learning_rate=learning_rate, 
                            num_leaves=num_leaves, 
                            random_state=42, 
                            verbose=-1,
                            n_jobs=-1,
                            objective='multiclass' if len(np.unique(y)) > 2 else 'binary'
                        )
                    elif model_name == "GradientBoosting":
                        n_estimators = trial.suggest_int("gb_n_estimators", 200, 1500)
                        learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.2, log=True)
                        max_depth = trial.suggest_int("gb_max_depth", 5, 20)
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators, 
                            learning_rate=learning_rate, 
                            max_depth=max_depth, 
                            random_state=42
                        )
                    else: # XGBoost
                        n_estimators = trial.suggest_int("xgb_n_estimators", 200, 1500)
                        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
                        max_depth = trial.suggest_int("xgb_max_depth", 5, 20)
                        model = XGBClassifier(
                            n_estimators=n_estimators, 
                            learning_rate=learning_rate, 
                            max_depth=max_depth, 
                            random_state=42,
                            n_jobs=-1,
                            objective='multi:softmax' if len(np.unique(y)) > 2 else 'binary:logistic',
                            eval_metric='mlogloss' if len(np.unique(y)) > 2 else 'logloss'
                        )
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    
                    acc = accuracy_score(y_val, preds)
                    f1 = f1_score(y_val, preds, average='weighted')
                    prec = precision_score(y_val, preds, average='weighted', zero_division=0)
                    rec = recall_score(y_val, preds, average='weighted', zero_division=0)
                    
                    mlflow.log_params(trial.params)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    
                    trial.set_user_attr("metrics", {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec})
                    return acc

                else: # regression
                    model_name = trial.suggest_categorical("model", ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "GradientBoosting"])
                    
                    if model_name == "RandomForest":
                        n_estimators = trial.suggest_int("rf_n_estimators", 200, 1500)
                        max_depth = trial.suggest_int("rf_max_depth", 10, 100)
                        model = RandomForestRegressor(
                            n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            random_state=42,
                            n_jobs=-1
                        )
                    elif model_name == "ExtraTrees":
                        n_estimators = trial.suggest_int("et_n_estimators", 200, 1500)
                        max_depth = trial.suggest_int("et_max_depth", 10, 100)
                        model = ExtraTreesRegressor(
                            n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            random_state=42,
                            n_jobs=-1
                        )
                    elif model_name == "LightGBM":
                        n_estimators = trial.suggest_int("lgb_n_estimators", 200, 1500)
                        learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True)
                        num_leaves = trial.suggest_int("lgb_num_leaves", 20, 250)
                        model = LGBMRegressor(
                            n_estimators=n_estimators, 
                            learning_rate=learning_rate, 
                            num_leaves=num_leaves, 
                            random_state=42, 
                            verbose=-1,
                            n_jobs=-1
                        )
                    elif model_name == "GradientBoosting":
                        n_estimators = trial.suggest_int("gb_n_estimators", 200, 1500)
                        learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.2, log=True)
                        max_depth = trial.suggest_int("gb_max_depth", 5, 20)
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators, 
                            learning_rate=learning_rate, 
                            max_depth=max_depth, 
                            random_state=42
                        )
                    else: # XGBoost
                        n_estimators = trial.suggest_int("xgb_n_estimators", 200, 1500)
                        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
                        max_depth = trial.suggest_int("xgb_max_depth", 5, 20)
                        model = XGBRegressor(
                            n_estimators=n_estimators, 
                            learning_rate=learning_rate, 
                            max_depth=max_depth, 
                            random_state=42,
                            n_jobs=-1
                        )
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    
                    r2 = r2_score(y_val, preds)
                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    mae = mean_absolute_error(y_val, preds)
                    
                    mlflow.log_params(trial.params)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("rmse", rmse)
                    
                    trial.set_user_attr("metrics", {"r2": r2, "rmse": rmse, "mae": mae})
                    return r2

        # Create or Load MLflow experiment
        experiment_name = existing_study_name or f"AutoML_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_experiment(experiment_name)

        # EXTREMELY AGGRESSIVE OPTIMIZATION FOR MAX PERFORMANCE!
        study = optuna.create_study(
            study_name=experiment_name,
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=15,
                n_ei_candidates=80,
                seed=42,
                multivariate=True,
                constant_liar=True
            ),
            load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials)

        # Process trials to compute cumulative best scores for frontend visualization
        cumulative_best = []
        current_best = -float('inf')
        for t in study.trials:
            if t.value is not None:
                if t.value > current_best:
                    current_best = t.value
                cumulative_best.append(current_best)
            else:
                cumulative_best.append(current_best if current_best != -float('inf') else None)

        # Train best model on ALL data
        best_params = study.best_params.copy()
        best_model_name = best_params.pop("model")
        
        if task_type == "classification":
            if best_model_name == "RandomForest":
                final_model = RandomForestClassifier(
                    n_estimators=best_params["rf_n_estimators"], 
                    max_depth=best_params["rf_max_depth"],
                    min_samples_split=best_params["rf_min_samples_split"],
                    random_state=42,
                    n_jobs=-1,
                    criterion='entropy'
                )
            elif best_model_name == "ExtraTrees":
                final_model = ExtraTreesClassifier(
                    n_estimators=best_params["et_n_estimators"], 
                    max_depth=best_params["et_max_depth"], 
                    random_state=42,
                    n_jobs=-1,
                    criterion='entropy'
                )
            elif best_model_name == "LightGBM":
                final_model = LGBMClassifier(
                    n_estimators=best_params["lgb_n_estimators"], 
                    learning_rate=best_params["lgb_learning_rate"], 
                    num_leaves=best_params["lgb_num_leaves"], 
                    random_state=42, 
                    verbose=-1,
                    n_jobs=-1,
                    objective='multiclass' if len(np.unique(y)) > 2 else 'binary'
                )
            elif best_model_name == "GradientBoosting":
                final_model = GradientBoostingClassifier(
                    n_estimators=best_params["gb_n_estimators"], 
                    learning_rate=best_params["gb_learning_rate"], 
                    max_depth=best_params["gb_max_depth"], 
                    random_state=42
                )
            else:
                final_model = XGBClassifier(
                    n_estimators=best_params["xgb_n_estimators"], 
                    learning_rate=best_params["xgb_learning_rate"], 
                    max_depth=best_params["xgb_max_depth"], 
                    random_state=42,
                    n_jobs=-1,
                    objective='multi:softmax' if len(np.unique(y)) > 2 else 'binary:logistic',
                    eval_metric='mlogloss' if len(np.unique(y)) > 2 else 'logloss'
                )
        else:
            if best_model_name == "RandomForest":
                final_model = RandomForestRegressor(
                    n_estimators=best_params["rf_n_estimators"], 
                    max_depth=best_params["rf_max_depth"], 
                    random_state=42,
                    n_jobs=-1
                )
            elif best_model_name == "ExtraTrees":
                final_model = ExtraTreesRegressor(
                    n_estimators=best_params["et_n_estimators"], 
                    max_depth=best_params["et_max_depth"], 
                    random_state=42,
                    n_jobs=-1
                )
            elif best_model_name == "LightGBM":
                final_model = LGBMRegressor(
                    n_estimators=best_params["lgb_n_estimators"], 
                    learning_rate=best_params["lgb_learning_rate"], 
                    num_leaves=best_params["lgb_num_leaves"], 
                    random_state=42, 
                    verbose=-1,
                    n_jobs=-1
                )
            elif best_model_name == "GradientBoosting":
                final_model = GradientBoostingRegressor(
                    n_estimators=best_params["gb_n_estimators"], 
                    learning_rate=best_params["gb_learning_rate"], 
                    max_depth=best_params["gb_max_depth"], 
                    random_state=42
                )
            else:
                final_model = XGBRegressor(
                    n_estimators=best_params["xgb_n_estimators"], 
                    learning_rate=best_params["xgb_learning_rate"], 
                    max_depth=best_params["xgb_max_depth"], 
                    random_state=42,
                    n_jobs=-1
                )

        # Train final model on ALL data
        final_model.fit(X, y)

        # Prepare final best metrics
        final_best_metrics = study.best_trial.user_attrs.get("metrics", {})

        # Log final model
        with mlflow.start_run(run_name="Best_Model") as run:
            mlflow.log_params(best_params)
            if final_best_metrics:
                mlflow.log_metrics(final_best_metrics)
            
            # Save model with MLflow
            try:
                mlflow.sklearn.log_model(
                    final_model, 
                    "model"
                )
            except Exception:
                # Fallback if MLflow fails - just save directly
                pass
                
            # ALSO SAVE A STANDALONE PKL FILE DIRECTLY FOR EASY DOWNLOAD!
            import joblib
            local_model_path = os.path.join(settings.UPLOAD_DIR, f"model_{run.info.run_id}.pkl")
            joblib.dump(final_model, local_model_path)
            print(f"💾 Model saved to: {local_model_path}")
                
            run_id = run.info.run_id

        # Store the local path in the results for easy access
        final_best_metrics["_model_path"] = local_model_path

        # Prepare trial history
        trials_data = []
        for t in study.trials:
            trials_data.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "metrics": t.user_attrs.get("metrics", {})
            })

        return {
            "best_model_type": best_model_name,
            "best_params": best_params,
            "best_score": study.best_value,
            "best_metrics": final_best_metrics,
            "trials": trials_data,
            "cumulative_best": cumulative_best,
            "run_id": run_id,
            "task_type": task_type,
            "target_column": target_col,
            "study_name": experiment_name
        }
