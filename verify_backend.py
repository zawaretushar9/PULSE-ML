import pandas as pd
import numpy as np
from app.core.ml_engine import MLEngine
import os

def test_automl_pipeline():
    print("🚀 Starting AutoML Pipeline Verification...")
    
    # 1. Create dummy data
    data = {
        'age': np.random.randint(18, 80, 100),
        'income': np.random.randint(20000, 150000, 100),
        'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], 100),
        'target': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(data)
    
    engine = MLEngine()
    
    # 2. Test Profiling
    print("📊 Testing Data Profiling...")
    profile = engine.profile_data(df)
    assert 'suggested_target' in profile
    assert profile['suggested_task'] == 'classification'
    print("✅ Profiling Successful")
    
    # 3. Test Training
    print("🧠 Testing AutoML Training (10 trials)...")
    results = engine.run_automl(df, target_col='target', task_type='classification', n_trials=5)
    
    assert 'best_model_type' in results
    assert 'run_id' in results
    assert len(results['trials']) == 5
    print(f"✅ Training Successful. Best Model: {results['best_model_type']}")
    print(f"📈 Best Score: {results['best_score']:.4f}")
    
    # 4. Check Features
    print(f"🔍 Selected Features: {engine.selected_features}")
    
    print("\n✨ All Backend Verifications Passed!")

if __name__ == "__main__":
    # Ensure we are in the right directory to import app
    import sys
    sys.path.append(os.getcwd())
    test_automl_pipeline()
