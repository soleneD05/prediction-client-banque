"""
Simplified Model Training Script - Random Forest Only
Uses optimized hyperparameters for faster training
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from src.utils import load_params
import warnings
warnings.filterwarnings('ignore')

def train_random_forest_model(params: dict):
    """
    Simplified training function that uses Random Forest with optimized parameters
    """
    
    # 1. Load preprocessed training data
    X_train = pd.read_csv(params["data"]["X_train_path"])
    y_train = pd.read_csv(params["data"]["y_train_path"]).iloc[:, 0]
    X_test = pd.read_csv(params["data"]["X_test_path"])
    y_test = pd.read_csv(params["data"]["y_test_path"]).iloc[:, 0]
    
    print("🚀 SIMPLIFIED RANDOM FOREST TRAINING")
    print("="*50)
    print(f"Training data: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test data: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"Target distribution: {y_train.value_counts().to_dict()}")
    print(f"Churn rate: {y_train.mean():.2%}\n")
    
    # 2. Create Random Forest with optimized hyperparameters
    print("🌲 Creating Random Forest with optimized parameters...")
    
    # Using the best parameters you found
    best_rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1  # Use all available cores for faster training
    }
    
    model = RandomForestClassifier(**best_rf_params)
    
    print("Parameters used:")
    for key, value in best_rf_params.items():
        print(f"  {key}: {value}")
    
    # 3. Cross-validation to validate performance
    print(f"\n🔄 Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"CV Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 4. Train the final model on full training set
    print(f"\n🎯 Training final model on full training set...")
    model.fit(X_train, y_train)
    
    # 5. Evaluate on training and test sets
    print(f"\n📊 EVALUATION RESULTS")
    print("="*50)
    
    # Training set evaluation
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    # Test set evaluation
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    print("TRAINING SET:")
    for metric, value in train_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nTEST SET:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\n📋 DETAILED CLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # 6. Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save feature importance
        feature_importance.to_csv("data/feature_importance.csv", index=False)
        print("Feature importance saved to: data/feature_importance.csv")
    
    # 7. Save the trained model
    model_path = params["model"]["path"]
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    
    # 8. Save training results
    results_summary = {
        'model_name': 'Random Forest',
        'cv_mean_f1': cv_scores.mean(),
        'cv_std_f1': cv_scores.std(),
        'train_accuracy': train_metrics['accuracy'],
        'train_f1': train_metrics['f1_score'],
        'train_roc_auc': train_metrics['roc_auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1_score'],
        'test_roc_auc': test_metrics['roc_auc'],
        'hyperparameters': str(best_rf_params)
    }
    
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv("data/training_results.csv", index=False)
    print("Training results saved to: data/training_results.csv")
    
    # 9. Performance assessment
    print(f"\n🏆 FINAL PERFORMANCE ASSESSMENT")
    print("="*50)
    test_f1 = test_metrics['f1_score']
    
    if test_f1 >= 0.80:
        status = "🥇 EXCELLENT"
    elif test_f1 >= 0.70:
        status = "🥈 GOOD"
    elif test_f1 >= 0.60:
        status = "🥉 MODERATE"
    else:
        status = "⚠️ NEEDS IMPROVEMENT"
    
    print(f"Model: Random Forest")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Performance Status: {status}")
    
    # Business insights
    print(f"\n💡 BUSINESS INSIGHTS:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"  • Model correctly identifies {tp}/{tp+fn} churning customers ({tp/(tp+fn)*100:.1f}%)")
    print(f"  • {fp} false alarms (unnecessary retention costs)")
    print(f"  • {fn} missed churners (potential revenue loss)")
    print(f"  • Overall accuracy: {test_metrics['accuracy']:.1%}")
    
    print(f"\n🎉 Training completed successfully!")
    print("✅ Ready for evaluation and API deployment!")
    
    return model, results_summary

if __name__ == "__main__":
    params = load_params()
    model, results = train_random_forest_model(params)