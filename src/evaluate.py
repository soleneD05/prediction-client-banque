"""
Enhanced Model Evaluation Script
Evaluates the selected best model and provides comprehensive analysis
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from src.utils import load_params

def evaluate_best_model(params: dict):
    """
    Comprehensive evaluation of the selected best model
    """
    
    # 1. Load model and test data
    model_path = params["model"]["path"]
    model = joblib.load(model_path)
    
    X_test = pd.read_csv(params["data"]["X_test_path"])
    y_test = pd.read_csv(params["data"]["y_test_path"]).iloc[:, 0]
    
    # Load model selection results if available
    try:
        model_comparison = pd.read_csv("data/model_selection_results.csv")
        best_model_name = model_comparison.loc[model_comparison['Best_Score'].idxmax(), 'Model']
        print(f"🏆 Best selected model: {best_model_name}")
    except:
        best_model_name = "Selected Model"
        print(f"🤖 Evaluating trained model")
    
    print(f"📁 Model loaded from: {model_path}")
    print(f"📊 Test data: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"⚖️ Test set churn rate: {y_test.mean():.2%}\n")
    
    # 2. Make predictions
    print("🔮 Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 3. Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Calculate business metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    business_metrics = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    }
    
    # 4. Print detailed results
    print("🎯 DETAILED EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {best_model_name}")
    print("="*60)
    
    print("\n📊 PERFORMANCE METRICS:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\n💼 BUSINESS IMPACT:")
    print(f"  ✅ Correctly identified churners: {tp} / {tp + fn} ({tp/(tp + fn)*100:.1f}%)")
    print(f"  ❌ Missed churners: {fn} (cost: potential revenue loss)")
    print(f"  ⚠️ False alarms: {fp} (cost: unnecessary retention efforts)")
    print(f"  ✅ Correctly identified loyal customers: {tn}")
    
    # 5. Create enhanced visualizations
    print("\n📈 Generating comprehensive visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Churn Prediction - {best_model_name} Evaluation', fontsize=16, fontweight='bold')
    
    # 5.1 Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 5.2 ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=3, 
                    label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 5.3 Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[0, 2].plot(recall_vals, precision_vals, color='darkgreen', lw=3,
                    label=f'PR (AP = {metrics["avg_precision"]:.3f})')
    axes[0, 2].axhline(y=y_test.mean(), color='red', linestyle='--', alpha=0.7,
                        label=f'Baseline = {y_test.mean():.3f}')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 5.4 Prediction Distribution
    axes[1, 0].hist(y_pred_proba[y_test == 0], bins=25, alpha=0.7, 
                    label='Not Churned', color='lightblue', density=True)
    axes[1, 0].hist(y_pred_proba[y_test == 1], bins=25, alpha=0.7, 
                    label='Churned', color='lightcoral', density=True)
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Threshold = 0.5')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Predicted Probabilities')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5.5 Performance Metrics Bar Chart
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = axes[1, 1].bar(metric_names, metric_values, color=colors[:len(metric_names)])
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 5.6 Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)  # Top 10
        
        axes[1, 2].barh(feature_importance['feature'], feature_importance['importance'], 
                        color='skyblue')
        axes[1, 2].set_title('Top 10 Feature Importance')
        axes[1, 2].set_xlabel('Importance')
        axes[1, 2].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 2].text(0.5, 0.5, 'Feature Importance\nNot Available\nfor this model', 
                        ha='center', va='center', transform=axes[1, 2].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # 6. Save all results
    all_metrics = {**metrics, **business_metrics}
    metrics_df = pd.DataFrame([all_metrics])
    metrics_path = params["evaluation"]["metrics_path"]
    metrics_df.to_csv(metrics_path, index=False)
    print(f"💾 Metrics saved to: {metrics_path}")
    
    # Save plots
    plots_path = params["evaluation"]["plots_path"]
    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"📊 Evaluation plots saved to: {plots_path}")
    
    # 7. Performance assessment
    thresholds = params["evaluation"]["thresholds"]
    f1 = metrics['f1_score']
    
    print("\n🎯 FINAL PERFORMANCE ASSESSMENT")
    print("="*60)
    print(f"📊 Test set size: {len(y_test):,} samples")
    print(f"🤖 Model: {best_model_name}")
    print(f"🎯 Overall accuracy: {metrics['accuracy']:.1%}")
    
    if f1 >= thresholds['excellent_f1']:
        status = "EXCELLENT ⭐⭐⭐"
        color = "🟢"
    elif f1 >= thresholds['good_f1_score']:
        status = "GOOD ⭐⭐"
        color = "🟡"
    elif f1 >= thresholds['moderate_f1_score']:
        status = "MODERATE ⭐"
        color = "🟠"
    else:
        status = "NEEDS IMPROVEMENT"
        color = "🔴"
    
    print(f"{color} Performance status: {status} (F1-Score: {f1:.3f})")
    
    # Business insights
    print(f"\n💡 BUSINESS INSIGHTS:")
    print(f"   • Model catches {tp}/{tp+fn} churning customers ({tp/(tp+fn)*100:.1f}%)")
    print(f"   • {fp} false alarms (unnecessary retention costs)")
    print(f"   • {fn} missed churners (potential revenue loss)")
    
    # Recommendations
    if f1 < 0.7:
        print(f"\n📋 RECOMMENDATIONS FOR IMPROVEMENT:")
        if metrics['precision'] < 0.7:
            print("   • High false positives → Consider adjusting prediction threshold")
        if metrics['recall'] < 0.7:
            print("   • Missing churners → Try ensemble methods or feature engineering")
        print("   • Consider collecting more data or additional features")
    
    print(f"\n🎉 Evaluation completed successfully!")
    
    plt.show()
    return metrics, best_model_name

if __name__ == "__main__":
    params = load_params()
    evaluate_best_model(params)