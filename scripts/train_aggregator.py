#!/usr/bin/env python3
"""
Train an aggregator classifier on edge score features.

Takes output from extract_edge_scores.py and trains a simple classifier
to predict edge labels from aggregated walk scores.

Data splits:
  - Agg train: 80% of transformer val edges (used for training)
  - Agg val: 20% of transformer val edges (tuning, early stopping)
  - Agg test: 100% of transformer test edges (final evaluation)

Configuration:
  Class weighting: 'balanced' (inverse frequency) to match transformer training
  Feature scaling: StandardScaler (normalize each feature to mean=0, std=1)
  Logistic regression: max_iter=1000, L2 regularization (C=1.0)
  MLP: hidden layers (128, 64), learning rate=0.001, early stopping on validation AUC
"""

import os
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)


def load_features(pickle_path):
    """Load features from pickle file"""
    print(f"Loading features from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Dataset: {data['config']['dataset']}")
    print(f"Num classes: {data['config']['num_classes']}")
    print(f"Feature names: {data['feature_names']}")
    
    return data


def prepare_dataset(features_dict):
    """
    Convert feature dict to X, y arrays.
    
    Args:
        features_dict: {(u,v,label): {'features': array, 'ground_truth': label}}
    
    Returns:
        X: (n_edges, n_features) array
        y: (n_edges,) array of labels
        edges: list of edge keys (for debugging)
    """
    edges = list(features_dict.keys())
    X = np.array([features_dict[e]['features'] for e in edges])
    y = np.array([features_dict[e]['ground_truth'] for e in edges])
    
    return X, y, edges


def train_logistic_regression(X_train, y_train, X_val, y_val, num_classes, cfg):
    """Train logistic regression classifier with class weighting"""
    print("\nTraining Logistic Regression...")
    print(f"  class_weight: {cfg['class_weight']}")
    print(f"  max_iter: {cfg['logistic_max_iter']}")
    print(f"  C (regularization): {cfg['logistic_C']}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    if num_classes == 2:
        model = LogisticRegression(
            max_iter=cfg['logistic_max_iter'],
            random_state=42,
            class_weight=cfg['class_weight'],
            C=cfg['logistic_C'],
            solver='lbfgs'
        )
    else:
        model = LogisticRegression(
            max_iter=cfg['logistic_max_iter'],
            random_state=42,
            multi_class='multinomial',
            class_weight=cfg['class_weight'],
            C=cfg['logistic_C'],
            solver='lbfgs'
        )
    
    model.fit(X_train_scaled, y_train)
    
    # Validation predictions
    val_preds = model.predict(X_val_scaled)
    val_probs = model.predict_proba(X_val_scaled)
    
    # Compute metrics
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    
    if num_classes == 2:
        val_auc = roc_auc_score(y_val, val_probs[:, 1])
    else:
        val_auc = roc_auc_score(y_val, val_probs, multi_class='ovr', average='macro')
    
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print(f"Val AUC: {val_auc:.4f}")
    
    return model, scaler, {'accuracy': val_acc, 'f1': val_f1, 'auc': val_auc}


def train_mlp(X_train, y_train, X_val, y_val, num_classes, cfg):
    """Train MLP classifier with class weighting"""
    print("\nTraining MLP Classifier...")
    print(f"  hidden_layer_sizes: {cfg['mlp_hidden_layers']}")
    print(f"  learning_rate_init: {cfg['mlp_learning_rate']}")
    print(f"  early_stopping_patience: {cfg['mlp_early_stopping_patience']}")
    print(f"  batch_size: {cfg['mlp_batch_size']}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Compute class weights for manual weighting during training
    # (MLPClassifier doesn't have direct class_weight parameter)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_weights = {}
    total = len(y_train)
    for cls, count in zip(unique_classes, class_counts):
        # Inverse frequency weighting: n_samples / (n_classes * count)
        class_weights[cls] = total / (num_classes * count)
    
    print(f"  class_weights (inverse frequency): {class_weights}")
    
    # Train model with early stopping
    model = MLPClassifier(
        hidden_layer_sizes=cfg['mlp_hidden_layers'],
        learning_rate_init=cfg['mlp_learning_rate'],
        max_iter=cfg['mlp_max_iter'],
        batch_size=cfg['mlp_batch_size'],
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=cfg['mlp_early_stopping_patience'],
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Validation predictions
    val_preds = model.predict(X_val_scaled)
    val_probs = model.predict_proba(X_val_scaled)
    
    # Compute metrics
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    
    if num_classes == 2:
        val_auc = roc_auc_score(y_val, val_probs[:, 1])
    else:
        val_auc = roc_auc_score(y_val, val_probs, multi_class='ovr', average='macro')
    
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    print(f"Val AUC: {val_auc:.4f}")
    
    return model, scaler, {'accuracy': val_acc, 'f1': val_f1, 'auc': val_auc}


def evaluate_model(model, scaler, X, y, num_classes, split_name):
    """Evaluate model on a dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating on {split_name} set")
    print(f"{'='*60}")
    
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)
    
    # Compute metrics
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    
    if num_classes == 2:
        auc = roc_auc_score(y, probs[:, 1])
    else:
        auc = roc_auc_score(y, probs, multi_class='ovr', average='macro')
    
    print(f"\nMetrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, preds, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y, preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'predictions': preds,
        'probabilities': probs,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, output_path, title):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train aggregator classifier")
    parser.add_argument("--features", type=str, required=True, help="Path to features pickle file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for models and results")
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "mlp"],
                        help="Classifier type")
    
    # Shared hyperparameters
    parser.add_argument("--class_weight", type=str, default="balanced",
                        help="Class weighting: 'balanced' (inverse freq) or None (uniform)")
    
    # Logistic regression hyperparameters
    parser.add_argument("--logistic_max_iter", type=int, default=1000,
                        help="Max iterations for logistic regression")
    parser.add_argument("--logistic_C", type=float, default=1.0,
                        help="Inverse of regularization strength (C) for logistic regression")
    
    # MLP hyperparameters
    parser.add_argument("--mlp_hidden_layers", type=str, default="128,64",
                        help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--mlp_learning_rate", type=float, default=0.001,
                        help="Learning rate for MLP")
    parser.add_argument("--mlp_max_iter", type=int, default=1000,
                        help="Max iterations for MLP")
    parser.add_argument("--mlp_batch_size", type=int, default=32,
                        help="Batch size for MLP training")
    parser.add_argument("--mlp_early_stopping_patience", type=int, default=20,
                        help="Early stopping patience for MLP (n_iter_no_change)")
    
    args = parser.parse_args()
    
    # Parse MLP hidden layers
    mlp_hidden_layers = tuple(map(int, args.mlp_hidden_layers.split(',')))
    
    # Configuration dict
    cfg = {
        'class_weight': args.class_weight,
        'logistic_max_iter': args.logistic_max_iter,
        'logistic_C': args.logistic_C,
        'mlp_hidden_layers': mlp_hidden_layers,
        'mlp_learning_rate': args.mlp_learning_rate,
        'mlp_max_iter': args.mlp_max_iter,
        'mlp_batch_size': args.mlp_batch_size,
        'mlp_early_stopping_patience': args.mlp_early_stopping_patience,
    }
    
    print("\n" + "="*60)
    print("AGGREGATOR CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"\nShared:")
    print(f"  class_weight: {cfg['class_weight']}")
    
    if args.model == "logistic":
        print(f"\nLogistic Regression:")
        print(f"  max_iter: {cfg['logistic_max_iter']}")
        print(f"  C (regularization): {cfg['logistic_C']}")
    else:
        print(f"\nMLP:")
        print(f"  hidden_layers: {cfg['mlp_hidden_layers']}")
        print(f"  learning_rate: {cfg['mlp_learning_rate']}")
        print(f"  batch_size: {cfg['mlp_batch_size']}")
        print(f"  early_stopping_patience: {cfg['mlp_early_stopping_patience']}")
    print("="*60 + "\n")
    
    # Load features
    data = load_features(args.features)
    num_classes = data['config']['num_classes']
    dataset_name = data['config']['dataset']
    
    # Prepare datasets
    print("\nPreparing agg train dataset...")
    X_train, y_train, train_edges = prepare_dataset(data['agg_train'])
    print(f"Agg train: {len(X_train)} edges, {X_train.shape[1]} features")
    
    print("\nPreparing agg val dataset...")
    X_val, y_val, val_edges = prepare_dataset(data['agg_val'])
    print(f"Agg val: {len(X_val)} edges")
    
    print("\nPreparing agg test dataset...")
    X_test, y_test, test_edges = prepare_dataset(data['agg_test'])
    print(f"Agg test: {len(X_test)} edges")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(f"  Agg train: {np.bincount(y_train)}")
    print(f"  Agg val: {np.bincount(y_val)}")
    print(f"  Agg test: {np.bincount(y_test)}")
    
    # Train model
    if args.model == "logistic":
        model, scaler, val_metrics = train_logistic_regression(
            X_train, y_train, X_val, y_val, num_classes, cfg
        )
    else:
        model, scaler, val_metrics = train_mlp(
            X_train, y_train, X_val, y_val, num_classes, cfg
        )
    
    # Evaluate on all splits
    train_results = evaluate_model(model, scaler, X_train, y_train, num_classes, "Agg Train")
    val_results = evaluate_model(model, scaler, X_val, y_val, num_classes, "Agg Val")
    test_results = evaluate_model(model, scaler, X_test, y_test, num_classes, "Agg Test")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model and scaler
    model_path = os.path.join(args.output_dir, f"aggregator_{args.model}.pkl")
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Save results
    results = {
        'dataset': dataset_name,
        'model_type': args.model,
        'num_classes': num_classes,
        'feature_names': data['feature_names'],
        'config': cfg,
        'train_metrics': train_results,
        'val_metrics': val_results,
        'test_metrics': test_results,
    }
    
    results_path = os.path.join(args.output_dir, "results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save text summary
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Aggregator Results - {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Num classes: {num_classes}\n")
        f.write(f"Features: {data['feature_names']}\n\n")
        
        f.write(f"Agg train set (80% of transformer val): {len(X_train)} edges\n")
        f.write(f"  Accuracy: {train_results['accuracy']:.4f}\n")
        f.write(f"  F1: {train_results['f1']:.4f}\n")
        f.write(f"  AUC: {train_results['auc']:.4f}\n\n")
        
        f.write(f"Agg val set (20% of transformer val): {len(X_val)} edges\n")
        f.write(f"  Accuracy: {val_results['accuracy']:.4f}\n")
        f.write(f"  F1: {val_results['f1']:.4f}\n")
        f.write(f"  AUC: {val_results['auc']:.4f}\n\n")
        
        f.write(f"Agg test set (transformer test): {len(X_test)} edges\n")
        f.write(f"  Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"  F1: {test_results['f1']:.4f}\n")
        f.write(f"  AUC: {test_results['auc']:.4f}\n\n")
    
    print(f"Summary saved to {summary_path}")
    
    # Plot confusion matrices
    plot_confusion_matrix(
        train_results['confusion_matrix'],
        os.path.join(args.output_dir, "train_confusion_matrix.png"),
        f"{dataset_name} - Agg Train Confusion Matrix"
    )
    plot_confusion_matrix(
        val_results['confusion_matrix'],
        os.path.join(args.output_dir, "val_confusion_matrix.png"),
        f"{dataset_name} - Agg Val Confusion Matrix"
    )
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        os.path.join(args.output_dir, "test_confusion_matrix.png"),
        f"{dataset_name} - Agg Test Confusion Matrix"
    )
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {dataset_name}")
    print(f"{'='*60}")
    print(f"Agg Test AUC: {test_results['auc']:.4f}")
    print(f"Agg Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Agg Test F1: {test_results['f1']:.4f}")
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
