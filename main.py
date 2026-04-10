import yaml
import os
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_loader import DataLoader
from src.visualization import Visualizer
from src.preprocessing import Preprocessor
from src.model import FraudDetectionModel
from src.trainer import Trainer
from src.evaluator import Evaluator


def load_config(config_path="config.yaml"):
    """Config dosyasını yükle"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Ana çalıştırma fonksiyonu"""
    
    # Config yükle
    config = load_config()
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    eval_config = config.get('evaluation', {})
    
    print("=" * 60)
    print("FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==================== DATA LOADING ====================
    print("\n📊 STEP 1: Loading Data...")
    loader = DataLoader(data_config.get('path', 'data/creditcard.csv'))
    df = loader.load_data()
    loader.basic_info()
    
    # ==================== VISUALIZATION (EDA) ====================
    print("\n📈 STEP 2: Exploratory Data Analysis...")
    visualizer = Visualizer(df)
    visualizer.plot_class_distribution()
    visualizer.plot_amount_distribution()
    visualizer.plot_time_distribution()
    visualizer.plot_fraud_vs_normal_amount()
    visualizer.plot_correlation_heatmap()
    
    # ==================== PREPROCESSING ====================
    print("\n🔧 STEP 3: Preprocessing...")
    preprocessor = Preprocessor(
        test_size=data_config.get('test_size', 0.2),
        random_state=data_config.get('random_state', 42)
    )
    
    X, y = preprocessor.split_features_and_target(df)
    X_train_raw, X_test_raw, y_train, y_test = preprocessor.split_data(X, y)
    X_train, X_test = preprocessor.scale_data(X_train_raw, X_test_raw)
    
    # Sampling
    sampling_method = config.get('sampling', {}).get('method', 'oversample')
    if sampling_method == 'undersample':
        X_train_balanced, y_train_balanced = preprocessor.apply_undersampling(X_train, y_train)
    elif sampling_method == 'smote':
        X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = preprocessor.apply_oversampling(X_train, y_train)
    
    print(f"\nOriginal Train Distribution:\n{y_train.value_counts()}")
    print(f"\nBalanced Train Distribution ({sampling_method}):\n{y_train_balanced.value_counts()}")
    
    # ==================== MODEL BUILDING ====================
    print("\n🧠 STEP 4: Building Model...")
    model_builder = FraudDetectionModel(
        input_dim=X_train_balanced.shape[1],
        config_path="config.yaml"
    )
    model = model_builder.build_model()
    print("\nModel Summary:")
    model.summary()
    
    # ==================== TRAINING ====================
    print("\n🏋️ STEP 5: Training Model...")
    trainer = Trainer(model, config_path="config.yaml")
    
    # Class weights (opsiyonel)
    class_weights = None
    if config.get('training', {}).get('use_class_weights', False):
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"Using class weights: {class_weights}")
    
    history = trainer.train(
        X_train_balanced, y_train_balanced,
        X_test, y_test,
        class_weights=class_weights
    )
    visualizer.plot_training_history(history)
    print("\n✅ Model training completed successfully.")
    
    # ==================== EVALUATION ====================
    print("\n📋 STEP 6: Evaluating Model...")
    evaluator = Evaluator(model)
    
    # Temel değerlendirme
    test_loss, test_acc = evaluator.evaluate_model(X_test, y_test)
    
    # Threshold analizi
    y_prob, y_pred_default = evaluator.get_predictions(
        X_test, 
        threshold=eval_config.get('default_threshold', 0.5)
    )
    
    print("\n--- Default Threshold Results ---")
    evaluator.print_classification_metrics(y_test, y_pred_default)
    
    # ROC Curve
    evaluator.roc_curve_plt(y_test, y_prob)
    
    # Threshold tuning
    best_threshold = evaluator.threshold_tuning(y_test, y_prob)
    y_pred_best = (y_prob >= best_threshold).astype(int)
    
    print(f"\n--- Best Threshold Results ({best_threshold:.2f}) ---")
    evaluator.print_classification_metrics(y_test, y_pred_best)
    
    # Confusion Matrix
    evaluator.confusion_matrix_plt(
        y_test, y_pred_best, 
        title=f"Confusion Matrix - Best Threshold ({best_threshold:.2f})"
    )
    
    # Threshold metrics plot
    evaluator.threshold_metrics_plot(y_test, y_prob)
    
    # Risk score distribution
    evaluator.risk_score_distribution_plot(y_test, y_prob)
    
    # ==================== SAMPLE PREDICTION ====================
    print("\n🎯 STEP 7: Sample Prediction...")
    sample_transaction = X_test.iloc[[0]]
    sample_transaction_raw = X_test_raw.iloc[0]
    risk_score = evaluator.predict_risk_score(sample_transaction)
    
    print(f"\n{'='*40}")
    print(f"Transaction Amount: ${sample_transaction_raw['Amount']:.2f}")
    print(f"Risk Score: {risk_score:.4f}")
    
    if risk_score >= best_threshold:
        print("⚠️  FRAUD DETECTED (High Risk)")
    else:
        print("✅ Normal Transaction (Low Risk)")
    print(f"{'='*40}")
    
    # ==================== MODEL SAVING ====================
    print("\n💾 STEP 8: Saving Model...")
    save_dir = config.get('checkpoint', {}).get('save_dir', 'models/')
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(save_dir, f'fraud_model_{timestamp}.keras')
    
    # Config ile birlikte kaydet
    model.save(model_path)
    
    # Threshold ve config bilgilerini de kaydet
    metadata = {
        'model_path': model_path,
        'best_threshold': float(best_threshold),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'timestamp': timestamp,
        'config': config
    }
    
    import json
    metadata_path = os.path.join(save_dir, f'model_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    # ==================== DONE ====================
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
