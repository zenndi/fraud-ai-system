from src.data_loader import DataLoader
from src.visualization import Visualizer
from src.preprocessing import Preprocessor
from src.model import FraudDetectionModel
from src.trainer import Trainer
from src.evaluator import Evaluator


def main():
    """Data Loader kısmı"""
    loader = DataLoader("data/creditcard.csv")
    df = loader.load_data() # Veri setini yükle
    loader.basic_info() #veri setine dair bilgiler

    """Visualization kısmı- EDA"""
    visualizer = Visualizer(df)
    visualizer.plot_class_distribution()
    visualizer.plot_amount_distribution()
    visualizer.plot_time_distribution()
    visualizer.plot_fraud_vs_normal_amount()
    visualizer.plot_correlation_heatmap()

    """Preprocessing kısmı"""
    preprocessor = Preprocessor()
    X, y = preprocessor.split_features_and_target(df)
    #önce ham traixn test split
    X_train_raw , X_test_raw, y_train, y_test = preprocessor.split_data(X, y) #gerçek amount değerlerini verecek

    #kopyaları scale edelim şimdi de
    X_train, X_test = preprocessor.scale_data(X_train_raw, X_test_raw) #modele verilecek scale edilmiş veri

   # X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
   # X_train, X_test = preprocessor.scale_data(X_train, X_test)

    X_under, y_under = preprocessor.apply_undersampling(X_train, y_train)
    X_over, y_over = preprocessor.apply_oversampling(X_train, y_train)

    print(f"\nOriginal Train Distribution:\n {y_train.value_counts()}")
    print(f"\nUndersampled Train Distribution:\n {y_under.value_counts()}")
    print(f"\nOversampled Train Distribution:\n {y_over.value_counts()}")


    """ Model Building kısmı"""
    model_builder = FraudDetectionModel(input_dim = X_over.shape[1]) #oversampling seçtik çünkü veri kaybı olmuyor
    model = model_builder.build_model()
    print("\nModel Summary:")
    model.summary()

    """ Model Training kısmı"""
    trainer = Trainer(model)
    history = trainer.train(X_over, y_over, X_test, y_test, epochs=30, batch_size=2048)
    visualizer.plot_training_history(history)
    print("\nModel training completed successfully.")


    """ Evaluation Kısmı"""
    evalutor = Evaluator(model)
    evalutor.evaluate_model(X_test, y_test)
    y_prob, y_pred = evalutor.get_predictions(X_test, threshold=0.9)
    evalutor.print_classification_metrics(y_test, y_pred)
    #evalutor.confusion_matrix_plt(y_test, y_pred, title="Confusion Matrix - Default Threshold")
    evalutor.roc_curve_plt(y_test, y_prob)
    best_threshold = evalutor.threshold_tuning(y_test, y_prob) #threshold analizi
    y_pred_best = (y_prob >= best_threshold).astype(int) #en iyi threshold ile yeni tahmin
    evalutor.threshold_metrics_plot(y_test, y_prob)
    evalutor.risk_score_distribution_plot(y_test, y_prob)
    print(f"\n Metrics with Best Threshold ({best_threshold:.2f})\n")
    evalutor.print_classification_metrics(y_test, y_pred_best)
    evalutor.confusion_matrix_plt(y_test, y_pred_best, title=f"Confusion Matrix - Best Threshold ({best_threshold:.2f})")

    #Final output risk score
    s_transaction  = X_test.iloc[[0]] 
    s_transaction_raw = X_test_raw.iloc[0]
    risk_score = evalutor.predict_risk_score(s_transaction)
    print(f"\nFINAL OUTPUT")
    print(f"Transaction Amount: ${s_transaction_raw['Amount']:.2f}")
    print(f"Risk Score: {risk_score:.4f}")

    if risk_score >= best_threshold:
        print("⚠ Fraud Detected (High Risk)")
    else:
        print("✅ Normal Transaction (Low Risk)")




#Loss curve:
 #Eğer training loss düşerken val loss düşüyorsa idealdir
 #Eğer training loss düşerken val loss artıyorsa overfitting vardır.

#Accuracy curve:
#Eğer train accuracy artarken ve val_accuracy artıyorsa idealdir ama fraud detecitonda accuracy bazen yanıltıcı olabilir çünkü dengesiz verimiz var



if __name__ == "__main__":
    main()
