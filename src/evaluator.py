import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from utils.metrics import calculate_classification_metrics, get_classification_report, confusion_matrix_val, calculate_roc_auc, find_best_threshold_by_f1

class Evaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate_model(self, X_test, y_test): #modeli test edeceğiz
        """Modelin test perofrmansı değerlendirilecek"""
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        return test_loss, test_acc
    
    def get_predictions(self, X_test, threshold=0.5):
        #threshold düşerse recall artar, precision düşebilir
        #threshold yükselirse precision artar, recall düşebilir
        y_prob = self.model.predict(X_test)
        y_prob = self.model.predict(X_test)
        y_pred = (y_prob > threshold).astype(int)

        return y_prob, y_pred
    def print_classification_metrics(self, y_test, y_pred):
        metrics = calculate_classification_metrics(y_test, y_pred)

        print("\nCLASSIFICATION METRICS")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

        print("\nCLASSIFICATION REPORT")
        print(get_classification_report(y_test,y_pred))

    def confusion_matrix_plt(self, y_test, y_pred, title="Confusion Matrix"):
        """
        	                Predicted Normal	           Predicted Fraud
      Actual Normal	       True Negative (TN)	          False Positive (FP)
      Actual Fraud	       False Negative (FN)	          True Positive (TP)
        
        """
        confusion_mat = confusion_matrix_val(y_test, y_pred)

        plt.figure(figsize=(9, 6))
        plt.imshow(confusion_mat, cmap='Blues', interpolation='nearest') #heatmap çizer
        plt.title(title)
        plt.colorbar()
        plt.xticks([0,1], ["Normal", "Fraud"]) #axis labellardır ve eksenleri isimlendirir
        plt.yticks([0,1], ["Normal", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        #grafiğin içine sayıları yazdırır yani tn, fp, fn, tp sayılarını yazdırır
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, confusion_mat[i, j], ha='center', va='center', color='white')
        plt.tight_layout()
        plt.savefig("assets/model/confusion_matrix.png", dpi=300)
        plt.show()

    #ROC = Receiver Operating Characteristic: modelin fraud ile normal işlemi ayırma gücünü ölçer
    #AUC = Area Under the Curve: ROC eğrisi altındaki iki boyutlu alanı ifade eden bir metrikdir. 0 ile 1 arasında değer alır
    #ve 1'e yakın olması modelin pozitif ve negatif sınıfları ayırmada çok başarılı, 0.5 ise rastgele tahmin kadar başarısız olduğunu gösterir
    #0.7 ortadır, 0.8 iyi, 0.9 ve üstü çok iyi
    def roc_curve_plt(self, y_test, y_prob):
        auc_score = calculate_roc_auc(y_test, y_prob)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob) #thresholdu almadık grafikte kullanmayacağız
        #false_positive_rate: FP / (FP + TN) yani normal işlemleri yanlış fraud tahmin etme oranı
        #true_positive_rate (recall): TP / (TP + FN) fraud işlemlerini yakalama oranı

        #roc curve grafiği:
        plt.figure(figsize=(9,6))
        plt.plot(false_positive_rate, true_positive_rate, label="ROC Curve")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("assets/model/roc_curve.png", dpi=300)
        plt.show()

        print(f"\nROC-AUC Score: {auc_score:.4f}")

    # Threshold tuning yapalım
    def threshold_tuning(self, y_test, y_prob, thresholds = None):
        """ 
        Farklı threshold değerlerinde precision, recall ve f1 skorlarını hesaplayıp karşılaştırır
        """
        best_threshold, best_f1, results = find_best_threshold_by_f1(y_test, y_prob, thresholds)

        print("Threshold | Precision | Recall | F1 Score")
        print("-" * 50)

        for item in results:
            print(f"{item['threshold']:.2f} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1_score']:.4f}") #threshold, precision, recall, f1_score verecek
        
        print(f"\nBest Threshold by F1: {best_threshold:.2f}")
        print(f"Best F1 Score: {best_f1:.4f}")
        return best_threshold


    def predict_risk_score(self, transaction):
        """ Tek bir işlem için fraud risk skorunu hesaplayalım """
        risk_score = self.model.predict(transaction, verbose=0).flatten()[0]
        return risk_score

    #Threshold tuningin grafikleştirilmiş hali
    def threshold_metrics_plot (self, y_test, y_prob, thresholds=None):
        _, _, results = find_best_threshold_by_f1(y_test, y_prob, thresholds)

        threshold_list = [item["threshold"] for item in results]
        precision_list = [item["precision"] for item in results]
        recall_list = [item["recall"] for item in results]
        f1_score_list = [item["f1_score"] for item in results]
        
        plt.figure(figsize=(10,6))
        plt.plot(threshold_list, precision_list, label="Precision")
        plt.plot(threshold_list, recall_list, label="Recall")
        plt.plot(threshold_list, f1_score_list, label="F1 Score")

        plt.title("Threshold vs Precision / Recall / F1")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("assets/model/threshold_metrics.png", dpi=300)
        plt.show()


    #Fraud ve normal işlemlerin risk score dağılımı grafik hali
    def risk_score_distribution_plot(self, y_test, y_prob):
        normal_score = y_prob[y_test == 0]
        fraud_score = y_prob[y_test == 1]
        plt.figure(figsize=(10,6))
        plt.hist(normal_score, bins=50, alpha=0.7, label="Normal", edgecolor="black", color="#2612DB")
        plt.hist(fraud_score, bins=50, alpha=0.7, label="Fraud", edgecolor="black", color="#E5D112")
        plt.title("Fraud vs Normal Risk Score Distribution")
        plt.xlabel("Predicted Risk Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig("assets/model/risk_distribution.png", dpi=300)
        plt.show()



"""
fraud sınıfını yakalama gücü : recall
fraud dediğinde gerçekten fraud olma oranı : precision
denge : F1
ayrım gücü : ROC-AUC
"""