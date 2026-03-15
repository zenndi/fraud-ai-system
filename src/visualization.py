#Exploratory Data Analysis
#Fraud dağılımı
#Transaction amount dağılımı
#Zaman analizi
#Korelasyon grafiklerine bakacağız

import matplotlib.pyplot as plt
class Visualizer:
    def __init__(self, df):
        
        self.df = df
    
    def plot_class_distribution(self) -> None:
        #normal ve fraud işlem sayısını gösterir
        class_counts = self.df["Class"].value_counts().sort_index()
        plt.figure(figsize=(10,6))
        plt.bar(["Normal (0)", "Fraud (1)"], class_counts.values, color="#9E250D")
        plt.title("Fraud Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("assets/eda/class_distribution.png", dpi=300)
        plt.show()

    
    def plot_amount_distribution(self) -> None:
        #işlem tutarlarının dağılımını gösterir
        plt.figure(figsize=(10,6))
        plt.hist(self.df["Amount"], bins=50, edgecolor="black", color="#770D9E")
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")
        plt.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("assets/eda/amount_distribution.png", dpi=300)
        plt.show()

    def plot_time_distribution(self) -> None:
        #İşlem zamanlarının dağılımını gösterir
        plt.figure(figsize=(10,6))
        plt.hist(self.df["Time"], bins=50, edgecolor="black", color="#0C9736")
        plt.title("Transaction Time Distribution")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("assets/eda/time_distribution.png", dpi=300)
        plt.show()

    def plot_fraud_vs_normal_amount(self) -> None:
        #Fraud ve normal işlemlerin amount dağılımını gösterir
        fraud_amount = self.df[self.df["Class"] == 1]["Amount"]
        normal_amount = self.df[self.df["Class"] == 0]["Amount"]

        plt.figure(figsize=(10,6))
        plt.hist(normal_amount, bins=50, alpha=0.7, label="Normal", edgecolor="black", color="#2612DB")
        plt.hist(fraud_amount, bins=50, alpha=0.7, label="Fraud", edgecolor="black", color="#E5D112")
        plt.title("Fraud vs Normal Transaction Amount Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("assets/eda/fraud_vs_normal_amount.png", dpi=300)
        plt.show()
    
    def plot_correlation_heatmap(self) -> None:
        #korelasyon matrisini heatmap olarak gösterelim
        corr_matrix = self.df.corr()

        plt.figure(figsize=(13,8))
        plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Heatmap")
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("assets/eda/correlation_heatmap.png", dpi=300)
        plt.show()

    def plot_training_history(self, history):
        #Loss curve
        plt.figure(figsize=(10,6))
        plt.plot(history.history["loss"], label="Training Loss") #her epochdaki training loss
        plt.plot(history.history["val_loss"], label="Validation Loss") #her epochdaki validation loss
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("assets/model/training_loss.png", dpi=300)
        plt.show()

        #Accuracy curve:
        plt.figure(figsize=(10,6))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("assets/model/training_accuracy.png", dpi=300)
        plt.show()

    