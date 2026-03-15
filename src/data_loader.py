#Veri yükleme yapacağız
import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self ) -> pd.DataFrame:
        """Csv dosyasını okur yükler"""
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def basic_info(self) -> None:
        """Veri setinin genel bilgilerini verir"""
        print("\n--- DATASET INFO ---")
        print(f"Dataset shape:  {self.df.shape}")
        print(f"\nColumns: {self.df.columns.tolist()}")
        print(f"\nNumber of missing data:\n{self.df.isnull().sum()}")
        print(f"\nClass distribution:\n{self.df['Class'].value_counts()}")

        fraud_ratio = self.df["Class"].mean()*100 #dolandırıcılık oranı
        print(f"\nFraud ratio: %{fraud_ratio:.3f}")