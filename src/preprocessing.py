import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


class Preprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Veri Ön İşleme Sınıfı
        
        Args:
            test_size: Test verisi oranı
            random_state: Rastgelelik için seed
        """
        self.amount_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state
    
    def split_features_and_target(self, df: pd.DataFrame):
        """
        Feature ve target ayrımı
        
        Args:
            df: Ham veri DataFrame'i
        
        Returns:
            tuple: (X, y)
        """
        X = df.drop("Class", axis=1).copy()
        y = df["Class"].copy()
        return X, y
    
    def split_data(self, X, y):
        """
        Train/Test split
        
        Args:
            X: Features
            y: Target
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def scale_data(self, X_train, X_test):
        """
        Amount ve Time sütunlarını ölçeklendir
        
        Args:
            X_train: Eğitim features
            X_test: Test features
        
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # Amount scaling
        X_train["Amount"] = self.amount_scaler.fit_transform(X_train[["Amount"]])
        X_test["Amount"] = self.amount_scaler.transform(X_test[["Amount"]])
        
        # Time scaling
        X_train["Time"] = self.time_scaler.fit_transform(X_train[["Time"]])
        X_test["Time"] = self.time_scaler.transform(X_test[["Time"]])
        
        return X_train, X_test
    
    def apply_undersampling(self, X_train, y_train):
        """
        Undersampling uygula - çoğunluk sınıfını azalt
        
        Args:
            X_train: Eğitim features
            y_train: Eğitim labels
        
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        train_df = X_train.copy()
        train_df["Class"] = y_train.values
        
        fraud_df = train_df[train_df["Class"] == 1]
        normal_df = train_df[train_df["Class"] == 0]
        
        normal_under = resample(
            normal_df, 
            replace=False, 
            n_samples=len(fraud_df), 
            random_state=self.random_state
        )
        balanced_df = pd.concat([normal_under, fraud_df]).sample(
            frac=1, random_state=self.random_state
        )
        
        X_balanced = balanced_df.drop("Class", axis=1)
        y_balanced = balanced_df["Class"]
        return X_balanced, y_balanced
    
    def apply_oversampling(self, X_train, y_train):
        """
        Oversampling uygula - azınlık sınıfını artır
        
        Args:
            X_train: Eğitim features
            y_train: Eğitim labels
        
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        train_df = X_train.copy()
        train_df["Class"] = y_train.values
        
        fraud_df = train_df[train_df["Class"] == 1]
        normal_df = train_df[train_df["Class"] == 0]
        
        fraud_over = resample(
            fraud_df, 
            replace=True, 
            n_samples=len(normal_df), 
            random_state=self.random_state
        )
        balanced_df = pd.concat([normal_df, fraud_over]).sample(
            frac=1, random_state=self.random_state
        )
        
        X_balanced = balanced_df.drop("Class", axis=1)
        y_balanced = balanced_df["Class"]
        return X_balanced, y_balanced
    
    def apply_smote(self, X_train, y_train):
        """
        SMOTE uygula - yapay örnek oluşturarak dengele
        
        Args:
            X_train: Eğitim features
            y_train: Eğitim labels
        
        Returns:
            tuple: (X_balanced, y_balanced)
        
        Not:
            SMOTE, azınlık sınıfından yeni yapay örnekler oluşturur.
            Oversampling'den farkı: sadece kopyalama yerine
            mevcut örnekler arasında interpolasyon yapar.
        """
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            return pd.DataFrame(X_balanced, columns=X_train.columns), pd.Series(y_balanced)
        except ImportError:
            print("⚠️ imbalanced-learn not installed. Falling back to oversampling.")
            return self.apply_oversampling(X_train, y_train)
    
    def apply_adasyn(self, X_train, y_train):
        """
        ADASYN uygula - Adaptive Synthetic Sampling
        
        Args:
            X_train: Eğitim features
            y_train: Eğitim labels
        
        Returns:
            tuple: (X_balanced, y_balanced)
        
        Not:
            ADASYN, SMOTE'nin gelişmiş versiyonu.
            Zor öğrenilen örneklerde daha fazla sentetik veri oluşturur.
        """
        try:
            from imblearn.over_sampling import ADASYN
            adasyn = ADASYN(random_state=self.random_state)
            X_balanced, y_balanced = adasyn.fit_resample(X_train, y_train)
            return pd.DataFrame(X_balanced, columns=X_train.columns), pd.Series(y_balanced)
        except ImportError:
            print("⚠️ imbalanced-learn not installed. Falling back to oversampling.")
            return self.apply_oversampling(X_train, y_train)
