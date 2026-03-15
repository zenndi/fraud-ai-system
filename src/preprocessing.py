import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

class Preprocessor:
    def __init__(self):
        self.amount_scaler = StandardScaler() #amount sütunu için scaler  
        self.time_scaler = StandardScaler()   #time sütunu için scaler

    def split_features_and_target(self, df : pd.DataFrame):
        """
        Feature ve target ayrımı yapar. 
        X -> Bağımsız değişkenler
        y -> bağımlı değişken, target burada Class sütunu
        """
        X = df.drop("Class", axis = 1).copy() 
        y = df["Class"].copy()
        return X,y 

    def split_data(self, X, y):
        #verinin %20'si test için ayrıldı.
        #stratify=y deme nedenimiz class oranını train ve testte korumak için çünkü fraud dataset çok dengesiz oluyor
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    
    def scale_data(self, X_train, X_test):
        #Amount ve Time sütunlrını ölçeklendiriyoruz
        #orijinal veriyi bozmamak için kopya alıyoruz
        X_train = X_train.copy() 
        X_test = X_test.copy()

        #Amount sütunu için scaling yapalım
        X_train["Amount"] = self.amount_scaler.fit_transform(X_train[["Amount"]])
        X_test["Amount"] = self.amount_scaler.transform(X_test[["Amount"]])

        #Time sütunu için scaling yapalım
        X_train["Time"] = self.time_scaler.fit_transform(X_train[["Time"]])
        X_test["Time"] = self.time_scaler.transform(X_test[["Time"]])

        return X_train, X_test

    #Undersampling işlemi: dengesiz veri setlerinde çoğunluk sınıfındaki örnek sayısını azaltarak, azınlık sınıfı ile dengelemek için kullanılan bir makine öğrenmesi veri ön işleme tekniğidir
    def apply_undersampling(self, X_train, y_train):
        #normal sınıftan rastgele azaltır ve fraud saysıına eşitler

        #train feauture ve label tekrar birleştirilir
        train_df = X_train.copy()
        train_df["Class"] = y_train.values

        #fraud sınıfı:
        fraud_df = train_df[train_df["Class"] == 1]
        #normal sınıf:
        normal_df = train_df[train_df["Class"] == 0]

        # normal sınıftan fraud sayısı kadar örnek seçilir ve fraud sınıfına eşitlenir
        normal_under = resample( normal_df, replace=False, n_samples=len(fraud_df), random_state=42)
        balanced_df = pd.concat([normal_under, fraud_df]).sample(frac=1,random_state=42) #iki sınıf birleştirilir ve kaırştırılır

        #tekrardan X ve y ayrılır
        X_balanced = balanced_df.drop("Class", axis=1)
        y_balanced = balanced_df["Class"]
        return X_balanced, y_balanced

    
    #Oversampling: veri biliminde dengesiz veri setlerini (azınlık sınıfın çok az olduğu) eşitlemek için bu azınlık örneklerini yapay olarak çoğaltma tekniğidir
    def apply_oversampling(self, X_train, y_train):
        #fraud sınıfını tekrar örnekleyip arttırıyor ve normal sayısına eşitliyor
        train_df = X_train.copy()
        train_df["Class"] = y_train.values

        fraud_df = train_df[train_df["Class"] == 1]
        normal_df = train_df[train_df["Class"] == 0]

        #fraud sınıfı tekrar örneklenerek artırılır
        fraud_over = resample( fraud_df, replace=True, n_samples=len(normal_df), random_state=42)
        balanced_df = pd.concat([normal_df, fraud_over]).sample(frac=1,random_state=42) #iki sınıf birleştirilir

        #tekrardan X ve y ayrılır
        X_balanced = balanced_df.drop("Class", axis=1)
        y_balanced = balanced_df["Class"]
        return X_balanced, y_balanced