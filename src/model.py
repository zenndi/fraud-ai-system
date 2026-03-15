import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

class FraudDetectionModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build_model(self):
        # Model oluşturuldu
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)), #giriş katmanı feature sayısı kadar input alır input_dim bizim verimizde 30 olur
            tf.keras.layers.Dense(32, activation='relu'), #32 nöron oluşturduk
            tf.keras.layers.Dense(16, activation='relu'), #16 nöron oluşturduk
            tf.keras.layers.Dense(8, activation='relu'), #8 nöron oluşturduk
            tf.keras.layers.Dense(1, activation='sigmoid') #çıkış katmanı binary classification olduğu için 0-1 arası olasılık üretir
        ])
        
        model.compile(
            optimizer="adam", #optimizasyon algoritması
            loss="binary_crossentropy", #ikili sınıf problemi old icin
            metrics=["accuracy"] #dogrulama
        )

        return model