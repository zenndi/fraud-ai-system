import tensorflow as tf
class Trainer:
    def __init__(self, model):
        self.model = model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=30, batch_size=2048):
        #monitor="val_loss" ile validation loss izlenir eğer iyileşmezse training durdururlur
        #patience=5 ise validation loss 5 epoch boyunca iyileşmezse training durur
        #restore_best_weights=True ise training sonunda en iyi epochun ağırlıkları geri yüklenir
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose = 1
        )
        return history