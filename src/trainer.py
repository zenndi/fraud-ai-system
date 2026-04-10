import tensorflow as tf
import yaml
class Trainer:
    def __init__(self, model, config_path=None):
        self.model = model

        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    

    def train(self, X_train, y_train, X_test, y_test, epochs=30, batch_size=2048, class_weights=None):
        #monitor="val_loss" ile validation loss izlenir eğer iyileşmezse training durdururlur
        #patience=5 ise validation loss 5 epoch boyunca iyileşmezse training durur
        #restore_best_weights=True ise training sonunda en iyi epochun ağırlıkları geri yüklenir
        
        # config'ten değerleri çek
        epochs = self.config.get("training", {}).get("epochs", epochs)
        batch_size = self.config.get("training", {}).get("batch_size", batch_size)
        patience = self.config.get("training", {}).get("early_stopping", {}).get("patience", 5)


        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose = 1
        )
        return history