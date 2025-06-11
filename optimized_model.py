import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, Concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration des paramètres
SEQUENCE_LENGTH = 20  # Augmenté pour capturer plus de patterns
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_NAME = f"euromillions_optimized_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

class EuromillionsPredictor:
    """
    Classe principale pour la prédiction des numéros Euromillions avec modèles multiples.
    """
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.main_scaler = MinMaxScaler(feature_range=(0, 1))
        self.stars_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
        # Modèles
        self.lstm_main_model = None
        self.lstm_stars_model = None
        self.rf_main_model = None
        self.rf_stars_model = None
        self.xgb_main_model = None
        self.xgb_stars_model = None
        
        # Historique d'entraînement
        self.training_history = {}
        
    def load_and_prepare_data(self, file_path):
        """
        Charge et prépare les données pour l'entraînement.
        
        Args:
            file_path: Chemin vers le fichier CSV contenant les données
        
        Returns:
            Tuple contenant les données préparées
        """
        # Charger les données
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Données chargées: {len(df)} tirages avec {len(df.columns)} caractéristiques")
        
        # Séparer les numéros principaux et les étoiles
        main_numbers = df[['N1', 'N2', 'N3', 'N4', 'N5']].values
        stars = df[['E1', 'E2']].values
        
        # Caractéristiques supplémentaires
        feature_cols = [col for col in df.columns if col not in ['Date', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']]
        features = df[feature_cols].values
        
        # Normaliser les données
        main_normalized = self.main_scaler.fit_transform(main_numbers)
        stars_normalized = self.stars_scaler.fit_transform(stars)
        features_normalized = self.feature_scaler.fit_transform(features)
        
        # Créer des séquences pour LSTM
        X_main_lstm, y_main_lstm = self.create_sequences(main_normalized, self.sequence_length)
        X_stars_lstm, y_stars_lstm = self.create_sequences(stars_normalized, self.sequence_length)
        X_features_lstm, _ = self.create_sequences(features_normalized, self.sequence_length)
        
        # Préparer les données pour Random Forest et XGBoost
        X_main_ml, y_main_ml = self.prepare_ml_data(main_normalized, features_normalized)
        X_stars_ml, y_stars_ml = self.prepare_ml_data(stars_normalized, features_normalized)
        
        # Division train/test avec validation temporelle
        split_idx = int(len(X_main_lstm) * 0.8)
        
        # Données LSTM
        X_main_lstm_train = X_main_lstm[:split_idx]
        X_main_lstm_test = X_main_lstm[split_idx:]
        y_main_lstm_train = y_main_lstm[:split_idx]
        y_main_lstm_test = y_main_lstm[split_idx:]
        
        X_stars_lstm_train = X_stars_lstm[:split_idx]
        X_stars_lstm_test = X_stars_lstm[split_idx:]
        y_stars_lstm_train = y_stars_lstm[:split_idx]
        y_stars_lstm_test = y_stars_lstm[split_idx:]
        
        X_features_lstm_train = X_features_lstm[:split_idx]
        X_features_lstm_test = X_features_lstm[split_idx:]
        
        # Données ML
        split_idx_ml = int(len(X_main_ml) * 0.8)
        
        X_main_ml_train = X_main_ml[:split_idx_ml]
        X_main_ml_test = X_main_ml[split_idx_ml:]
        y_main_ml_train = y_main_ml[:split_idx_ml]
        y_main_ml_test = y_main_ml[split_idx_ml:]
        
        X_stars_ml_train = X_stars_ml[:split_idx_ml]
        X_stars_ml_test = X_stars_ml[split_idx_ml:]
        y_stars_ml_train = y_stars_ml[:split_idx_ml]
        y_stars_ml_test = y_stars_ml[split_idx_ml:]
        
        return {
            'lstm': {
                'main': (X_main_lstm_train, y_main_lstm_train, X_main_lstm_test, y_main_lstm_test),
                'stars': (X_stars_lstm_train, y_stars_lstm_train, X_stars_lstm_test, y_stars_lstm_test),
                'features': (X_features_lstm_train, X_features_lstm_test)
            },
            'ml': {
                'main': (X_main_ml_train, y_main_ml_train, X_main_ml_test, y_main_ml_test),
                'stars': (X_stars_ml_train, y_stars_ml_train, X_stars_ml_test, y_stars_ml_test)
            }
        }
    
    def create_sequences(self, data, seq_length):
        """
        Crée des séquences pour l'apprentissage LSTM.
        
        Args:
            data: Données normalisées
            seq_length: Longueur de la séquence
        
        Returns:
            Tuple (X, y) où X contient les séquences et y les valeurs cibles
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def prepare_ml_data(self, target_data, features_data):
        """
        Prépare les données pour Random Forest et XGBoost.
        
        Args:
            target_data: Données cibles normalisées
            features_data: Caractéristiques normalisées
        
        Returns:
            Tuple (X, y) pour l'apprentissage ML
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(target_data)):
            # Utiliser les dernières valeurs et caractéristiques
            sequence_data = target_data[i-self.sequence_length:i].flatten()
            current_features = features_data[i]
            combined_features = np.concatenate([sequence_data, current_features])
            
            X.append(combined_features)
            y.append(target_data[i])
        
        return np.array(X), np.array(y)
    
    def create_advanced_lstm_model(self, input_shape, output_dim, model_type='main'):
        """
        Crée un modèle LSTM avancé avec attention.
        
        Args:
            input_shape: Forme des données d'entrée
            output_dim: Dimension de sortie
            model_type: Type de modèle ('main' ou 'stars')
        
        Returns:
            Modèle Keras compilé
        """
        # Entrée principale (séquences)
        sequence_input = Input(shape=input_shape, name='sequence_input')
        
        # Couches LSTM avec attention
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(sequence_input)
        lstm1_norm = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm1_norm)
        lstm2_norm = BatchNormalization()(lstm2)
        
        lstm3 = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm2_norm)
        lstm3_norm = BatchNormalization()(lstm3)
        
        # Couches denses
        dense1 = Dense(128, activation='relu')(lstm3_norm)
        dense1_norm = BatchNormalization()(dense1)
        dense1_drop = Dropout(0.3)(dense1_norm)
        
        dense2 = Dense(64, activation='relu')(dense1_drop)
        dense2_norm = BatchNormalization()(dense2)
        dense2_drop = Dropout(0.3)(dense2_norm)
        
        dense3 = Dense(32, activation='relu')(dense2_drop)
        dense3_norm = BatchNormalization()(dense3)
        dense3_drop = Dropout(0.2)(dense3_norm)
        
        # Couche de sortie
        output = Dense(output_dim, activation='sigmoid', name='output')(dense3_drop)
        
        # Créer le modèle
        model = Model(inputs=sequence_input, outputs=output)
        
        # Compiler le modèle
        optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_models(self, data):
        """
        Entraîne les modèles LSTM pour les numéros principaux et les étoiles.
        
        Args:
            data: Données préparées
        """
        print("Entraînement des modèles LSTM...")
        
        # Modèle pour les numéros principaux
        X_main_train, y_main_train, X_main_test, y_main_test = data['lstm']['main']
        
        self.lstm_main_model = self.create_advanced_lstm_model(
            (X_main_train.shape[1], X_main_train.shape[2]), 5, 'main'
        )
        
        # Callbacks
        callbacks_main = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss'),
            ModelCheckpoint(f'models/{MODEL_NAME}_lstm_main.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Entraînement
        history_main = self.lstm_main_model.fit(
            X_main_train, y_main_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_main_test, y_main_test),
            callbacks=callbacks_main,
            verbose=1
        )
        
        self.training_history['lstm_main'] = history_main
        
        # Modèle pour les étoiles
        X_stars_train, y_stars_train, X_stars_test, y_stars_test = data['lstm']['stars']
        
        self.lstm_stars_model = self.create_advanced_lstm_model(
            (X_stars_train.shape[1], X_stars_train.shape[2]), 2, 'stars'
        )
        
        # Callbacks
        callbacks_stars = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss'),
            ModelCheckpoint(f'models/{MODEL_NAME}_lstm_stars.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Entraînement
        history_stars = self.lstm_stars_model.fit(
            X_stars_train, y_stars_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_stars_test, y_stars_test),
            callbacks=callbacks_stars,
            verbose=1
        )
        
        self.training_history['lstm_stars'] = history_stars
        
        print("Modèles LSTM entraînés avec succès!")
    
    def train_random_forest_models(self, data):
        """
        Entraîne les modèles Random Forest.
        
        Args:
            data: Données préparées
        """
        print("Entraînement des modèles Random Forest...")
        
        # Modèle pour les numéros principaux
        X_main_train, y_main_train, X_main_test, y_main_test = data['ml']['main']
        
        self.rf_main_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Entraîner pour chaque numéro principal séparément
        self.rf_main_models = []
        for i in range(5):
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42 + i,
                n_jobs=-1
            )
            rf_model.fit(X_main_train, y_main_train[:, i])
            self.rf_main_models.append(rf_model)
        
        # Modèle pour les étoiles
        X_stars_train, y_stars_train, X_stars_test, y_stars_test = data['ml']['stars']
        
        self.rf_stars_models = []
        for i in range(2):
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42 + i,
                n_jobs=-1
            )
            rf_model.fit(X_stars_train, y_stars_train[:, i])
            self.rf_stars_models.append(rf_model)
        
        print("Modèles Random Forest entraînés avec succès!")
    
    def train_xgboost_models(self, data):
        """
        Entraîne les modèles XGBoost.
        
        Args:
            data: Données préparées
        """
        print("Entraînement des modèles XGBoost...")
        
        # Modèle pour les numéros principaux
        X_main_train, y_main_train, X_main_test, y_main_test = data['ml']['main']
        
        self.xgb_main_models = []
        for i in range(5):
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + i,
                n_jobs=-1
            )
            xgb_model.fit(X_main_train, y_main_train[:, i])
            self.xgb_main_models.append(xgb_model)
        
        # Modèle pour les étoiles
        X_stars_train, y_stars_train, X_stars_test, y_stars_test = data['ml']['stars']
        
        self.xgb_stars_models = []
        for i in range(2):
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + i,
                n_jobs=-1
            )
            xgb_model.fit(X_stars_train, y_stars_train[:, i])
            self.xgb_stars_models.append(xgb_model)
        
        print("Modèles XGBoost entraînés avec succès!")
    
    def ensemble_predict(self, X_lstm, X_ml):
        """
        Fait des prédictions en combinant tous les modèles.
        
        Args:
            X_lstm: Données pour les modèles LSTM
            X_ml: Données pour les modèles ML
        
        Returns:
            Tuple (numéros principaux prédits, étoiles prédites)
        """
        # Prédictions LSTM
        lstm_main_pred = self.lstm_main_model.predict(X_lstm['main'])
        lstm_stars_pred = self.lstm_stars_model.predict(X_lstm['stars'])
        
        # Prédictions Random Forest
        rf_main_pred = np.array([model.predict(X_ml['main']) for model in self.rf_main_models]).T
        rf_stars_pred = np.array([model.predict(X_ml['stars']) for model in self.rf_stars_models]).T
        
        # Prédictions XGBoost
        xgb_main_pred = np.array([model.predict(X_ml['main']) for model in self.xgb_main_models]).T
        xgb_stars_pred = np.array([model.predict(X_ml['stars']) for model in self.xgb_stars_models]).T
        
        # Ensemble avec pondération
        weights_main = [0.5, 0.25, 0.25]  # LSTM, RF, XGB
        weights_stars = [0.5, 0.25, 0.25]
        
        ensemble_main = (weights_main[0] * lstm_main_pred + 
                        weights_main[1] * rf_main_pred + 
                        weights_main[2] * xgb_main_pred)
        
        ensemble_stars = (weights_stars[0] * lstm_stars_pred + 
                         weights_stars[1] * rf_stars_pred + 
                         weights_stars[2] * xgb_stars_pred)
        
        return ensemble_main, ensemble_stars
    
    def predict_next_numbers(self, df):
        """
        Prédit les numéros pour le prochain tirage.
        
        Args:
            df: DataFrame avec les données historiques
        
        Returns:
            Tuple (numéros principaux prédits, étoiles prédites)
        """
        # Préparer les dernières données
        main_numbers = df[['N1', 'N2', 'N3', 'N4', 'N5']].values
        stars = df[['E1', 'E2']].values
        feature_cols = [col for col in df.columns if col not in ['Date', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']]
        features = df[feature_cols].values
        
        # Normaliser
        main_normalized = self.main_scaler.transform(main_numbers)
        stars_normalized = self.stars_scaler.transform(stars)
        features_normalized = self.feature_scaler.transform(features)
        
        # Préparer les données pour LSTM
        X_main_lstm = main_normalized[-self.sequence_length:].reshape(1, self.sequence_length, 5)
        X_stars_lstm = stars_normalized[-self.sequence_length:].reshape(1, self.sequence_length, 2)
        
        # Préparer les données pour ML
        sequence_data_main = main_normalized[-self.sequence_length:].flatten()
        sequence_data_stars = stars_normalized[-self.sequence_length:].flatten()
        current_features = features_normalized[-1]
        
        X_main_ml = np.concatenate([sequence_data_main, current_features]).reshape(1, -1)
        X_stars_ml = np.concatenate([sequence_data_stars, current_features]).reshape(1, -1)
        
        # Faire les prédictions ensemble
        ensemble_main, ensemble_stars = self.ensemble_predict(
            {'main': X_main_lstm, 'stars': X_stars_lstm},
            {'main': X_main_ml, 'stars': X_stars_ml}
        )
        
        # Inverser la normalisation
        main_pred = self.main_scaler.inverse_transform(ensemble_main)
        stars_pred = self.stars_scaler.inverse_transform(ensemble_stars)
        
        # Arrondir et ajuster
        main_numbers = np.round(main_pred[0]).astype(int)
        star_numbers = np.round(stars_pred[0]).astype(int)
        
        # S'assurer que les numéros sont dans les plages valides
        main_numbers = np.clip(main_numbers, 1, 50)
        star_numbers = np.clip(star_numbers, 1, 12)
        
        # Éliminer les doublons et compléter si nécessaire
        main_numbers = self.ensure_unique_numbers(main_numbers, 5, 1, 50)
        star_numbers = self.ensure_unique_numbers(star_numbers, 2, 1, 12)
        
        return main_numbers, star_numbers
    
    def ensure_unique_numbers(self, numbers, target_count, min_val, max_val):
        """
        S'assure que nous avons le bon nombre de numéros uniques.
        
        Args:
            numbers: Numéros prédits
            target_count: Nombre cible de numéros
            min_val: Valeur minimale
            max_val: Valeur maximale
        
        Returns:
            Array de numéros uniques
        """
        unique_numbers = np.unique(numbers)
        
        while len(unique_numbers) < target_count:
            # Ajouter des numéros aléatoires
            new_num = np.random.randint(min_val, max_val + 1)
            if new_num not in unique_numbers:
                unique_numbers = np.append(unique_numbers, new_num)
        
        # Trier et prendre les premiers
        return np.sort(unique_numbers)[:target_count]
    
    def plot_training_history(self):
        """
        Visualise l'historique d'entraînement des modèles LSTM.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Modèle principal - Perte
        if 'lstm_main' in self.training_history:
            history = self.training_history['lstm_main']
            axes[0, 0].plot(history.history['loss'], label='Entraînement')
            axes[0, 0].plot(history.history['val_loss'], label='Validation')
            axes[0, 0].set_title('Perte - Numéros Principaux')
            axes[0, 0].set_xlabel('Époque')
            axes[0, 0].set_ylabel('Perte')
            axes[0, 0].legend()
        
        # Modèle principal - MAE
        if 'lstm_main' in self.training_history:
            history = self.training_history['lstm_main']
            axes[0, 1].plot(history.history['mae'], label='Entraînement')
            axes[0, 1].plot(history.history['val_mae'], label='Validation')
            axes[0, 1].set_title('MAE - Numéros Principaux')
            axes[0, 1].set_xlabel('Époque')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
        
        # Modèle étoiles - Perte
        if 'lstm_stars' in self.training_history:
            history = self.training_history['lstm_stars']
            axes[1, 0].plot(history.history['loss'], label='Entraînement')
            axes[1, 0].plot(history.history['val_loss'], label='Validation')
            axes[1, 0].set_title('Perte - Étoiles')
            axes[1, 0].set_xlabel('Époque')
            axes[1, 0].set_ylabel('Perte')
            axes[1, 0].legend()
        
        # Modèle étoiles - MAE
        if 'lstm_stars' in self.training_history:
            history = self.training_history['lstm_stars']
            axes[1, 1].plot(history.history['mae'], label='Entraînement')
            axes[1, 1].plot(history.history['val_mae'], label='Validation')
            axes[1, 1].set_title('MAE - Étoiles')
            axes[1, 1].set_xlabel('Époque')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{MODEL_NAME}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Fonction principale pour entraîner le modèle optimisé.
    """
    print("=== Entraînement du modèle Euromillions optimisé ===")
    
    # Créer les répertoires nécessaires
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialiser le prédicteur
    predictor = EuromillionsPredictor()
    
    # Charger et préparer les données
    print("Chargement et préparation des données...")
    data = predictor.load_and_prepare_data("euromillions_enhanced_dataset.csv")
    
    # Entraîner tous les modèles
    predictor.train_lstm_models(data)
    predictor.train_random_forest_models(data)
    predictor.train_xgboost_models(data)
    
    # Visualiser l'historique d'entraînement
    predictor.plot_training_history()
    
    # Faire une prédiction
    print("\nGénération d'une prédiction...")
    df = pd.read_csv("euromillions_enhanced_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    main_numbers, star_numbers = predictor.predict_next_numbers(df)
    
    print(f"\nPrédiction optimisée pour le prochain tirage:")
    print(f"Numéros principaux: {main_numbers}")
    print(f"Étoiles: {star_numbers}")
    
    # Sauvegarder la prédiction
    with open("prediction_optimized.txt", "w") as f:
        f.write(f"Prédiction optimisée pour le prochain tirage de l'Euromillions (générée le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}):\n")
        f.write(f"Numéros principaux: {', '.join(map(str, main_numbers))}\n")
        f.write(f"Étoiles: {', '.join(map(str, star_numbers))}\n")
        f.write(f"\nModèles utilisés: LSTM avancé + Random Forest + XGBoost (ensemble learning)\n")
        f.write(f"Données: {len(df)} tirages réels avec {len(df.columns)} caractéristiques\n")
    
    print(f"\nPrédiction sauvegardée dans 'prediction_optimized.txt'")
    print("Modèle optimisé entraîné avec succès!")

if __name__ == "__main__":
    main()

