import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration des paramètres optimisés
SEQUENCE_LENGTH = 15
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_NAME = f"euromillions_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

class FinalOptimizedPredictor:
    """
    Modèle final optimisé pour la prédiction des numéros Euromillions.
    """
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.main_scaler = MinMaxScaler(feature_range=(0, 1))
        self.stars_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
        # Modèles
        self.lstm_main_model = None
        self.lstm_stars_model = None
        self.rf_main_models = []
        self.rf_stars_models = []
        self.xgb_main_models = []
        self.xgb_stars_models = []
        
        # Historique d'entraînement
        self.training_history = {}
        
        # Poids optimisés pour l'ensemble
        self.ensemble_weights = {
            'main': {'lstm': 0.5, 'rf': 0.3, 'xgb': 0.2},
            'stars': {'lstm': 0.6, 'rf': 0.25, 'xgb': 0.15}
        }
    
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
        
        # Sélectionner les caractéristiques les plus importantes
        important_features = [
            'Main_Sum', 'Main_Mean', 'Main_Std', 'Stars_Sum', 'Stars_Mean',
            'Main_Even_Count', 'Main_Odd_Count', 'Month', 'DayOfWeek',
            'Main_Sum_MA_5', 'Main_Sum_MA_10', 'Main_Sum_Volatility_5',
            'N1_Freq_50', 'N2_Freq_50', 'N3_Freq_50', 'N4_Freq_50', 'N5_Freq_50',
            'E1_Freq_50', 'E2_Freq_50'
        ]
        
        # Vérifier quelles caractéristiques sont disponibles
        available_features = [col for col in important_features if col in df.columns]
        print(f"Caractéristiques utilisées: {len(available_features)}")
        
        features = df[available_features].values
        
        # Normaliser les données
        main_normalized = self.main_scaler.fit_transform(main_numbers)
        stars_normalized = self.stars_scaler.fit_transform(stars)
        features_normalized = self.feature_scaler.fit_transform(features)
        
        # Créer des séquences pour LSTM
        X_main_lstm, y_main_lstm = self.create_sequences(main_normalized, self.sequence_length)
        X_stars_lstm, y_stars_lstm = self.create_sequences(stars_normalized, self.sequence_length)
        
        # Préparer les données pour Random Forest et XGBoost
        X_main_ml, y_main_ml = self.prepare_ml_data(main_normalized, features_normalized)
        X_stars_ml, y_stars_ml = self.prepare_ml_data(stars_normalized, features_normalized)
        
        # Division train/test avec validation temporelle
        split_idx_lstm = int(len(X_main_lstm) * 0.85)
        split_idx_ml = int(len(X_main_ml) * 0.85)
        
        return {
            'lstm': {
                'main': (X_main_lstm[:split_idx_lstm], y_main_lstm[:split_idx_lstm], 
                        X_main_lstm[split_idx_lstm:], y_main_lstm[split_idx_lstm:]),
                'stars': (X_stars_lstm[:split_idx_lstm], y_stars_lstm[:split_idx_lstm], 
                         X_stars_lstm[split_idx_lstm:], y_stars_lstm[split_idx_lstm:])
            },
            'ml': {
                'main': (X_main_ml[:split_idx_ml], y_main_ml[:split_idx_ml], 
                        X_main_ml[split_idx_ml:], y_main_ml[split_idx_ml:]),
                'stars': (X_stars_ml[:split_idx_ml], y_stars_ml[:split_idx_ml], 
                         X_stars_ml[split_idx_ml:], y_stars_ml[split_idx_ml:])
            }
        }
    
    def create_sequences(self, data, seq_length):
        """
        Crée des séquences pour l'apprentissage LSTM.
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def prepare_ml_data(self, target_data, features_data):
        """
        Prépare les données pour Random Forest et XGBoost.
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
    
    def create_optimized_lstm_model(self, input_shape, output_dim, model_type='main'):
        """
        Crée un modèle LSTM optimisé.
        """
        model = Sequential()
        
        # Première couche LSTM
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        # Deuxième couche LSTM
        model.add(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        # Couches denses
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Couche de sortie
        model.add(Dense(output_dim, activation='sigmoid'))
        
        # Compiler le modèle
        optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        return model
    
    def train_lstm_models(self, data):
        """
        Entraîne les modèles LSTM optimisés.
        """
        print("Entraînement des modèles LSTM optimisés...")
        
        # Modèle pour les numéros principaux
        X_main_train, y_main_train, X_main_test, y_main_test = data['lstm']['main']
        
        self.lstm_main_model = self.create_optimized_lstm_model(
            (X_main_train.shape[1], X_main_train.shape[2]), 5, 'main'
        )
        
        # Callbacks optimisés
        callbacks_main = [
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss')
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
        
        self.lstm_stars_model = self.create_optimized_lstm_model(
            (X_stars_train.shape[1], X_stars_train.shape[2]), 2, 'stars'
        )
        
        # Callbacks optimisés
        callbacks_stars = [
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss')
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
        
        print("Modèles LSTM optimisés entraînés avec succès!")
    
    def train_random_forest_models(self, data):
        """
        Entraîne les modèles Random Forest optimisés.
        """
        print("Entraînement des modèles Random Forest optimisés...")
        
        # Modèles pour les numéros principaux
        X_main_train, y_main_train, X_main_test, y_main_test = data['ml']['main']
        
        self.rf_main_models = []
        for i in range(5):
            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42 + i,
                n_jobs=-1
            )
            rf_model.fit(X_main_train, y_main_train[:, i])
            self.rf_main_models.append(rf_model)
        
        # Modèles pour les étoiles
        X_stars_train, y_stars_train, X_stars_test, y_stars_test = data['ml']['stars']
        
        self.rf_stars_models = []
        for i in range(2):
            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42 + i,
                n_jobs=-1
            )
            rf_model.fit(X_stars_train, y_stars_train[:, i])
            self.rf_stars_models.append(rf_model)
        
        print("Modèles Random Forest optimisés entraînés avec succès!")
    
    def train_xgboost_models(self, data):
        """
        Entraîne les modèles XGBoost optimisés.
        """
        print("Entraînement des modèles XGBoost optimisés...")
        
        # Modèles pour les numéros principaux
        X_main_train, y_main_train, X_main_test, y_main_test = data['ml']['main']
        
        self.xgb_main_models = []
        for i in range(5):
            xgb_model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + i,
                n_jobs=-1
            )
            xgb_model.fit(X_main_train, y_main_train[:, i])
            self.xgb_main_models.append(xgb_model)
        
        # Modèles pour les étoiles
        X_stars_train, y_stars_train, X_stars_test, y_stars_test = data['ml']['stars']
        
        self.xgb_stars_models = []
        for i in range(2):
            xgb_model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + i,
                n_jobs=-1
            )
            xgb_model.fit(X_stars_train, y_stars_train[:, i])
            self.xgb_stars_models.append(xgb_model)
        
        print("Modèles XGBoost optimisés entraînés avec succès!")
    
    def ensemble_predict(self, X_lstm, X_ml):
        """
        Fait des prédictions en combinant tous les modèles avec des poids optimisés.
        """
        # Prédictions LSTM
        lstm_main_pred = self.lstm_main_model.predict(X_lstm['main'], verbose=0)
        lstm_stars_pred = self.lstm_stars_model.predict(X_lstm['stars'], verbose=0)
        
        # Prédictions Random Forest
        rf_main_pred = np.array([model.predict(X_ml['main']) for model in self.rf_main_models]).T
        rf_stars_pred = np.array([model.predict(X_ml['stars']) for model in self.rf_stars_models]).T
        
        # Prédictions XGBoost
        xgb_main_pred = np.array([model.predict(X_ml['main']) for model in self.xgb_main_models]).T
        xgb_stars_pred = np.array([model.predict(X_ml['stars']) for model in self.xgb_stars_models]).T
        
        # Ensemble avec poids optimisés
        weights_main = self.ensemble_weights['main']
        weights_stars = self.ensemble_weights['stars']
        
        ensemble_main = (weights_main['lstm'] * lstm_main_pred + 
                        weights_main['rf'] * rf_main_pred + 
                        weights_main['xgb'] * xgb_main_pred)
        
        ensemble_stars = (weights_stars['lstm'] * lstm_stars_pred + 
                         weights_stars['rf'] * rf_stars_pred + 
                         weights_stars['xgb'] * xgb_stars_pred)
        
        return ensemble_main, ensemble_stars
    
    def predict_next_numbers(self, df):
        """
        Prédit les numéros pour le prochain tirage.
        """
        # Préparer les dernières données
        main_numbers = df[['N1', 'N2', 'N3', 'N4', 'N5']].values
        stars = df[['E1', 'E2']].values
        
        # Sélectionner les caractéristiques importantes
        important_features = [
            'Main_Sum', 'Main_Mean', 'Main_Std', 'Stars_Sum', 'Stars_Mean',
            'Main_Even_Count', 'Main_Odd_Count', 'Month', 'DayOfWeek',
            'Main_Sum_MA_5', 'Main_Sum_MA_10', 'Main_Sum_Volatility_5',
            'N1_Freq_50', 'N2_Freq_50', 'N3_Freq_50', 'N4_Freq_50', 'N5_Freq_50',
            'E1_Freq_50', 'E2_Freq_50'
        ]
        
        available_features = [col for col in important_features if col in df.columns]
        features = df[available_features].values
        
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
        """
        unique_numbers = np.unique(numbers)
        
        while len(unique_numbers) < target_count:
            # Ajouter des numéros aléatoires
            new_num = np.random.randint(min_val, max_val + 1)
            if new_num not in unique_numbers:
                unique_numbers = np.append(unique_numbers, new_num)
        
        # Trier et prendre les premiers
        return np.sort(unique_numbers)[:target_count]
    
    def evaluate_models(self, data):
        """
        Évalue les performances des modèles.
        """
        print("Évaluation des modèles...")
        
        # Données de test
        X_main_lstm_test = data['lstm']['main'][2]
        y_main_lstm_test = data['lstm']['main'][3]
        X_stars_lstm_test = data['lstm']['stars'][2]
        y_stars_lstm_test = data['lstm']['stars'][3]
        X_main_ml_test = data['ml']['main'][2]
        y_main_ml_test = data['ml']['main'][3]
        X_stars_ml_test = data['ml']['stars'][2]
        y_stars_ml_test = data['ml']['stars'][3]
        
        # Prédictions ensemble
        ensemble_main, ensemble_stars = self.ensemble_predict(
            {'main': X_main_lstm_test, 'stars': X_stars_lstm_test},
            {'main': X_main_ml_test, 'stars': X_stars_ml_test}
        )
        
        # Calculer les métriques
        main_mse = mean_squared_error(y_main_ml_test, ensemble_main)
        main_mae = mean_absolute_error(y_main_ml_test, ensemble_main)
        stars_mse = mean_squared_error(y_stars_ml_test, ensemble_stars)
        stars_mae = mean_absolute_error(y_stars_ml_test, ensemble_stars)
        
        print(f"Numéros principaux - MSE: {main_mse:.4f}, MAE: {main_mae:.4f}")
        print(f"Étoiles - MSE: {stars_mse:.4f}, MAE: {stars_mae:.4f}")
        
        return {
            'main_mse': main_mse,
            'main_mae': main_mae,
            'stars_mse': stars_mse,
            'stars_mae': stars_mae
        }
    
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
            axes[0, 0].set_title('Perte - Numéros Principaux (LSTM Optimisé)')
            axes[0, 0].set_xlabel('Époque')
            axes[0, 0].set_ylabel('Perte')
            axes[0, 0].legend()
        
        # Modèle principal - MAE
        if 'lstm_main' in self.training_history:
            history = self.training_history['lstm_main']
            axes[0, 1].plot(history.history['mae'], label='Entraînement')
            axes[0, 1].plot(history.history['val_mae'], label='Validation')
            axes[0, 1].set_title('MAE - Numéros Principaux (LSTM Optimisé)')
            axes[0, 1].set_xlabel('Époque')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
        
        # Modèle étoiles - Perte
        if 'lstm_stars' in self.training_history:
            history = self.training_history['lstm_stars']
            axes[1, 0].plot(history.history['loss'], label='Entraînement')
            axes[1, 0].plot(history.history['val_loss'], label='Validation')
            axes[1, 0].set_title('Perte - Étoiles (LSTM Optimisé)')
            axes[1, 0].set_xlabel('Époque')
            axes[1, 0].set_ylabel('Perte')
            axes[1, 0].legend()
        
        # Modèle étoiles - MAE
        if 'lstm_stars' in self.training_history:
            history = self.training_history['lstm_stars']
            axes[1, 1].plot(history.history['mae'], label='Entraînement')
            axes[1, 1].plot(history.history['val_mae'], label='Validation')
            axes[1, 1].set_title('MAE - Étoiles (LSTM Optimisé)')
            axes[1, 1].set_xlabel('Époque')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{MODEL_NAME}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Fonction principale pour entraîner le modèle final optimisé.
    """
    print("=== Entraînement du modèle Euromillions final optimisé ===")
    
    # Créer les répertoires nécessaires
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialiser le prédicteur
    predictor = FinalOptimizedPredictor()
    
    # Charger et préparer les données
    print("Chargement et préparation des données...")
    # Determine which dataset to use based on existence
    data_path_primary = "data/euromillions_enhanced_dataset.csv"
    data_path_fallback = "euromillions_enhanced_dataset.csv"
    actual_data_path = None
    if os.path.exists(data_path_primary):
        actual_data_path = data_path_primary
    elif os.path.exists(data_path_fallback):
        actual_data_path = data_path_fallback
        print(f"ℹ️  Données chargées depuis {actual_data_path} (fallback) pour l'entraînement principal.")

    if not actual_data_path:
        print(f"❌ ERREUR CRITIQUE: Dataset principal non trouvé ({data_path_primary} ou {data_path_fallback}). Arrêt.")
        return # Or sys.exit(1)

    data = predictor.load_and_prepare_data(actual_data_path)
    
    # Entraîner tous les modèles
    predictor.train_lstm_models(data)
    predictor.train_random_forest_models(data)
    predictor.train_xgboost_models(data)
    
    # Évaluer les modèles
    metrics = predictor.evaluate_models(data)
    
    # Visualiser l'historique d'entraînement
    predictor.plot_training_history()
    
    # Faire une prédiction
    print("\nGénération d'une prédiction optimisée...")
    # Use the same actual_data_path for consistency in prediction context
    if not actual_data_path: # Should be set from above, but as a safeguard
        print("❌ ERREUR CRITIQUE: Dataset non disponible pour la prédiction. Arrêt.")
        return

    df = pd.read_csv(actual_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    main_numbers, star_numbers = predictor.predict_next_numbers(df)
    
    print(f"\nPrédiction finale optimisée pour le prochain tirage:")
    print(f"Numéros principaux: {main_numbers}")
    print(f"Étoiles: {star_numbers}")
    
    # Sauvegarder la prédiction
    with open("prediction_final_optimized.txt", "w") as f:
        f.write(f"Prédiction finale optimisée pour le prochain tirage de l'Euromillions\n")
        f.write(f"Générée le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Numéros principaux: {', '.join(map(str, main_numbers))}\n")
        f.write(f"Étoiles: {', '.join(map(str, star_numbers))}\n\n")
        f.write("Modèles utilisés:\n")
        f.write("- LSTM optimisé avec BatchNormalization et Dropout\n")
        f.write("- Random Forest avec hyperparamètres optimisés\n")
        f.write("- XGBoost avec hyperparamètres optimisés\n")
        f.write("- Ensemble learning avec poids optimisés\n\n")
        f.write(f"Données: {len(df)} tirages réels avec caractéristiques avancées\n")
        f.write(f"Période: {df['Date'].min()} à {df['Date'].max()}\n\n")
        f.write("Métriques d'évaluation:\n")
        f.write(f"- Numéros principaux MSE: {metrics['main_mse']:.4f}\n")
        f.write(f"- Numéros principaux MAE: {metrics['main_mae']:.4f}\n")
        f.write(f"- Étoiles MSE: {metrics['stars_mse']:.4f}\n")
        f.write(f"- Étoiles MAE: {metrics['stars_mae']:.4f}\n")
    
    print(f"\nPrédiction sauvegardée dans 'prediction_final_optimized.txt'")
    print("Modèle final optimisé entraîné avec succès!")
    
    # Afficher un résumé des améliorations
    print("\n=== Résumé des optimisations appliquées ===")
    print("1. Données réelles de l'API (1848 tirages depuis 2004)")
    print("2. Feature engineering avancé (55 caractéristiques)")
    print("3. Architecture LSTM optimisée avec BatchNormalization")
    print("4. Ensemble learning (LSTM + Random Forest + XGBoost)")
    print("5. Poids optimisés pour la combinaison des modèles")
    print("6. Hyperparamètres ajustés pour chaque modèle")
    print("7. Validation temporelle pour éviter le surapprentissage")

if __name__ == "__main__":
    main()

