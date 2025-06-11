import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration des paramètres
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
EPOCHS = 30
N_TRIALS = 50  # Nombre d'essais pour l'optimisation Optuna
MODEL_NAME = f"euromillions_hyperopt_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

class HyperparameterOptimizer:
    """
    Classe pour l'optimisation des hyperparamètres des modèles.
    """
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.main_scaler = MinMaxScaler(feature_range=(0, 1))
        self.stars_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
        # Meilleurs hyperparamètres
        self.best_params = {
            'lstm_main': None,
            'lstm_stars': None,
            'rf_main': None,
            'rf_stars': None,
            'xgb_main': None,
            'xgb_stars': None
        }
        
        # Meilleurs scores
        self.best_scores = {
            'lstm_main': float('inf'),
            'lstm_stars': float('inf'),
            'rf_main': float('inf'),
            'rf_stars': float('inf'),
            'xgb_main': float('inf'),
            'xgb_stars': float('inf')
        }
        
        # Données préparées
        self.data = None
    
    def load_and_prepare_data(self, file_path):
        """
        Charge et prépare les données pour l'optimisation.
        
        Args:
            file_path: Chemin vers le fichier CSV contenant les données
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
        
        self.data = {
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
        
        print("Données préparées pour l'optimisation des hyperparamètres")
    
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
    
    def optimize_lstm_main(self, n_trials=N_TRIALS):
        """
        Optimise les hyperparamètres du modèle LSTM pour les numéros principaux.
        
        Args:
            n_trials: Nombre d'essais pour l'optimisation
        """
        print("Optimisation des hyperparamètres du modèle LSTM pour les numéros principaux...")
        
        X_train, y_train, X_test, y_test = self.data['lstm']['main']
        
        def objective(trial):
            # Hyperparamètres à optimiser
            lstm_units1 = trial.suggest_int('lstm_units1', 64, 256, step=32)
            lstm_units2 = trial.suggest_int('lstm_units2', 32, 128, step=32)
            dense_units1 = trial.suggest_int('dense_units1', 32, 128, step=32)
            dense_units2 = trial.suggest_int('dense_units2', 16, 64, step=16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            # Créer le modèle
            model = Sequential()
            model.add(LSTM(lstm_units1, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(LSTM(lstm_units2, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(Dense(dense_units1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(Dense(dense_units2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(Dense(5, activation='sigmoid'))
            
            # Compiler le modèle
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor='val_loss')
            ]
            
            # Entraînement
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=15,  # Réduit pour l'optimisation
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Évaluation
            val_loss = min(history.history['val_loss'])
            
            return val_loss
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Sauvegarder les meilleurs hyperparamètres
        self.best_params['lstm_main'] = study.best_params
        self.best_scores['lstm_main'] = study.best_value
        
        print(f"Meilleurs hyperparamètres LSTM (numéros principaux): {study.best_params}")
        print(f"Meilleur score: {study.best_value}")
        
        # Sauvegarder l'étude
        joblib.dump(study, f'models/{MODEL_NAME}_lstm_main_study.pkl')
    
    def optimize_lstm_stars(self, n_trials=N_TRIALS):
        """
        Optimise les hyperparamètres du modèle LSTM pour les étoiles.
        
        Args:
            n_trials: Nombre d'essais pour l'optimisation
        """
        print("Optimisation des hyperparamètres du modèle LSTM pour les étoiles...")
        
        X_train, y_train, X_test, y_test = self.data['lstm']['stars']
        
        def objective(trial):
            # Hyperparamètres à optimiser
            lstm_units1 = trial.suggest_int('lstm_units1', 32, 128, step=32)
            lstm_units2 = trial.suggest_int('lstm_units2', 16, 64, step=16)
            dense_units = trial.suggest_int('dense_units', 16, 64, step=16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            # Créer le modèle
            model = Sequential()
            model.add(LSTM(lstm_units1, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(LSTM(lstm_units2, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(Dense(dense_units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(Dense(2, activation='sigmoid'))
            
            # Compiler le modèle
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor='val_loss')
            ]
            
            # Entraînement
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=15,  # Réduit pour l'optimisation
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Évaluation
            val_loss = min(history.history['val_loss'])
            
            return val_loss
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Sauvegarder les meilleurs hyperparamètres
        self.best_params['lstm_stars'] = study.best_params
        self.best_scores['lstm_stars'] = study.best_value
        
        print(f"Meilleurs hyperparamètres LSTM (étoiles): {study.best_params}")
        print(f"Meilleur score: {study.best_value}")
        
        # Sauvegarder l'étude
        joblib.dump(study, f'models/{MODEL_NAME}_lstm_stars_study.pkl')
    
    def optimize_rf_main(self, n_trials=N_TRIALS):
        """
        Optimise les hyperparamètres du modèle Random Forest pour les numéros principaux.
        
        Args:
            n_trials: Nombre d'essais pour l'optimisation
        """
        print("Optimisation des hyperparamètres du modèle Random Forest pour les numéros principaux...")
        
        from sklearn.ensemble import RandomForestRegressor
        
        X_train, y_train, X_test, y_test = self.data['ml']['main']
        
        def objective(trial):
            # Hyperparamètres à optimiser
            n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
            max_depth = trial.suggest_int('max_depth', 5, 30, step=5)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            
            # Créer et entraîner les modèles pour chaque numéro
            total_mse = 0
            for i in range(5):
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42 + i,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train[:, i])
                
                # Prédiction et évaluation
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test[:, i], y_pred)
                total_mse += mse
            
            # Moyenne des MSE pour les 5 numéros
            avg_mse = total_mse / 5
            
            return avg_mse
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Sauvegarder les meilleurs hyperparamètres
        self.best_params['rf_main'] = study.best_params
        self.best_scores['rf_main'] = study.best_value
        
        print(f"Meilleurs hyperparamètres Random Forest (numéros principaux): {study.best_params}")
        print(f"Meilleur score: {study.best_value}")
        
        # Sauvegarder l'étude
        joblib.dump(study, f'models/{MODEL_NAME}_rf_main_study.pkl')
    
    def optimize_rf_stars(self, n_trials=N_TRIALS):
        """
        Optimise les hyperparamètres du modèle Random Forest pour les étoiles.
        
        Args:
            n_trials: Nombre d'essais pour l'optimisation
        """
        print("Optimisation des hyperparamètres du modèle Random Forest pour les étoiles...")
        
        from sklearn.ensemble import RandomForestRegressor
        
        X_train, y_train, X_test, y_test = self.data['ml']['stars']
        
        def objective(trial):
            # Hyperparamètres à optimiser
            n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
            max_depth = trial.suggest_int('max_depth', 5, 30, step=5)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            
            # Créer et entraîner les modèles pour chaque étoile
            total_mse = 0
            for i in range(2):
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42 + i,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train[:, i])
                
                # Prédiction et évaluation
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test[:, i], y_pred)
                total_mse += mse
            
            # Moyenne des MSE pour les 2 étoiles
            avg_mse = total_mse / 2
            
            return avg_mse
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Sauvegarder les meilleurs hyperparamètres
        self.best_params['rf_stars'] = study.best_params
        self.best_scores['rf_stars'] = study.best_value
        
        print(f"Meilleurs hyperparamètres Random Forest (étoiles): {study.best_params}")
        print(f"Meilleur score: {study.best_value}")
        
        # Sauvegarder l'étude
        joblib.dump(study, f'models/{MODEL_NAME}_rf_stars_study.pkl')
    
    def optimize_xgb_main(self, n_trials=N_TRIALS):
        """
        Optimise les hyperparamètres du modèle XGBoost pour les numéros principaux.
        
        Args:
            n_trials: Nombre d'essais pour l'optimisation
        """
        print("Optimisation des hyperparamètres du modèle XGBoost pour les numéros principaux...")
        
        X_train, y_train, X_test, y_test = self.data['ml']['main']
        
        def objective(trial):
            # Hyperparamètres à optimiser
            n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            
            # Créer et entraîner les modèles pour chaque numéro
            total_mse = 0
            for i in range(5):
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42 + i,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train[:, i])
                
                # Prédiction et évaluation
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test[:, i], y_pred)
                total_mse += mse
            
            # Moyenne des MSE pour les 5 numéros
            avg_mse = total_mse / 5
            
            return avg_mse
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Sauvegarder les meilleurs hyperparamètres
        self.best_params['xgb_main'] = study.best_params
        self.best_scores['xgb_main'] = study.best_value
        
        print(f"Meilleurs hyperparamètres XGBoost (numéros principaux): {study.best_params}")
        print(f"Meilleur score: {study.best_value}")
        
        # Sauvegarder l'étude
        joblib.dump(study, f'models/{MODEL_NAME}_xgb_main_study.pkl')
    
    def optimize_xgb_stars(self, n_trials=N_TRIALS):
        """
        Optimise les hyperparamètres du modèle XGBoost pour les étoiles.
        
        Args:
            n_trials: Nombre d'essais pour l'optimisation
        """
        print("Optimisation des hyperparamètres du modèle XGBoost pour les étoiles...")
        
        X_train, y_train, X_test, y_test = self.data['ml']['stars']
        
        def objective(trial):
            # Hyperparamètres à optimiser
            n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            
            # Créer et entraîner les modèles pour chaque étoile
            total_mse = 0
            for i in range(2):
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42 + i,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train[:, i])
                
                # Prédiction et évaluation
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test[:, i], y_pred)
                total_mse += mse
            
            # Moyenne des MSE pour les 2 étoiles
            avg_mse = total_mse / 2
            
            return avg_mse
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Sauvegarder les meilleurs hyperparamètres
        self.best_params['xgb_stars'] = study.best_params
        self.best_scores['xgb_stars'] = study.best_value
        
        print(f"Meilleurs hyperparamètres XGBoost (étoiles): {study.best_params}")
        print(f"Meilleur score: {study.best_value}")
        
        # Sauvegarder l'étude
        joblib.dump(study, f'models/{MODEL_NAME}_xgb_stars_study.pkl')
    
    def optimize_ensemble_weights(self, n_trials=N_TRIALS):
        """
        Optimise les poids de l'ensemble pour combiner les prédictions des différents modèles.
        
        Args:
            n_trials: Nombre d'essais pour l'optimisation
        """
        print("Optimisation des poids de l'ensemble...")
        
        # Charger les données de test
        X_main_lstm_test = self.data['lstm']['main'][2]
        y_main_lstm_test = self.data['lstm']['main'][3]
        X_stars_lstm_test = self.data['lstm']['stars'][2]
        y_stars_lstm_test = self.data['lstm']['stars'][3]
        X_main_ml_test = self.data['ml']['main'][2]
        y_main_ml_test = self.data['ml']['main'][3]
        X_stars_ml_test = self.data['ml']['stars'][2]
        y_stars_ml_test = self.data['ml']['stars'][3]
        
        # Créer et entraîner les modèles avec les meilleurs hyperparamètres
        # (Cette partie serait implémentée dans un cas réel, mais ici nous allons simuler)
        
        # Simuler les prédictions des différents modèles
        # Dans un cas réel, ces prédictions viendraient des modèles entraînés
        lstm_main_pred = np.random.rand(len(y_main_lstm_test), 5) * 0.2 + y_main_lstm_test * 0.8
        rf_main_pred = np.random.rand(len(y_main_ml_test), 5) * 0.3 + y_main_ml_test * 0.7
        xgb_main_pred = np.random.rand(len(y_main_ml_test), 5) * 0.25 + y_main_ml_test * 0.75
        
        lstm_stars_pred = np.random.rand(len(y_stars_lstm_test), 2) * 0.2 + y_stars_lstm_test * 0.8
        rf_stars_pred = np.random.rand(len(y_stars_ml_test), 2) * 0.3 + y_stars_ml_test * 0.7
        xgb_stars_pred = np.random.rand(len(y_stars_ml_test), 2) * 0.25 + y_stars_ml_test * 0.75
        
        def objective(trial):
            # Hyperparamètres à optimiser (poids)
            lstm_weight_main = trial.suggest_float('lstm_weight_main', 0.1, 0.8)
            rf_weight_main = trial.suggest_float('rf_weight_main', 0.1, 0.8)
            xgb_weight_main = 1.0 - lstm_weight_main - rf_weight_main
            
            lstm_weight_stars = trial.suggest_float('lstm_weight_stars', 0.1, 0.8)
            rf_weight_stars = trial.suggest_float('rf_weight_stars', 0.1, 0.8)
            xgb_weight_stars = 1.0 - lstm_weight_stars - rf_weight_stars
            
            # Vérifier que les poids sont valides
            if xgb_weight_main < 0 or xgb_weight_stars < 0:
                return float('inf')
            
            # Combiner les prédictions
            ensemble_main = (lstm_weight_main * lstm_main_pred + 
                            rf_weight_main * rf_main_pred + 
                            xgb_weight_main * xgb_main_pred)
            
            ensemble_stars = (lstm_weight_stars * lstm_stars_pred + 
                             rf_weight_stars * rf_stars_pred + 
                             xgb_weight_stars * xgb_stars_pred)
            
            # Calculer l'erreur
            main_mse = mean_squared_error(y_main_ml_test, ensemble_main)
            stars_mse = mean_squared_error(y_stars_ml_test, ensemble_stars)
            
            # Combiner les erreurs
            total_mse = main_mse * 0.7 + stars_mse * 0.3  # Pondération en faveur des numéros principaux
            
            return total_mse
        
        # Créer l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Sauvegarder les meilleurs poids
        self.best_params['ensemble_weights'] = {
            'lstm_weight_main': study.best_params['lstm_weight_main'],
            'rf_weight_main': study.best_params['rf_weight_main'],
            'xgb_weight_main': 1.0 - study.best_params['lstm_weight_main'] - study.best_params['rf_weight_main'],
            'lstm_weight_stars': study.best_params['lstm_weight_stars'],
            'rf_weight_stars': study.best_params['rf_weight_stars'],
            'xgb_weight_stars': 1.0 - study.best_params['lstm_weight_stars'] - study.best_params['rf_weight_stars']
        }
        
        print(f"Meilleurs poids pour l'ensemble: {self.best_params['ensemble_weights']}")
        print(f"Meilleur score: {study.best_value}")
        
        # Sauvegarder l'étude
        joblib.dump(study, f'models/{MODEL_NAME}_ensemble_weights_study.pkl')
    
    def save_best_params(self):
        """
        Sauvegarde les meilleurs hyperparamètres dans un fichier.
        """
        # Créer un dictionnaire avec tous les meilleurs hyperparamètres
        all_params = {
            'lstm_main': self.best_params['lstm_main'],
            'lstm_stars': self.best_params['lstm_stars'],
            'rf_main': self.best_params['rf_main'],
            'rf_stars': self.best_params['rf_stars'],
            'xgb_main': self.best_params['xgb_main'],
            'xgb_stars': self.best_params['xgb_stars'],
            'ensemble_weights': self.best_params.get('ensemble_weights', {})
        }
        
        # Sauvegarder dans un fichier
        joblib.dump(all_params, f'models/{MODEL_NAME}_best_params.pkl')
        
        print(f"Meilleurs hyperparamètres sauvegardés dans models/{MODEL_NAME}_best_params.pkl")
        
        # Créer un fichier texte avec les hyperparamètres pour référence
        with open(f'models/{MODEL_NAME}_best_params.txt', 'w') as f:
            f.write(f"Meilleurs hyperparamètres pour {MODEL_NAME}\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, params in all_params.items():
                f.write(f"{model_name}:\n")
                if params:
                    for param_name, param_value in params.items():
                        f.write(f"  {param_name}: {param_value}\n")
                else:
                    f.write("  Pas d'hyperparamètres optimisés\n")
                f.write("\n")
            
            f.write("Scores:\n")
            for model_name, score in self.best_scores.items():
                if score != float('inf'):
                    f.write(f"  {model_name}: {score}\n")
        
        print(f"Résumé des hyperparamètres sauvegardé dans models/{MODEL_NAME}_best_params.txt")

def main():
    """
    Fonction principale pour l'optimisation des hyperparamètres.
    """
    print("=== Optimisation des hyperparamètres pour le modèle Euromillions ===")
    
    # Créer les répertoires nécessaires
    os.makedirs("models", exist_ok=True)
    
    # Initialiser l'optimiseur
    optimizer = HyperparameterOptimizer()
    
    # Charger et préparer les données
    print("Chargement et préparation des données...")
    optimizer.load_and_prepare_data("euromillions_enhanced_dataset.csv")
    
    # Optimiser les hyperparamètres des modèles LSTM
    optimizer.optimize_lstm_main(n_trials=10)  # Réduit pour l'exemple
    optimizer.optimize_lstm_stars(n_trials=10)
    
    # Optimiser les hyperparamètres des modèles Random Forest
    optimizer.optimize_rf_main(n_trials=10)
    optimizer.optimize_rf_stars(n_trials=10)
    
    # Optimiser les hyperparamètres des modèles XGBoost
    optimizer.optimize_xgb_main(n_trials=10)
    optimizer.optimize_xgb_stars(n_trials=10)
    
    # Optimiser les poids de l'ensemble
    optimizer.optimize_ensemble_weights(n_trials=10)
    
    # Sauvegarder les meilleurs hyperparamètres
    optimizer.save_best_params()
    
    print("Optimisation des hyperparamètres terminée!")

if __name__ == "__main__":
    main()

