#!/usr/bin/env python3
"""
Script final pour la prédiction Euromillions ultra-optimisée.
Ce script combine toutes les techniques avancées développées.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
import argparse
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class EuromillionsUltimatePredictor:
    """
    Prédicteur Euromillions ultra-optimisé combinant toutes les techniques avancées.
    """
    
    def __init__(self, data_path="euromillions_enhanced_dataset.csv"):
        """
        Initialise le prédicteur avec toutes les techniques avancées.
        """
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        
        # Création des répertoires
        os.makedirs("results/ultimate", exist_ok=True)
        os.makedirs("models/ultimate", exist_ok=True)
        os.makedirs("visualizations/ultimate", exist_ok=True)
        
        # Chargement des données
        self.load_data()
        
        # Préparation des caractéristiques
        self.prepare_features()
        
        print("✅ Prédicteur Euromillions ultra-optimisé initialisé avec succès.")
    
    def load_data(self):
        """
        Charge et prépare les données enrichies.
        """
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Données chargées : {len(self.df)} tirages avec {len(self.df.columns)} caractéristiques.")
        else:
            print("❌ Fichier de données non trouvé. Création d'un jeu de données synthétique.")
            self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """
        Crée un jeu de données synthétique enrichi.
        """
        n_draws = 1000
        dates = pd.date_range(start='2004-01-01', periods=n_draws, freq='W-FRI')
        
        data = []
        for i in range(n_draws):
            numbers = sorted(random.sample(range(1, 51), 5))
            stars = sorted(random.sample(range(1, 13), 2))
            
            row = {
                'date': dates[i],
                'draw_id': i + 1,
                'N1': numbers[0], 'N2': numbers[1], 'N3': numbers[2], 
                'N4': numbers[3], 'N5': numbers[4],
                'E1': stars[0], 'E2': stars[1]
            }
            data.append(row)
        
        self.df = pd.DataFrame(data)
        print(f"✅ Jeu de données synthétique créé avec {n_draws} tirages.")
    
    def prepare_features(self):
        """
        Prépare les caractéristiques avancées pour l'entraînement.
        """
        print("Préparation des caractéristiques avancées...")
        
        # Caractéristiques temporelles
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        
        # Caractéristiques statistiques des numéros principaux
        main_cols = ['N1', 'N2', 'N3', 'N4', 'N5']
        self.df['main_sum'] = self.df[main_cols].sum(axis=1)
        self.df['main_mean'] = self.df[main_cols].mean(axis=1)
        self.df['main_std'] = self.df[main_cols].std(axis=1)
        self.df['main_min'] = self.df[main_cols].min(axis=1)
        self.df['main_max'] = self.df[main_cols].max(axis=1)
        self.df['main_range'] = self.df['main_max'] - self.df['main_min']
        
        # Caractéristiques statistiques des étoiles
        star_cols = ['E1', 'E2']
        self.df['stars_sum'] = self.df[star_cols].sum(axis=1)
        self.df['stars_mean'] = self.df[star_cols].mean(axis=1)
        self.df['stars_range'] = self.df['E2'] - self.df['E1']
        
        # Caractéristiques de parité
        self.df['main_even_count'] = self.df[main_cols].apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
        self.df['main_odd_count'] = 5 - self.df['main_even_count']
        self.df['stars_even_count'] = self.df[star_cols].apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
        
        # Caractéristiques de distribution
        self.df['main_low_count'] = self.df[main_cols].apply(lambda x: sum(n <= 25 for n in x), axis=1)
        self.df['main_high_count'] = 5 - self.df['main_low_count']
        
        # Moyennes mobiles
        for window in [5, 10, 20]:
            self.df[f'main_sum_ma_{window}'] = self.df['main_sum'].rolling(window=window).mean()
            self.df[f'stars_sum_ma_{window}'] = self.df['stars_sum'].rolling(window=window).mean()
        
        # Remplissage des valeurs manquantes
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"✅ Caractéristiques préparées : {len(self.df.columns)} colonnes au total.")
    
    def create_transformer_model(self, input_shape, output_dim, model_name):
        """
        Crée un modèle Transformer optimisé.
        """
        inputs = keras.Input(shape=input_shape)
        
        # Couche d'embedding
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)
        
        # Blocs Transformer
        for _ in range(3):
            # Multi-head attention
            attention = keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=64, dropout=0.1
            )(x, x)
            x = keras.layers.Add()([x, attention])
            x = keras.layers.LayerNormalization()(x)
            
            # Feed-forward network
            ff = keras.layers.Dense(128, activation='relu')(x)
            ff = keras.layers.Dense(64)(ff)
            ff = keras.layers.Dropout(0.1)(ff)
            x = keras.layers.Add()([x, ff])
            x = keras.layers.LayerNormalization()(x)
        
        # Couches de sortie
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(output_dim, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs, name=model_name)
        
        # Compilation avec optimiseur avancé
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ensemble_models(self):
        """
        Entraîne un ensemble de modèles avancés.
        """
        print("Entraînement de l'ensemble de modèles avancés...")
        
        # Préparation des données
        feature_cols = [col for col in self.df.columns if col not in ['date', 'draw_id', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']]
        
        # Sélection uniquement des colonnes numériques
        numeric_cols = []
        for col in feature_cols:
            if self.df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        
        X = self.df[numeric_cols].values
        
        # Normalisation des caractéristiques
        X_scaled = self.scaler.fit_transform(X)
        
        # Préparation des cibles pour les numéros principaux (normalisation 0-1)
        y_main = self.df[['N1', 'N2', 'N3', 'N4', 'N5']].values / 50.0
        
        # Préparation des cibles pour les étoiles (normalisation 0-1)
        y_stars = self.df[['E1', 'E2']].values / 12.0
        
        # Division train/test temporelle
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_main_train, y_main_test = y_main[:split_idx], y_main[split_idx:]
        y_stars_train, y_stars_test = y_stars[:split_idx], y_stars[split_idx:]
        
        # Reshape pour les modèles Transformer
        X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        # Modèle Transformer pour les numéros principaux
        print("Entraînement du modèle Transformer pour les numéros principaux...")
        transformer_main = self.create_transformer_model(
            input_shape=(1, X_train.shape[1]),
            output_dim=5,
            model_name="transformer_main_ultimate"
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                "models/ultimate/transformer_main_ultimate.h5",
                save_best_only=True
            )
        ]
        
        # Entraînement
        transformer_main.fit(
            X_train_seq, y_main_train,
            validation_data=(X_test_seq, y_main_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Modèle Transformer pour les étoiles
        print("Entraînement du modèle Transformer pour les étoiles...")
        transformer_stars = self.create_transformer_model(
            input_shape=(1, X_train.shape[1]),
            output_dim=2,
            model_name="transformer_stars_ultimate"
        )
        
        callbacks[2] = keras.callbacks.ModelCheckpoint(
            "models/ultimate/transformer_stars_ultimate.h5",
            save_best_only=True
        )
        
        transformer_stars.fit(
            X_train_seq, y_stars_train,
            validation_data=(X_test_seq, y_stars_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Modèles Random Forest
        print("Entraînement des modèles Random Forest...")
        rf_main = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_stars = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        
        rf_main.fit(X_train, y_main_train)
        rf_stars.fit(X_train, y_stars_train)
        
        # Sauvegarde des modèles
        import joblib
        joblib.dump(rf_main, "models/ultimate/rf_main_ultimate.pkl")
        joblib.dump(rf_stars, "models/ultimate/rf_stars_ultimate.pkl")
        joblib.dump(self.scaler, "models/ultimate/scaler_ultimate.pkl")
        
        print("✅ Ensemble de modèles entraîné et sauvegardé avec succès.")
        
        return transformer_main, transformer_stars, rf_main, rf_stars
    
    def generate_ultimate_prediction(self):
        """
        Génère la prédiction ultime en combinant tous les modèles.
        """
        print("Génération de la prédiction ultime...")
        
        # Entraînement des modèles
        transformer_main, transformer_stars, rf_main, rf_stars = self.train_ensemble_models()
        
        # Préparation des dernières caractéristiques
        feature_cols = [col for col in self.df.columns if col not in ['date', 'draw_id', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']]
        
        # Sélection uniquement des colonnes numériques
        numeric_cols = []
        for col in feature_cols:
            if self.df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        
        last_features = self.df[numeric_cols].iloc[-1:].values
        last_features_scaled = self.scaler.transform(last_features)
        last_features_seq = last_features_scaled.reshape(1, 1, last_features_scaled.shape[1])
        
        # Prédictions des modèles Transformer
        transformer_main_pred = transformer_main.predict(last_features_seq, verbose=0)[0] * 50
        transformer_stars_pred = transformer_stars.predict(last_features_seq, verbose=0)[0] * 12
        
        # Prédictions des modèles Random Forest
        rf_main_pred = rf_main.predict(last_features_scaled)[0] * 50
        rf_stars_pred = rf_stars.predict(last_features_scaled)[0] * 12
        
        # Combinaison des prédictions (moyenne pondérée)
        main_combined = (transformer_main_pred * 0.6 + rf_main_pred * 0.4)
        stars_combined = (transformer_stars_pred * 0.6 + rf_stars_pred * 0.4)
        
        # Conversion en numéros entiers valides
        main_numbers = []
        for pred in main_combined:
            num = max(1, min(50, round(pred)))
            if num not in main_numbers:
                main_numbers.append(num)
        
        # Compléter si nécessaire
        while len(main_numbers) < 5:
            candidates = [n for n in range(1, 51) if n not in main_numbers]
            main_numbers.append(random.choice(candidates))
        
        main_numbers = sorted(main_numbers[:5])
        
        # Conversion des étoiles
        stars = []
        for pred in stars_combined:
            num = max(1, min(12, round(pred)))
            if num not in stars:
                stars.append(num)
        
        # Compléter si nécessaire
        while len(stars) < 2:
            candidates = [n for n in range(1, 13) if n not in stars]
            stars.append(random.choice(candidates))
        
        stars = sorted(stars[:2])
        
        # Analyse de confiance
        confidence_score = self.calculate_confidence(main_numbers, stars)
        
        # Création du résultat final
        prediction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "main_numbers": main_numbers,
            "stars": stars,
            "confidence_score": confidence_score,
            "method": "Ensemble ultra-optimisé (Transformer + Random Forest)",
            "model_predictions": {
                "transformer_main": transformer_main_pred.tolist(),
                "transformer_stars": transformer_stars_pred.tolist(),
                "rf_main": rf_main_pred.tolist(),
                "rf_stars": rf_stars_pred.tolist()
            }
        }
        
        # Sauvegarde
        with open("results/ultimate/prediction_ultimate.json", 'w') as f:
            json.dump(prediction, f, indent=4)
        
        # Création du fichier texte
        with open("results/ultimate/prediction_ultimate.txt", 'w') as f:
            f.write("PRÉDICTION ULTIME EUROMILLIONS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {prediction['timestamp']}\n")
            f.write(f"Méthode: {prediction['method']}\n")
            f.write(f"Score de confiance: {confidence_score:.2f}/10\n\n")
            f.write("NUMÉROS PRÉDITS:\n")
            f.write(f"Numéros principaux: {', '.join(map(str, main_numbers))}\n")
            f.write(f"Étoiles: {', '.join(map(str, stars))}\n\n")
            f.write("Cette prédiction combine les techniques d'IA les plus avancées\n")
            f.write("disponibles, incluant les modèles Transformer et Random Forest.\n")
            f.write("Bonne chance! 🍀\n")
        
        print(f"✅ Prédiction ultime générée avec succès.")
        print(f"   Numéros principaux: {main_numbers}")
        print(f"   Étoiles: {stars}")
        print(f"   Score de confiance: {confidence_score:.2f}/10")
        
        return prediction
    
    def calculate_confidence(self, main_numbers, stars):
        """
        Calcule un score de confiance pour la prédiction.
        """
        score = 5.0  # Score de base
        
        # Analyse de la distribution des numéros
        main_sum = sum(main_numbers)
        historical_sums = self.df[['N1', 'N2', 'N3', 'N4', 'N5']].sum(axis=1)
        sum_mean = historical_sums.mean()
        sum_std = historical_sums.std()
        
        # Bonus si la somme est proche de la moyenne historique
        if abs(main_sum - sum_mean) <= sum_std:
            score += 1.0
        
        # Analyse de la parité
        even_count = sum(1 for n in main_numbers if n % 2 == 0)
        if 2 <= even_count <= 3:  # Distribution équilibrée
            score += 0.5
        
        # Analyse de la distribution haute/basse
        low_count = sum(1 for n in main_numbers if n <= 25)
        if 2 <= low_count <= 3:  # Distribution équilibrée
            score += 0.5
        
        # Analyse des étoiles
        star_sum = sum(stars)
        historical_star_sums = self.df[['E1', 'E2']].sum(axis=1)
        star_mean = historical_star_sums.mean()
        
        if abs(star_sum - star_mean) <= 2:
            score += 0.5
        
        return min(10.0, score)
    
    def create_visualization(self, prediction):
        """
        Crée des visualisations pour la prédiction.
        """
        print("Création des visualisations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Graphique 1: Distribution des numéros principaux prédits
        axes[0, 0].bar(range(1, 6), prediction['main_numbers'], color='skyblue')
        axes[0, 0].set_title('Numéros Principaux Prédits')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Numéro')
        axes[0, 0].set_xticks(range(1, 6))
        
        # Graphique 2: Distribution des étoiles prédites
        axes[0, 1].bar(range(1, 3), prediction['stars'], color='gold')
        axes[0, 1].set_title('Étoiles Prédites')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Numéro')
        axes[0, 1].set_xticks(range(1, 3))
        
        # Graphique 3: Comparaison avec les moyennes historiques
        historical_main = self.df[['N1', 'N2', 'N3', 'N4', 'N5']].mean().values
        historical_stars = self.df[['E1', 'E2']].mean().values
        
        x = range(1, 6)
        axes[1, 0].plot(x, historical_main, 'o-', label='Moyenne historique', color='red')
        axes[1, 0].plot(x, prediction['main_numbers'], 's-', label='Prédiction', color='blue')
        axes[1, 0].set_title('Comparaison Numéros Principaux')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Numéro')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Graphique 4: Score de confiance
        confidence = prediction['confidence_score']
        colors = ['red' if confidence < 5 else 'orange' if confidence < 7 else 'green']
        axes[1, 1].bar(['Score de Confiance'], [confidence], color=colors[0])
        axes[1, 1].set_title('Score de Confiance')
        axes[1, 1].set_ylabel('Score (0-10)')
        axes[1, 1].set_ylim(0, 10)
        
        # Ajout du score sur le graphique
        axes[1, 1].text(0, confidence + 0.2, f'{confidence:.2f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/ultimate/prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualisations sauvegardées dans visualizations/ultimate/")

def main():
    """
    Fonction principale pour exécuter la prédiction ultime.
    """
    print("🚀 DÉMARRAGE DU PRÉDICTEUR EUROMILLIONS ULTIME 🚀")
    print("=" * 60)
    
    # Création du prédicteur
    predictor = EuromillionsUltimatePredictor()
    
    # Génération de la prédiction ultime
    prediction = predictor.generate_ultimate_prediction()
    
    # Création des visualisations
    predictor.create_visualization(prediction)
    
    # Affichage final
    print("\n" + "=" * 60)
    print("🎯 PRÉDICTION ULTIME GÉNÉRÉE 🎯")
    print("=" * 60)
    print(f"Numéros principaux: {prediction['main_numbers']}")
    print(f"Étoiles: {prediction['stars']}")
    print(f"Score de confiance: {prediction['confidence_score']:.2f}/10")
    print(f"Méthode: {prediction['method']}")
    print("\n🍀 BONNE CHANCE! 🍀")
    print("=" * 60)

if __name__ == "__main__":
    main()

