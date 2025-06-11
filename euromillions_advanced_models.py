import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime
import random
import time

# Création des répertoires pour les modèles et les résultats
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

class EuromillionsAdvancedModels:
    """
    Classe pour développer et entraîner des modèles d'IA avancés
    pour la prédiction des numéros de l'Euromillions.
    """
    
    def __init__(self, data_path="data_enriched/euromillions_enhanced_dataset.csv"):
        """
        Initialise la classe avec le chemin vers les données enrichies.
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train_main = None
        self.y_val_main = None
        self.y_test_main = None
        self.y_train_stars = None
        self.y_val_stars = None
        self.y_test_stars = None
        self.main_scaler = None
        self.stars_scaler = None
        self.feature_scaler = None
        self.models = {}
        self.predictions = {}
        self.results = {}
        
        # Paramètres pour les modèles
        self.sequence_length = 10  # Nombre de tirages précédents à considérer
        self.batch_size = 32
        self.epochs = 100
        self.patience = 20  # Pour l'early stopping
        
        # Vérification de la disponibilité des données
        if not os.path.exists(self.data_path):
            print(f"❌ Fichier de données {self.data_path} non trouvé.")
            print("⚠️ Utilisation d'un jeu de données synthétique pour le développement des modèles.")
            self.create_synthetic_dataset()
        else:
            print(f"✅ Fichier de données {self.data_path} trouvé.")
    
    def create_synthetic_dataset(self):
        """
        Crée un jeu de données synthétique pour le développement des modèles
        en attendant que les données réelles soient disponibles.
        """
        print("Création d'un jeu de données synthétique...")
        
        # Nombre de tirages synthétiques
        n_draws = 1000
        
        # Création d'un DataFrame avec des dates
        dates = pd.date_range(start='2004-01-01', periods=n_draws, freq='W-FRI')
        
        # Initialisation du DataFrame
        data = []
        
        # Génération des tirages synthétiques
        for i in range(n_draws):
            # Numéros principaux (1-50)
            numbers = sorted(random.sample(range(1, 51), 5))
            
            # Étoiles (1-12)
            stars = sorted(random.sample(range(1, 13), 2))
            
            # Création d'une ligne de données
            row = {
                'date': dates[i],
                'draw_id': i + 1,
                'has_winner': random.choice([True, False]),
                'N1': numbers[0],
                'N2': numbers[1],
                'N3': numbers[2],
                'N4': numbers[3],
                'N5': numbers[4],
                'E1': stars[0],
                'E2': stars[1],
                'jackpot': random.uniform(17000000, 200000000)
            }
            
            # Ajout de caractéristiques synthétiques
            row['Main_Sum'] = sum(numbers)
            row['Main_Mean'] = sum(numbers) / 5
            row['Main_Std'] = np.std(numbers)
            row['Stars_Sum'] = sum(stars)
            row['Year'] = dates[i].year
            row['Month'] = dates[i].month
            row['DayOfWeek'] = dates[i].dayofweek
            
            # Ajout de caractéristiques de fréquence synthétiques
            for num in range(1, 51):
                row[f'N{num}_Freq_50'] = random.uniform(0, 0.2)
                row[f'N{num}_LastSeen'] = random.randint(0, 50)
            
            for num in range(1, 13):
                row[f'E{num}_Freq_50'] = random.uniform(0, 0.3)
                row[f'E{num}_LastSeen'] = random.randint(0, 30)
            
            # Ajout de la ligne au tableau de données
            data.append(row)
        
        # Création du DataFrame
        self.df = pd.DataFrame(data)
        
        # Sauvegarde du jeu de données synthétique
        os.makedirs("data_enriched", exist_ok=True)
        self.df.to_csv("data_enriched/synthetic_euromillions_dataset.csv", index=False)
        
        print(f"✅ Jeu de données synthétique créé avec {n_draws} tirages.")
        
        # Mise à jour du chemin des données
        self.data_path = "data_enriched/synthetic_euromillions_dataset.csv"
    
    def load_data(self):
        """
        Charge les données enrichies.
        """
        print(f"Chargement des données depuis {self.data_path}...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Conversion de la colonne date en datetime
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            print(f"✅ Données chargées avec succès : {len(self.df)} lignes et {len(self.df.columns)} colonnes.")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            return False
    
    def prepare_data_for_transformer(self):
        """
        Prépare les données pour l'entraînement d'un modèle Transformer.
        """
        print("Préparation des données pour le modèle Transformer...")
        
        # Sélection des caractéristiques pertinentes
        feature_columns = [col for col in self.df.columns if col not in [
            'date', 'draw_id', 'has_winner', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2'
        ]]
        
        # Normalisation des caractéristiques
        self.feature_scaler = StandardScaler()
        features = self.feature_scaler.fit_transform(self.df[feature_columns])
        
        # Création des séquences pour le Transformer
        X = []
        y_main = []
        y_stars = []
        
        for i in range(self.sequence_length, len(self.df)):
            # Séquence de caractéristiques
            X.append(features[i-self.sequence_length:i])
            
            # Cibles : numéros principaux et étoiles
            y_main.append([
                self.df.iloc[i]['N1'],
                self.df.iloc[i]['N2'],
                self.df.iloc[i]['N3'],
                self.df.iloc[i]['N4'],
                self.df.iloc[i]['N5']
            ])
            
            y_stars.append([
                self.df.iloc[i]['E1'],
                self.df.iloc[i]['E2']
            ])
        
        # Conversion en arrays numpy
        X = np.array(X)
        y_main = np.array(y_main)
        y_stars = np.array(y_stars)
        
        # Normalisation des cibles
        self.main_scaler = MinMaxScaler(feature_range=(0, 1))
        self.stars_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Reshape pour la normalisation
        y_main_reshaped = y_main.reshape(-1, 5)
        y_stars_reshaped = y_stars.reshape(-1, 2)
        
        y_main_scaled = self.main_scaler.fit_transform(y_main_reshaped)
        y_stars_scaled = self.stars_scaler.fit_transform(y_stars_reshaped)
        
        # Reshape pour revenir à la forme originale
        y_main_scaled = y_main_scaled.reshape(-1, 5)
        y_stars_scaled = y_stars_scaled.reshape(-1, 2)
        
        # Division en ensembles d'entraînement, de validation et de test
        # 70% entraînement, 15% validation, 15% test
        X_train_val, self.X_test, y_main_train_val, self.y_test_main, y_stars_train_val, self.y_test_stars = train_test_split(
            X, y_main_scaled, y_stars_scaled, test_size=0.15, shuffle=False
        )
        
        self.X_train, self.X_val, self.y_train_main, self.y_val_main, self.y_train_stars, self.y_val_stars = train_test_split(
            X_train_val, y_main_train_val, y_stars_train_val, test_size=0.15/0.85, shuffle=False
        )
        
        print(f"✅ Données préparées pour le modèle Transformer :")
        print(f"   - Ensemble d'entraînement : {len(self.X_train)} séquences")
        print(f"   - Ensemble de validation : {len(self.X_val)} séquences")
        print(f"   - Ensemble de test : {len(self.X_test)} séquences")
        
        return True
    
    def build_transformer_model(self, name="transformer_main"):
        """
        Construit un modèle Transformer pour la prédiction des numéros.
        """
        print(f"Construction du modèle Transformer {name}...")
        
        # Détermination de la taille de sortie (5 pour les numéros principaux, 2 pour les étoiles)
        output_size = 5 if "main" in name else 2
        
        # Paramètres du modèle
        input_shape = self.X_train.shape[1:]
        d_model = 64  # Dimension du modèle
        num_heads = 4  # Nombre de têtes d'attention
        ff_dim = 128  # Dimension du feed-forward network
        num_transformer_blocks = 2  # Nombre de blocs Transformer
        dropout_rate = 0.1  # Taux de dropout
        
        # Construction du modèle
        inputs = keras.Input(shape=input_shape)
        
        # Couche d'embedding pour transformer les caractéristiques en vecteurs de dimension d_model
        x = layers.Dense(d_model)(inputs)
        
        # Ajout de l'encodage positionnel
        position_embedding = keras.layers.Embedding(
            input_dim=input_shape[0], output_dim=d_model
        )(tf.range(start=0, limit=input_shape[0], delta=1))
        
        x = x + position_embedding
        
        # Blocs Transformer
        for _ in range(num_transformer_blocks):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads
            )(x, x)
            
            # Skip connection et normalisation
            x = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn_output = layers.Dense(ff_dim, activation="relu")(x)
            ffn_output = layers.Dense(d_model)(ffn_output)
            
            # Skip connection et normalisation
            x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)
            
            # Dropout
            x = layers.Dropout(dropout_rate)(x)
        
        # Global average pooling pour obtenir un vecteur de caractéristiques
        x = layers.GlobalAveragePooling1D()(x)
        
        # Couches denses finales
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        
        # Couche de sortie
        outputs = layers.Dense(output_size, activation="sigmoid")(x)
        
        # Création du modèle
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilation du modèle
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        # Résumé du modèle
        model.summary()
        
        # Sauvegarde du modèle dans le dictionnaire
        self.models[name] = model
        
        print(f"✅ Modèle Transformer {name} construit avec succès.")
        
        return model
    
    def build_gan_model(self, name="gan_main"):
        """
        Construit un modèle GAN pour la génération de numéros.
        """
        print(f"Construction du modèle GAN {name}...")
        
        # Détermination de la taille de sortie (5 pour les numéros principaux, 2 pour les étoiles)
        output_size = 5 if "main" in name else 2
        
        # Paramètres du modèle
        input_shape = self.X_train.shape[1:]
        latent_dim = 128  # Dimension de l'espace latent
        
        # Construction du générateur
        generator_input = keras.Input(shape=(latent_dim,))
        
        x = layers.Dense(256, activation="relu")(generator_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        generator_output = layers.Dense(output_size, activation="sigmoid")(x)
        
        generator = keras.Model(generator_input, generator_output)
        
        # Construction du discriminateur
        discriminator_input = keras.Input(shape=(output_size,))
        
        x = layers.Dense(256, activation="relu")(discriminator_input)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        
        discriminator_output = layers.Dense(1, activation="sigmoid")(x)
        
        discriminator = keras.Model(discriminator_input, discriminator_output)
        
        # Compilation du discriminateur
        discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Construction du GAN complet
        discriminator.trainable = False
        
        gan_input = keras.Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        
        gan = keras.Model(gan_input, gan_output)
        
        # Compilation du GAN
        gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss="binary_crossentropy"
        )
        
        # Résumé des modèles
        generator.summary()
        discriminator.summary()
        gan.summary()
        
        # Sauvegarde des modèles dans le dictionnaire
        self.models[f"{name}_generator"] = generator
        self.models[f"{name}_discriminator"] = discriminator
        self.models[name] = gan
        
        print(f"✅ Modèle GAN {name} construit avec succès.")
        
        return generator, discriminator, gan
    
    def build_lstm_transformer_model(self, name="lstm_transformer_main"):
        """
        Construit un modèle hybride LSTM-Transformer pour la prédiction des numéros.
        """
        print(f"Construction du modèle hybride LSTM-Transformer {name}...")
        
        # Détermination de la taille de sortie (5 pour les numéros principaux, 2 pour les étoiles)
        output_size = 5 if "main" in name else 2
        
        # Paramètres du modèle
        input_shape = self.X_train.shape[1:]
        d_model = 64  # Dimension du modèle
        num_heads = 4  # Nombre de têtes d'attention
        ff_dim = 128  # Dimension du feed-forward network
        
        # Construction du modèle
        inputs = keras.Input(shape=input_shape)
        
        # Couche LSTM bidirectionnelle
        x = layers.Bidirectional(layers.LSTM(d_model, return_sequences=True))(inputs)
        
        # Couche d'attention multi-têtes
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        
        # Normalisation et skip connection
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        
        # Normalisation et skip connection
        x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Couches denses finales
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        
        # Couche de sortie
        outputs = layers.Dense(output_size, activation="sigmoid")(x)
        
        # Création du modèle
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilation du modèle
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        # Résumé du modèle
        model.summary()
        
        # Sauvegarde du modèle dans le dictionnaire
        self.models[name] = model
        
        print(f"✅ Modèle hybride LSTM-Transformer {name} construit avec succès.")
        
        return model
    
    def train_transformer_model(self, name="transformer_main"):
        """
        Entraîne un modèle Transformer pour la prédiction des numéros.
        """
        print(f"Entraînement du modèle Transformer {name}...")
        
        # Vérification de l'existence du modèle
        if name not in self.models:
            print(f"❌ Modèle {name} non trouvé. Construction du modèle...")
            self.build_transformer_model(name)
        
        # Sélection des données d'entraînement en fonction du type de modèle
        if "main" in name:
            y_train = self.y_train_main
            y_val = self.y_val_main
        else:
            y_train = self.y_train_stars
            y_val = self.y_val_stars
        
        # Callbacks pour l'entraînement
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f"models/{name}_best.h5",
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entraînement du modèle
        history = self.models[name].fit(
            self.X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarde de l'historique d'entraînement
        self.results[f"{name}_history"] = history.history
        
        # Sauvegarde du modèle final
        self.models[name].save(f"models/{name}_final.h5")
        
        # Visualisation de l'historique d'entraînement
        self.plot_training_history(history, name)
        
        print(f"✅ Modèle Transformer {name} entraîné avec succès.")
        
        return history
    
    def train_gan_model(self, name="gan_main", epochs=10000, batch_size=32, sample_interval=1000):
        """
        Entraîne un modèle GAN pour la génération de numéros.
        """
        print(f"Entraînement du modèle GAN {name}...")
        
        # Vérification de l'existence du modèle
        if f"{name}_generator" not in self.models or f"{name}_discriminator" not in self.models:
            print(f"❌ Modèle GAN {name} non trouvé. Construction du modèle...")
            self.build_gan_model(name)
        
        # Récupération des modèles
        generator = self.models[f"{name}_generator"]
        discriminator = self.models[f"{name}_discriminator"]
        gan = self.models[name]
        
        # Sélection des données d'entraînement en fonction du type de modèle
        if "main" in name:
            y_train = self.y_train_main
        else:
            y_train = self.y_train_stars
        
        # Dimension de l'espace latent
        latent_dim = 128
        
        # Historique d'entraînement
        d_losses = []
        g_losses = []
        
        # Entraînement du GAN
        for epoch in range(epochs):
            # Sélection d'un batch aléatoire
            idx = np.random.randint(0, y_train.shape[0], batch_size)
            real_numbers = y_train[idx]
            
            # Génération de bruit aléatoire
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            
            # Génération de numéros synthétiques
            gen_numbers = generator.predict(noise)
            
            # Étiquettes pour les données réelles et synthétiques
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            # Entraînement du discriminateur
            d_loss_real = discriminator.train_on_batch(real_numbers, real_labels)
            d_loss_fake = discriminator.train_on_batch(gen_numbers, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Entraînement du générateur
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, real_labels)
            
            # Sauvegarde des pertes
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            
            # Affichage de la progression
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.4f}] [G loss: {g_loss:.4f}]")
                
                # Génération d'exemples
                self.generate_gan_examples(generator, name, epoch)
        
        # Sauvegarde de l'historique d'entraînement
        self.results[f"{name}_d_losses"] = d_losses
        self.results[f"{name}_g_losses"] = g_losses
        
        # Sauvegarde des modèles finaux
        generator.save(f"models/{name}_generator_final.h5")
        discriminator.save(f"models/{name}_discriminator_final.h5")
        
        # Visualisation de l'historique d'entraînement
        self.plot_gan_training_history(d_losses, g_losses, name)
        
        print(f"✅ Modèle GAN {name} entraîné avec succès.")
        
        return d_losses, g_losses
    
    def train_lstm_transformer_model(self, name="lstm_transformer_main"):
        """
        Entraîne un modèle hybride LSTM-Transformer pour la prédiction des numéros.
        """
        print(f"Entraînement du modèle hybride LSTM-Transformer {name}...")
        
        # Vérification de l'existence du modèle
        if name not in self.models:
            print(f"❌ Modèle {name} non trouvé. Construction du modèle...")
            self.build_lstm_transformer_model(name)
        
        # Sélection des données d'entraînement en fonction du type de modèle
        if "main" in name:
            y_train = self.y_train_main
            y_val = self.y_val_main
        else:
            y_train = self.y_train_stars
            y_val = self.y_val_stars
        
        # Callbacks pour l'entraînement
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f"models/{name}_best.h5",
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entraînement du modèle
        history = self.models[name].fit(
            self.X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarde de l'historique d'entraînement
        self.results[f"{name}_history"] = history.history
        
        # Sauvegarde du modèle final
        self.models[name].save(f"models/{name}_final.h5")
        
        # Visualisation de l'historique d'entraînement
        self.plot_training_history(history, name)
        
        print(f"✅ Modèle hybride LSTM-Transformer {name} entraîné avec succès.")
        
        return history
    
    def plot_training_history(self, history, name):
        """
        Visualise l'historique d'entraînement d'un modèle.
        """
        # Création du répertoire pour les visualisations
        os.makedirs(f"results/{name}", exist_ok=True)
        
        # Tracé de la perte
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Perte du modèle')
        plt.ylabel('Perte')
        plt.xlabel('Epoch')
        plt.legend(['Entraînement', 'Validation'], loc='upper right')
        
        # Tracé de l'erreur absolue moyenne
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Erreur absolue moyenne')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Entraînement', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"results/{name}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gan_training_history(self, d_losses, g_losses, name):
        """
        Visualise l'historique d'entraînement d'un modèle GAN.
        """
        # Création du répertoire pour les visualisations
        os.makedirs(f"results/{name}", exist_ok=True)
        
        # Tracé des pertes
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminateur')
        plt.plot(g_losses, label='Générateur')
        plt.title('Pertes du modèle GAN')
        plt.ylabel('Perte')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"results/{name}/gan_training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_gan_examples(self, generator, name, epoch):
        """
        Génère des exemples de numéros à partir du générateur GAN.
        """
        # Création du répertoire pour les exemples
        os.makedirs(f"results/{name}/examples", exist_ok=True)
        
        # Nombre d'exemples à générer
        n_examples = 10
        
        # Génération de bruit aléatoire
        noise = np.random.normal(0, 1, (n_examples, 128))
        
        # Génération de numéros
        gen_numbers = generator.predict(noise)
        
        # Conversion des numéros normalisés en numéros réels
        if "main" in name:
            # Reshape pour la dénormalisation
            gen_numbers_reshaped = gen_numbers.reshape(-1, 5)
            gen_numbers_real = self.main_scaler.inverse_transform(gen_numbers_reshaped)
            gen_numbers_real = gen_numbers_real.reshape(-1, 5)
            
            # Arrondi et tri des numéros
            gen_numbers_real = np.round(gen_numbers_real).astype(int)
            gen_numbers_real = np.sort(gen_numbers_real, axis=1)
            
            # Limitation des numéros à la plage 1-50
            gen_numbers_real = np.clip(gen_numbers_real, 1, 50)
            
            # Sauvegarde des exemples
            with open(f"results/{name}/examples/epoch_{epoch}.txt", 'w') as f:
                f.write(f"Exemples de numéros principaux générés à l'epoch {epoch}:\n\n")
                for i, nums in enumerate(gen_numbers_real):
                    f.write(f"Exemple {i+1}: {nums[0]}, {nums[1]}, {nums[2]}, {nums[3]}, {nums[4]}\n")
        else:
            # Reshape pour la dénormalisation
            gen_numbers_reshaped = gen_numbers.reshape(-1, 2)
            gen_numbers_real = self.stars_scaler.inverse_transform(gen_numbers_reshaped)
            gen_numbers_real = gen_numbers_real.reshape(-1, 2)
            
            # Arrondi et tri des numéros
            gen_numbers_real = np.round(gen_numbers_real).astype(int)
            gen_numbers_real = np.sort(gen_numbers_real, axis=1)
            
            # Limitation des numéros à la plage 1-12
            gen_numbers_real = np.clip(gen_numbers_real, 1, 12)
            
            # Sauvegarde des exemples
            with open(f"results/{name}/examples/epoch_{epoch}.txt", 'w') as f:
                f.write(f"Exemples d'étoiles générées à l'epoch {epoch}:\n\n")
                for i, stars in enumerate(gen_numbers_real):
                    f.write(f"Exemple {i+1}: {stars[0]}, {stars[1]}\n")
    
    def evaluate_models(self):
        """
        Évalue les performances des modèles sur l'ensemble de test.
        """
        print("Évaluation des modèles sur l'ensemble de test...")
        
        # Dictionnaire pour stocker les résultats d'évaluation
        evaluation_results = {}
        
        # Évaluation des modèles pour les numéros principaux
        for name in [model_name for model_name in self.models.keys() if "main" in model_name and not any(x in model_name for x in ["generator", "discriminator"])]:
            print(f"Évaluation du modèle {name}...")
            
            # Prédiction sur l'ensemble de test
            y_pred = self.models[name].predict(self.X_test)
            
            # Conversion des prédictions normalisées en numéros réels
            y_pred_reshaped = y_pred.reshape(-1, 5)
            y_pred_real = self.main_scaler.inverse_transform(y_pred_reshaped)
            y_pred_real = y_pred_real.reshape(-1, 5)
            
            # Arrondi et tri des numéros
            y_pred_real = np.round(y_pred_real).astype(int)
            y_pred_real = np.sort(y_pred_real, axis=1)
            
            # Limitation des numéros à la plage 1-50
            y_pred_real = np.clip(y_pred_real, 1, 50)
            
            # Conversion des cibles normalisées en numéros réels
            y_true_reshaped = self.y_test_main.reshape(-1, 5)
            y_true_real = self.main_scaler.inverse_transform(y_true_reshaped)
            y_true_real = y_true_real.reshape(-1, 5)
            
            # Arrondi et tri des numéros
            y_true_real = np.round(y_true_real).astype(int)
            y_true_real = np.sort(y_true_real, axis=1)
            
            # Calcul des métriques d'évaluation
            mse = mean_squared_error(y_true_real, y_pred_real)
            mae = mean_absolute_error(y_true_real, y_pred_real)
            
            # Calcul du nombre moyen de numéros correctement prédits
            correct_numbers = []
            for i in range(len(y_true_real)):
                correct = len(set(y_true_real[i]) & set(y_pred_real[i]))
                correct_numbers.append(correct)
            
            avg_correct = np.mean(correct_numbers)
            
            # Sauvegarde des résultats
            evaluation_results[name] = {
                "mse": mse,
                "mae": mae,
                "avg_correct_numbers": avg_correct
            }
            
            print(f"   - MSE: {mse:.4f}")
            print(f"   - MAE: {mae:.4f}")
            print(f"   - Nombre moyen de numéros correctement prédits: {avg_correct:.2f}/5")
        
        # Évaluation des modèles pour les étoiles
        for name in [model_name for model_name in self.models.keys() if "stars" in model_name and not any(x in model_name for x in ["generator", "discriminator"])]:
            print(f"Évaluation du modèle {name}...")
            
            # Prédiction sur l'ensemble de test
            y_pred = self.models[name].predict(self.X_test)
            
            # Conversion des prédictions normalisées en numéros réels
            y_pred_reshaped = y_pred.reshape(-1, 2)
            y_pred_real = self.stars_scaler.inverse_transform(y_pred_reshaped)
            y_pred_real = y_pred_real.reshape(-1, 2)
            
            # Arrondi et tri des numéros
            y_pred_real = np.round(y_pred_real).astype(int)
            y_pred_real = np.sort(y_pred_real, axis=1)
            
            # Limitation des numéros à la plage 1-12
            y_pred_real = np.clip(y_pred_real, 1, 12)
            
            # Conversion des cibles normalisées en numéros réels
            y_true_reshaped = self.y_test_stars.reshape(-1, 2)
            y_true_real = self.stars_scaler.inverse_transform(y_true_reshaped)
            y_true_real = y_true_real.reshape(-1, 2)
            
            # Arrondi et tri des numéros
            y_true_real = np.round(y_true_real).astype(int)
            y_true_real = np.sort(y_true_real, axis=1)
            
            # Calcul des métriques d'évaluation
            mse = mean_squared_error(y_true_real, y_pred_real)
            mae = mean_absolute_error(y_true_real, y_pred_real)
            
            # Calcul du nombre moyen d'étoiles correctement prédites
            correct_stars = []
            for i in range(len(y_true_real)):
                correct = len(set(y_true_real[i]) & set(y_pred_real[i]))
                correct_stars.append(correct)
            
            avg_correct = np.mean(correct_stars)
            
            # Sauvegarde des résultats
            evaluation_results[name] = {
                "mse": mse,
                "mae": mae,
                "avg_correct_stars": avg_correct
            }
            
            print(f"   - MSE: {mse:.4f}")
            print(f"   - MAE: {mae:.4f}")
            print(f"   - Nombre moyen d'étoiles correctement prédites: {avg_correct:.2f}/2")
        
        # Sauvegarde des résultats d'évaluation
        with open("results/evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        print(f"✅ Résultats d'évaluation sauvegardés dans results/evaluation_results.json")
        
        return evaluation_results
    
    def generate_predictions(self):
        """
        Génère des prédictions pour le prochain tirage.
        """
        print("Génération de prédictions pour le prochain tirage...")
        
        # Vérification de l'existence des modèles
        if not self.models:
            print("❌ Aucun modèle disponible pour générer des prédictions.")
            return None
        
        # Préparation des données pour la prédiction
        # On utilise les dernières séquences de l'ensemble de test
        last_sequence = self.X_test[-1:]
        
        # Dictionnaire pour stocker les prédictions
        predictions = {}
        
        # Génération des prédictions pour les numéros principaux
        for name in [model_name for model_name in self.models.keys() if "main" in model_name and not any(x in model_name for x in ["generator", "discriminator"])]:
            print(f"Génération de prédictions avec le modèle {name}...")
            
            # Prédiction
            y_pred = self.models[name].predict(last_sequence)
            
            # Conversion des prédictions normalisées en numéros réels
            y_pred_reshaped = y_pred.reshape(-1, 5)
            y_pred_real = self.main_scaler.inverse_transform(y_pred_reshaped)
            y_pred_real = y_pred_real.reshape(-1, 5)
            
            # Arrondi et tri des numéros
            y_pred_real = np.round(y_pred_real).astype(int)
            y_pred_real = np.sort(y_pred_real, axis=1)
            
            # Limitation des numéros à la plage 1-50
            y_pred_real = np.clip(y_pred_real, 1, 50)
            
            # Sauvegarde des prédictions
            predictions[name] = y_pred_real[0].tolist()
            
            print(f"   - Numéros prédits: {y_pred_real[0]}")
        
        # Génération des prédictions pour les étoiles
        for name in [model_name for model_name in self.models.keys() if "stars" in model_name and not any(x in model_name for x in ["generator", "discriminator"])]:
            print(f"Génération de prédictions avec le modèle {name}...")
            
            # Prédiction
            y_pred = self.models[name].predict(last_sequence)
            
            # Conversion des prédictions normalisées en numéros réels
            y_pred_reshaped = y_pred.reshape(-1, 2)
            y_pred_real = self.stars_scaler.inverse_transform(y_pred_reshaped)
            y_pred_real = y_pred_real.reshape(-1, 2)
            
            # Arrondi et tri des numéros
            y_pred_real = np.round(y_pred_real).astype(int)
            y_pred_real = np.sort(y_pred_real, axis=1)
            
            # Limitation des numéros à la plage 1-12
            y_pred_real = np.clip(y_pred_real, 1, 12)
            
            # Sauvegarde des prédictions
            predictions[name] = y_pred_real[0].tolist()
            
            print(f"   - Étoiles prédites: {y_pred_real[0]}")
        
        # Génération de prédictions avec les modèles GAN
        for name in [model_name for model_name in self.models.keys() if "generator" in model_name]:
            print(f"Génération de prédictions avec le modèle {name}...")
            
            # Génération de bruit aléatoire
            noise = np.random.normal(0, 1, (10, 128))
            
            # Génération de numéros
            gen_numbers = self.models[name].predict(noise)
            
            # Conversion des numéros normalisés en numéros réels
            if "main" in name:
                # Reshape pour la dénormalisation
                gen_numbers_reshaped = gen_numbers.reshape(-1, 5)
                gen_numbers_real = self.main_scaler.inverse_transform(gen_numbers_reshaped)
                gen_numbers_real = gen_numbers_real.reshape(-1, 5)
                
                # Arrondi et tri des numéros
                gen_numbers_real = np.round(gen_numbers_real).astype(int)
                gen_numbers_real = np.sort(gen_numbers_real, axis=1)
                
                # Limitation des numéros à la plage 1-50
                gen_numbers_real = np.clip(gen_numbers_real, 1, 50)
                
                # Sauvegarde des prédictions (moyenne des 10 générations)
                predictions[name] = np.mean(gen_numbers_real, axis=0).astype(int).tolist()
                
                print(f"   - Numéros générés (moyenne): {predictions[name]}")
            else:
                # Reshape pour la dénormalisation
                gen_numbers_reshaped = gen_numbers.reshape(-1, 2)
                gen_numbers_real = self.stars_scaler.inverse_transform(gen_numbers_reshaped)
                gen_numbers_real = gen_numbers_real.reshape(-1, 2)
                
                # Arrondi et tri des numéros
                gen_numbers_real = np.round(gen_numbers_real).astype(int)
                gen_numbers_real = np.sort(gen_numbers_real, axis=1)
                
                # Limitation des numéros à la plage 1-12
                gen_numbers_real = np.clip(gen_numbers_real, 1, 12)
                
                # Sauvegarde des prédictions (moyenne des 10 générations)
                predictions[name] = np.mean(gen_numbers_real, axis=0).astype(int).tolist()
                
                print(f"   - Étoiles générées (moyenne): {predictions[name]}")
        
        # Sauvegarde des prédictions
        self.predictions = predictions
        
        # Sauvegarde des prédictions dans un fichier
        with open("results/predictions.json", 'w') as f:
            json.dump(predictions, f, indent=4)
        
        # Création d'un fichier de prédiction plus lisible
        with open("results/prediction.txt", 'w') as f:
            f.write("Prédictions pour le prochain tirage de l'Euromillions\n")
            f.write("=================================================\n\n")
            
            f.write("Date de génération: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            f.write("Prédictions des numéros principaux:\n")
            for name in [model_name for model_name in predictions.keys() if "main" in model_name]:
                f.write(f"   - Modèle {name}: {', '.join(map(str, predictions[name]))}\n")
            
            f.write("\nPrédictions des étoiles:\n")
            for name in [model_name for model_name in predictions.keys() if "stars" in model_name]:
                f.write(f"   - Modèle {name}: {', '.join(map(str, predictions[name]))}\n")
            
            f.write("\nPrédiction finale (consensus):\n")
            
            # Calcul du consensus pour les numéros principaux
            main_predictions = [predictions[name] for name in predictions.keys() if "main" in name]
            main_consensus = self.calculate_consensus(main_predictions, 5, 50)
            
            # Calcul du consensus pour les étoiles
            stars_predictions = [predictions[name] for name in predictions.keys() if "stars" in name]
            stars_consensus = self.calculate_consensus(stars_predictions, 2, 12)
            
            f.write(f"   - Numéros principaux: {', '.join(map(str, main_consensus))}\n")
            f.write(f"   - Étoiles: {', '.join(map(str, stars_consensus))}\n")
        
        print(f"✅ Prédictions sauvegardées dans results/predictions.json et results/prediction.txt")
        
        return predictions
    
    def calculate_consensus(self, predictions_list, num_to_select, max_value):
        """
        Calcule un consensus à partir de plusieurs prédictions.
        """
        # Comptage des occurrences de chaque numéro
        counts = {}
        for prediction in predictions_list:
            for num in prediction:
                if num in counts:
                    counts[num] += 1
                else:
                    counts[num] = 1
        
        # Tri des numéros par nombre d'occurrences
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection des numéros les plus fréquents
        consensus = [num for num, _ in sorted_counts[:num_to_select]]
        
        # Si pas assez de numéros, on complète avec des numéros aléatoires
        if len(consensus) < num_to_select:
            available_nums = [n for n in range(1, max_value + 1) if n not in consensus]
            additional_nums = random.sample(available_nums, num_to_select - len(consensus))
            consensus.extend(additional_nums)
        
        # Tri des numéros
        consensus.sort()
        
        return consensus
    
    def run_full_pipeline(self, quick_mode=False):
        """
        Exécute le pipeline complet de développement et d'entraînement des modèles.
        """
        print("Démarrage du pipeline de développement des modèles avancés...")
        
        # 1. Chargement des données
        self.load_data()
        
        # 2. Préparation des données pour les modèles
        self.prepare_data_for_transformer()
        
        # 3. Construction et entraînement des modèles
        if quick_mode:
            # Mode rapide : moins d'époques et un seul modèle
            self.epochs = 20
            self.patience = 5
            
            # Construction et entraînement du modèle Transformer pour les numéros principaux
            self.build_transformer_model("transformer_main")
            self.train_transformer_model("transformer_main")
            
            # Construction et entraînement du modèle Transformer pour les étoiles
            self.build_transformer_model("transformer_stars")
            self.train_transformer_model("transformer_stars")
        else:
            # Mode complet : tous les modèles
            
            # Construction et entraînement des modèles Transformer
            self.build_transformer_model("transformer_main")
            self.train_transformer_model("transformer_main")
            
            self.build_transformer_model("transformer_stars")
            self.train_transformer_model("transformer_stars")
            
            # Construction et entraînement des modèles hybrides LSTM-Transformer
            self.build_lstm_transformer_model("lstm_transformer_main")
            self.train_lstm_transformer_model("lstm_transformer_main")
            
            self.build_lstm_transformer_model("lstm_transformer_stars")
            self.train_lstm_transformer_model("lstm_transformer_stars")
            
            # Construction et entraînement des modèles GAN
            self.build_gan_model("gan_main")
            self.train_gan_model("gan_main", epochs=1000, sample_interval=100)
            
            self.build_gan_model("gan_stars")
            self.train_gan_model("gan_stars", epochs=1000, sample_interval=100)
        
        # 4. Évaluation des modèles
        self.evaluate_models()
        
        # 5. Génération de prédictions
        self.generate_predictions()
        
        print("✅ Pipeline de développement des modèles avancés terminé avec succès!")
        
        return self.predictions

# Exécution du pipeline
if __name__ == "__main__":
    # Vérification de l'existence du fichier de données enrichies
    if os.path.exists("data_enriched/euromillions_enhanced_dataset.csv"):
        # Mode complet
        advanced_models = EuromillionsAdvancedModels()
        advanced_models.run_full_pipeline()
    else:
        # Mode rapide avec données synthétiques
        print("⚠️ Fichier de données enrichies non trouvé. Exécution en mode rapide avec données synthétiques.")
        advanced_models = EuromillionsAdvancedModels()
        advanced_models.run_full_pipeline(quick_mode=True)

