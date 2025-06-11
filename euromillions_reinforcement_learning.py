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
import json
import random
import time
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import plot_optimization_history, plot_param_importances
import gym
from gym import spaces
import stable_baselines3
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import joblib
from datetime import datetime

# Création des répertoires pour les modèles et les résultats
os.makedirs("models/reinforcement", exist_ok=True)
os.makedirs("results/reinforcement", exist_ok=True)
os.makedirs("models/auto_ml", exist_ok=True)
os.makedirs("results/auto_ml", exist_ok=True)

class EuromillionsEnv(gym.Env):
    """
    Environnement Gym pour l'apprentissage par renforcement
    appliqué à la prédiction des numéros de l'Euromillions.
    """
    
    def __init__(self, data_path="data_enriched/euromillions_enhanced_dataset.csv", sequence_length=10):
        super(EuromillionsEnv, self).__init__()
        
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.current_step = 0
        self.max_steps = 100  # Nombre maximum d'étapes par épisode
        
        # Chargement des données
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
        else:
            print(f"❌ Fichier de données {self.data_path} non trouvé.")
            print("⚠️ Création d'un jeu de données synthétique...")
            self.create_synthetic_dataset()
        
        # Préparation des données
        self.prepare_data()
        
        # Définition de l'espace d'action
        # Pour les numéros principaux : 5 numéros entre 1 et 50
        # Pour les étoiles : 2 numéros entre 1 et 12
        self.action_space = spaces.MultiDiscrete([50, 50, 50, 50, 50, 12, 12])
        
        # Définition de l'espace d'observation
        # Caractéristiques des séquences précédentes
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sequence_length, self.n_features),
            dtype=np.float32
        )
        
        # État actuel
        self.state = None
        
        # Tirage actuel (cible)
        self.current_draw = None
    
    def create_synthetic_dataset(self):
        """
        Crée un jeu de données synthétique pour le développement.
        """
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
    
    def prepare_data(self):
        """
        Prépare les données pour l'environnement d'apprentissage par renforcement.
        """
        # Sélection des caractéristiques pertinentes
        feature_columns = [col for col in self.df.columns if col not in [
            'date', 'draw_id', 'has_winner', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2'
        ]]
        
        self.n_features = len(feature_columns)
        
        # Normalisation des caractéristiques
        self.feature_scaler = StandardScaler()
        self.features = self.feature_scaler.fit_transform(self.df[feature_columns])
        
        # Extraction des numéros cibles
        self.targets = np.column_stack([
            self.df['N1'].values,
            self.df['N2'].values,
            self.df['N3'].values,
            self.df['N4'].values,
            self.df['N5'].values,
            self.df['E1'].values,
            self.df['E2'].values
        ])
        
        # Normalisation des cibles
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.targets_scaled = self.target_scaler.fit_transform(self.targets)
        
        # Création des séquences
        self.sequences = []
        self.sequence_targets = []
        
        for i in range(self.sequence_length, len(self.features)):
            self.sequences.append(self.features[i-self.sequence_length:i])
            self.sequence_targets.append(self.targets[i])
        
        self.sequences = np.array(self.sequences)
        self.sequence_targets = np.array(self.sequence_targets)
        
        # Division en ensembles d'entraînement et de test
        self.train_indices, self.test_indices = train_test_split(
            np.arange(len(self.sequences)),
            test_size=0.2,
            shuffle=False
        )
        
        print(f"✅ Données préparées pour l'environnement d'apprentissage par renforcement :")
        print(f"   - Nombre de séquences : {len(self.sequences)}")
        print(f"   - Nombre de caractéristiques : {self.n_features}")
    
    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        # Sélection aléatoire d'une séquence d'entraînement
        idx = np.random.choice(self.train_indices)
        
        # Initialisation de l'état
        self.state = self.sequences[idx]
        
        # Initialisation du tirage cible
        self.current_draw = self.sequence_targets[idx]
        
        # Réinitialisation du compteur d'étapes
        self.current_step = 0
        
        return self.state
    
    def step(self, action):
        """
        Exécute une action dans l'environnement.
        """
        self.current_step += 1
        
        # Conversion de l'action en numéros (1-50 pour les numéros principaux, 1-12 pour les étoiles)
        action = np.array(action)
        action[:5] = np.clip(action[:5], 0, 49) + 1  # Numéros principaux (1-50)
        action[5:] = np.clip(action[5:], 0, 11) + 1  # Étoiles (1-12)
        
        # Tri des numéros principaux et des étoiles
        main_numbers = np.sort(action[:5])
        stars = np.sort(action[5:])
        
        # Vérification des doublons
        if len(np.unique(main_numbers)) < 5 or len(np.unique(stars)) < 2:
            # Pénalité pour les doublons
            reward = -10
            done = True
            info = {"error": "Doublons détectés dans les numéros"}
            return self.state, reward, done, info
        
        # Calcul de la récompense
        reward = self.calculate_reward(main_numbers, stars)
        
        # Vérification de la fin de l'épisode
        done = self.current_step >= self.max_steps
        
        # Informations supplémentaires
        info = {
            "predicted_numbers": main_numbers,
            "predicted_stars": stars,
            "target_numbers": self.current_draw[:5],
            "target_stars": self.current_draw[5:],
            "correct_numbers": len(set(main_numbers) & set(self.current_draw[:5])),
            "correct_stars": len(set(stars) & set(self.current_draw[5:]))
        }
        
        return self.state, reward, done, info
    
    def calculate_reward(self, main_numbers, stars):
        """
        Calcule la récompense en fonction des numéros prédits et des numéros cibles.
        """
        # Nombre de numéros principaux corrects
        correct_main = len(set(main_numbers) & set(self.current_draw[:5]))
        
        # Nombre d'étoiles correctes
        correct_stars = len(set(stars) & set(self.current_draw[5:]))
        
        # Récompense de base pour les numéros corrects
        reward = correct_main * 2 + correct_stars * 3
        
        # Bonus pour les combinaisons gagnantes
        if correct_main == 5 and correct_stars == 2:
            # Jackpot
            reward += 100
        elif correct_main == 5 and correct_stars == 1:
            reward += 50
        elif correct_main == 5 and correct_stars == 0:
            reward += 30
        elif correct_main == 4 and correct_stars == 2:
            reward += 20
        elif correct_main == 4 and correct_stars == 1:
            reward += 10
        elif correct_main == 3 and correct_stars == 2:
            reward += 5
        
        return reward
    
    def render(self, mode='human'):
        """
        Affiche l'état actuel de l'environnement.
        """
        if mode == 'human':
            print(f"Étape : {self.current_step}")
            print(f"Tirage cible : {self.current_draw}")
        
        return None
    
    def close(self):
        """
        Ferme l'environnement.
        """
        pass

class EuromillionsReinforcementLearning:
    """
    Classe pour l'apprentissage par renforcement appliqué à la prédiction
    des numéros de l'Euromillions.
    """
    
    def __init__(self, data_path="data_enriched/euromillions_enhanced_dataset.csv"):
        """
        Initialise la classe avec le chemin vers les données enrichies.
        """
        self.data_path = data_path
        self.env = None
        self.models = {}
        self.results = {}
    
    def create_environment(self):
        """
        Crée l'environnement d'apprentissage par renforcement.
        """
        print("Création de l'environnement d'apprentissage par renforcement...")
        
        # Création de l'environnement
        self.env = EuromillionsEnv(data_path=self.data_path)
        
        # Vectorisation de l'environnement
        self.vec_env = make_vec_env(lambda: self.env, n_envs=1)
        
        print("✅ Environnement d'apprentissage par renforcement créé.")
        
        return self.env
    
    def train_ppo_model(self, total_timesteps=100000):
        """
        Entraîne un modèle PPO (Proximal Policy Optimization).
        """
        print("Entraînement du modèle PPO...")
        
        # Vérification de l'existence de l'environnement
        if self.env is None:
            self.create_environment()
        
        # Création du modèle PPO
        model = PPO(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./results/reinforcement/tensorboard/"
        )
        
        # Création d'un callback pour l'évaluation
        eval_callback = EvalCallback(
            self.vec_env,
            best_model_save_path="./models/reinforcement/",
            log_path="./results/reinforcement/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Entraînement du modèle
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name="ppo_euromillions"
        )
        
        # Sauvegarde du modèle
        model.save("models/reinforcement/ppo_euromillions")
        
        # Sauvegarde du modèle dans le dictionnaire
        self.models["ppo"] = model
        
        print("✅ Modèle PPO entraîné avec succès.")
        
        return model
    
    def train_a2c_model(self, total_timesteps=100000):
        """
        Entraîne un modèle A2C (Advantage Actor Critic).
        """
        print("Entraînement du modèle A2C...")
        
        # Vérification de l'existence de l'environnement
        if self.env is None:
            self.create_environment()
        
        # Création du modèle A2C
        model = A2C(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_rms_prop=True,
            normalize_advantage=False,
            tensorboard_log="./results/reinforcement/tensorboard/"
        )
        
        # Création d'un callback pour l'évaluation
        eval_callback = EvalCallback(
            self.vec_env,
            best_model_save_path="./models/reinforcement/",
            log_path="./results/reinforcement/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Entraînement du modèle
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name="a2c_euromillions"
        )
        
        # Sauvegarde du modèle
        model.save("models/reinforcement/a2c_euromillions")
        
        # Sauvegarde du modèle dans le dictionnaire
        self.models["a2c"] = model
        
        print("✅ Modèle A2C entraîné avec succès.")
        
        return model
    
    def evaluate_model(self, model_name="ppo", n_eval_episodes=100):
        """
        Évalue un modèle d'apprentissage par renforcement.
        """
        print(f"Évaluation du modèle {model_name}...")
        
        # Vérification de l'existence du modèle
        if model_name not in self.models:
            if model_name == "ppo":
                # Chargement du modèle PPO
                model = PPO.load("models/reinforcement/ppo_euromillions")
                self.models["ppo"] = model
            elif model_name == "a2c":
                # Chargement du modèle A2C
                model = A2C.load("models/reinforcement/a2c_euromillions")
                self.models["a2c"] = model
            else:
                print(f"❌ Modèle {model_name} non trouvé.")
                return None
        
        # Évaluation du modèle
        mean_reward, std_reward = evaluate_policy(
            self.models[model_name],
            self.vec_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        print(f"✅ Évaluation du modèle {model_name} terminée.")
        print(f"   - Récompense moyenne : {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Sauvegarde des résultats
        self.results[f"{model_name}_evaluation"] = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_eval_episodes": n_eval_episodes
        }
        
        # Sauvegarde des résultats dans un fichier
        with open(f"results/reinforcement/{model_name}_evaluation.json", 'w') as f:
            json.dump(self.results[f"{model_name}_evaluation"], f, indent=4)
        
        return mean_reward, std_reward
    
    def generate_predictions(self, model_name="ppo", n_predictions=10):
        """
        Génère des prédictions avec un modèle d'apprentissage par renforcement.
        """
        print(f"Génération de prédictions avec le modèle {model_name}...")
        
        # Vérification de l'existence du modèle
        if model_name not in self.models:
            if model_name == "ppo":
                # Chargement du modèle PPO
                model = PPO.load("models/reinforcement/ppo_euromillions")
                self.models["ppo"] = model
            elif model_name == "a2c":
                # Chargement du modèle A2C
                model = A2C.load("models/reinforcement/a2c_euromillions")
                self.models["a2c"] = model
            else:
                print(f"❌ Modèle {model_name} non trouvé.")
                return None
        
        # Réinitialisation de l'environnement
        obs = self.vec_env.reset()
        
        # Génération des prédictions
        predictions = []
        
        for _ in range(n_predictions):
            # Prédiction de l'action
            action, _ = self.models[model_name].predict(obs, deterministic=True)
            
            # Conversion de l'action en numéros
            main_numbers = np.sort(np.clip(action[0, :5], 0, 49) + 1)
            stars = np.sort(np.clip(action[0, 5:], 0, 11) + 1)
            
            # Vérification des doublons
            if len(np.unique(main_numbers)) < 5:
                # Remplacement des doublons
                unique_numbers = set(main_numbers)
                missing_count = 5 - len(unique_numbers)
                available_numbers = [n for n in range(1, 51) if n not in unique_numbers]
                replacement_numbers = np.random.choice(available_numbers, missing_count, replace=False)
                main_numbers = np.sort(np.concatenate([list(unique_numbers), replacement_numbers]))
            
            if len(np.unique(stars)) < 2:
                # Remplacement des doublons
                unique_stars = set(stars)
                missing_count = 2 - len(unique_stars)
                available_stars = [n for n in range(1, 13) if n not in unique_stars]
                replacement_stars = np.random.choice(available_stars, missing_count, replace=False)
                stars = np.sort(np.concatenate([list(unique_stars), replacement_stars]))
            
            # Ajout de la prédiction
            predictions.append({
                "main_numbers": main_numbers.tolist(),
                "stars": stars.tolist()
            })
            
            # Exécution de l'action dans l'environnement
            obs, _, done, _ = self.vec_env.step(action)
            
            if done:
                obs = self.vec_env.reset()
        
        # Sauvegarde des prédictions
        self.results[f"{model_name}_predictions"] = predictions
        
        # Sauvegarde des prédictions dans un fichier
        with open(f"results/reinforcement/{model_name}_predictions.json", 'w') as f:
            json.dump(predictions, f, indent=4)
        
        # Création d'un fichier de prédiction plus lisible
        with open(f"results/reinforcement/{model_name}_prediction.txt", 'w') as f:
            f.write(f"Prédictions générées par le modèle {model_name}\n")
            f.write("=================================================\n\n")
            
            f.write("Date de génération: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            for i, pred in enumerate(predictions):
                f.write(f"Prédiction {i+1}:\n")
                f.write(f"   - Numéros principaux: {', '.join(map(str, pred['main_numbers']))}\n")
                f.write(f"   - Étoiles: {', '.join(map(str, pred['stars']))}\n\n")
            
            # Calcul du consensus
            main_consensus = self.calculate_consensus([p["main_numbers"] for p in predictions], 5, 50)
            stars_consensus = self.calculate_consensus([p["stars"] for p in predictions], 2, 12)
            
            f.write("Prédiction finale (consensus):\n")
            f.write(f"   - Numéros principaux: {', '.join(map(str, main_consensus))}\n")
            f.write(f"   - Étoiles: {', '.join(map(str, stars_consensus))}\n")
        
        print(f"✅ Prédictions générées avec le modèle {model_name}.")
        
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
    
    def run_full_pipeline(self, quick_mode=True):
        """
        Exécute le pipeline complet d'apprentissage par renforcement.
        """
        print("Démarrage du pipeline d'apprentissage par renforcement...")
        
        # 1. Création de l'environnement
        self.create_environment()
        
        # 2. Entraînement des modèles
        if quick_mode:
            # Mode rapide : moins d'étapes
            self.train_ppo_model(total_timesteps=10000)
        else:
            # Mode complet : tous les modèles
            self.train_ppo_model(total_timesteps=100000)
            self.train_a2c_model(total_timesteps=100000)
        
        # 3. Évaluation des modèles
        if quick_mode:
            self.evaluate_model(model_name="ppo", n_eval_episodes=10)
        else:
            self.evaluate_model(model_name="ppo", n_eval_episodes=100)
            self.evaluate_model(model_name="a2c", n_eval_episodes=100)
        
        # 4. Génération de prédictions
        if quick_mode:
            self.generate_predictions(model_name="ppo", n_predictions=5)
        else:
            self.generate_predictions(model_name="ppo", n_predictions=10)
            self.generate_predictions(model_name="a2c", n_predictions=10)
        
        print("✅ Pipeline d'apprentissage par renforcement terminé avec succès!")
        
        return self.results

class EuromillionsAutoML:
    """
    Classe pour l'optimisation automatique des hyperparamètres et
    l'auto-adaptation des modèles de prédiction des numéros de l'Euromillions.
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
        self.best_models = {}
        self.best_params = {}
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
        Crée un jeu de données synthétique pour le développement.
        """
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
    
    def prepare_data(self):
        """
        Prépare les données pour l'optimisation des hyperparamètres.
        """
        print("Préparation des données pour l'optimisation des hyperparamètres...")
        
        # Sélection des caractéristiques pertinentes
        feature_columns = [col for col in self.df.columns if col not in [
            'date', 'draw_id', 'has_winner', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2'
        ]]
        
        # Normalisation des caractéristiques
        self.feature_scaler = StandardScaler()
        features = self.feature_scaler.fit_transform(self.df[feature_columns])
        
        # Création des séquences
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
        
        print(f"✅ Données préparées pour l'optimisation des hyperparamètres :")
        print(f"   - Ensemble d'entraînement : {len(self.X_train)} séquences")
        print(f"   - Ensemble de validation : {len(self.X_val)} séquences")
        print(f"   - Ensemble de test : {len(self.X_test)} séquences")
        
        return True
    
    def create_transformer_model(self, trial, target_type="main"):
        """
        Crée un modèle Transformer avec des hyperparamètres optimisés par Optuna.
        """
        # Détermination de la taille de sortie (5 pour les numéros principaux, 2 pour les étoiles)
        output_size = 5 if target_type == "main" else 2
        
        # Paramètres du modèle à optimiser
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
        num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 1, 3)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        
        # Paramètres du modèle
        input_shape = self.X_train.shape[1:]
        
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
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation="relu")(x)
        
        # Couche de sortie
        outputs = layers.Dense(output_size, activation="sigmoid")(x)
        
        # Création du modèle
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilation du modèle
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def create_lstm_transformer_model(self, trial, target_type="main"):
        """
        Crée un modèle hybride LSTM-Transformer avec des hyperparamètres optimisés par Optuna.
        """
        # Détermination de la taille de sortie (5 pour les numéros principaux, 2 pour les étoiles)
        output_size = 5 if target_type == "main" else 2
        
        # Paramètres du modèle à optimiser
        lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128])
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        
        # Paramètres du modèle
        input_shape = self.X_train.shape[1:]
        
        # Construction du modèle
        inputs = keras.Input(shape=input_shape)
        
        # Couche LSTM bidirectionnelle
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
        
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
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation="relu")(x)
        
        # Couche de sortie
        outputs = layers.Dense(output_size, activation="sigmoid")(x)
        
        # Création du modèle
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilation du modèle
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def objective_transformer(self, trial, target_type="main"):
        """
        Fonction objectif pour l'optimisation des hyperparamètres du modèle Transformer.
        """
        # Création du modèle avec les hyperparamètres suggérés
        model = self.create_transformer_model(trial, target_type)
        
        # Sélection des données d'entraînement en fonction du type de modèle
        if target_type == "main":
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
            TFKerasPruningCallback(trial, "val_loss")
        ]
        
        # Entraînement du modèle
        history = model.fit(
            self.X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Évaluation du modèle sur l'ensemble de validation
        val_loss = history.history["val_loss"][-1]
        
        return val_loss
    
    def objective_lstm_transformer(self, trial, target_type="main"):
        """
        Fonction objectif pour l'optimisation des hyperparamètres du modèle hybride LSTM-Transformer.
        """
        # Création du modèle avec les hyperparamètres suggérés
        model = self.create_lstm_transformer_model(trial, target_type)
        
        # Sélection des données d'entraînement en fonction du type de modèle
        if target_type == "main":
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
            TFKerasPruningCallback(trial, "val_loss")
        ]
        
        # Entraînement du modèle
        history = model.fit(
            self.X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Évaluation du modèle sur l'ensemble de validation
        val_loss = history.history["val_loss"][-1]
        
        return val_loss
    
    def optimize_transformer(self, target_type="main", n_trials=100):
        """
        Optimise les hyperparamètres du modèle Transformer.
        """
        print(f"Optimisation des hyperparamètres du modèle Transformer pour {target_type}...")
        
        # Création de l'étude Optuna
        study_name = f"transformer_{target_type}"
        storage_name = f"sqlite:///models/auto_ml/{study_name}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimisation des hyperparamètres
        study.optimize(
            lambda trial: self.objective_transformer(trial, target_type),
            n_trials=n_trials
        )
        
        # Récupération des meilleurs hyperparamètres
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"✅ Optimisation terminée pour le modèle Transformer {target_type}.")
        print(f"   - Meilleure valeur : {best_value:.6f}")
        print(f"   - Meilleurs hyperparamètres : {best_params}")
        
        # Sauvegarde des meilleurs hyperparamètres
        self.best_params[f"transformer_{target_type}"] = best_params
        
        # Sauvegarde des résultats de l'optimisation
        self.results[f"transformer_{target_type}_optimization"] = {
            "best_value": best_value,
            "best_params": best_params
        }
        
        # Sauvegarde des résultats dans un fichier
        with open(f"results/auto_ml/transformer_{target_type}_optimization.json", 'w') as f:
            json.dump(self.results[f"transformer_{target_type}_optimization"], f, indent=4)
        
        # Visualisation des résultats
        fig1 = plot_optimization_history(study)
        fig1.write_image(f"results/auto_ml/transformer_{target_type}_history.png")
        
        fig2 = plot_param_importances(study)
        fig2.write_image(f"results/auto_ml/transformer_{target_type}_importance.png")
        
        # Entraînement du meilleur modèle
        self.train_best_transformer(target_type)
        
        return best_params, best_value
    
    def optimize_lstm_transformer(self, target_type="main", n_trials=100):
        """
        Optimise les hyperparamètres du modèle hybride LSTM-Transformer.
        """
        print(f"Optimisation des hyperparamètres du modèle LSTM-Transformer pour {target_type}...")
        
        # Création de l'étude Optuna
        study_name = f"lstm_transformer_{target_type}"
        storage_name = f"sqlite:///models/auto_ml/{study_name}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimisation des hyperparamètres
        study.optimize(
            lambda trial: self.objective_lstm_transformer(trial, target_type),
            n_trials=n_trials
        )
        
        # Récupération des meilleurs hyperparamètres
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"✅ Optimisation terminée pour le modèle LSTM-Transformer {target_type}.")
        print(f"   - Meilleure valeur : {best_value:.6f}")
        print(f"   - Meilleurs hyperparamètres : {best_params}")
        
        # Sauvegarde des meilleurs hyperparamètres
        self.best_params[f"lstm_transformer_{target_type}"] = best_params
        
        # Sauvegarde des résultats de l'optimisation
        self.results[f"lstm_transformer_{target_type}_optimization"] = {
            "best_value": best_value,
            "best_params": best_params
        }
        
        # Sauvegarde des résultats dans un fichier
        with open(f"results/auto_ml/lstm_transformer_{target_type}_optimization.json", 'w') as f:
            json.dump(self.results[f"lstm_transformer_{target_type}_optimization"], f, indent=4)
        
        # Visualisation des résultats
        fig1 = plot_optimization_history(study)
        fig1.write_image(f"results/auto_ml/lstm_transformer_{target_type}_history.png")
        
        fig2 = plot_param_importances(study)
        fig2.write_image(f"results/auto_ml/lstm_transformer_{target_type}_importance.png")
        
        # Entraînement du meilleur modèle
        self.train_best_lstm_transformer(target_type)
        
        return best_params, best_value
    
    def train_best_transformer(self, target_type="main"):
        """
        Entraîne le meilleur modèle Transformer avec les hyperparamètres optimisés.
        """
        print(f"Entraînement du meilleur modèle Transformer pour {target_type}...")
        
        # Vérification de l'existence des meilleurs hyperparamètres
        if f"transformer_{target_type}" not in self.best_params:
            print(f"❌ Meilleurs hyperparamètres non trouvés pour le modèle Transformer {target_type}.")
            return None
        
        # Récupération des meilleurs hyperparamètres
        best_params = self.best_params[f"transformer_{target_type}"]
        
        # Création d'un essai factice pour la création du modèle
        class DummyTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_categorical(self, name, choices):
                return self.params[name]
            
            def suggest_int(self, name, low, high):
                return self.params[name]
            
            def suggest_float(self, name, low, high, log=False):
                return self.params[name]
        
        dummy_trial = DummyTrial(best_params)
        
        # Création du modèle avec les meilleurs hyperparamètres
        model = self.create_transformer_model(dummy_trial, target_type)
        
        # Sélection des données d'entraînement en fonction du type de modèle
        if target_type == "main":
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
                filepath=f"models/auto_ml/transformer_{target_type}_best.h5",
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entraînement du modèle
        history = model.fit(
            self.X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarde du modèle final
        model.save(f"models/auto_ml/transformer_{target_type}_final.h5")
        
        # Sauvegarde du modèle dans le dictionnaire
        self.best_models[f"transformer_{target_type}"] = model
        
        # Visualisation de l'historique d'entraînement
        self.plot_training_history(history, f"transformer_{target_type}")
        
        print(f"✅ Meilleur modèle Transformer {target_type} entraîné avec succès.")
        
        return model
    
    def train_best_lstm_transformer(self, target_type="main"):
        """
        Entraîne le meilleur modèle LSTM-Transformer avec les hyperparamètres optimisés.
        """
        print(f"Entraînement du meilleur modèle LSTM-Transformer pour {target_type}...")
        
        # Vérification de l'existence des meilleurs hyperparamètres
        if f"lstm_transformer_{target_type}" not in self.best_params:
            print(f"❌ Meilleurs hyperparamètres non trouvés pour le modèle LSTM-Transformer {target_type}.")
            return None
        
        # Récupération des meilleurs hyperparamètres
        best_params = self.best_params[f"lstm_transformer_{target_type}"]
        
        # Création d'un essai factice pour la création du modèle
        class DummyTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_categorical(self, name, choices):
                return self.params[name]
            
            def suggest_int(self, name, low, high):
                return self.params[name]
            
            def suggest_float(self, name, low, high, log=False):
                return self.params[name]
        
        dummy_trial = DummyTrial(best_params)
        
        # Création du modèle avec les meilleurs hyperparamètres
        model = self.create_lstm_transformer_model(dummy_trial, target_type)
        
        # Sélection des données d'entraînement en fonction du type de modèle
        if target_type == "main":
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
                filepath=f"models/auto_ml/lstm_transformer_{target_type}_best.h5",
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entraînement du modèle
        history = model.fit(
            self.X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarde du modèle final
        model.save(f"models/auto_ml/lstm_transformer_{target_type}_final.h5")
        
        # Sauvegarde du modèle dans le dictionnaire
        self.best_models[f"lstm_transformer_{target_type}"] = model
        
        # Visualisation de l'historique d'entraînement
        self.plot_training_history(history, f"lstm_transformer_{target_type}")
        
        print(f"✅ Meilleur modèle LSTM-Transformer {target_type} entraîné avec succès.")
        
        return model
    
    def plot_training_history(self, history, name):
        """
        Visualise l'historique d'entraînement d'un modèle.
        """
        # Création du répertoire pour les visualisations
        os.makedirs(f"results/auto_ml/{name}", exist_ok=True)
        
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
        plt.savefig(f"results/auto_ml/{name}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_models(self):
        """
        Évalue les performances des modèles optimisés sur l'ensemble de test.
        """
        print("Évaluation des modèles optimisés sur l'ensemble de test...")
        
        # Dictionnaire pour stocker les résultats d'évaluation
        evaluation_results = {}
        
        # Évaluation des modèles pour les numéros principaux
        for name in [model_name for model_name in self.best_models.keys() if "main" in model_name]:
            print(f"Évaluation du modèle {name}...")
            
            # Prédiction sur l'ensemble de test
            y_pred = self.best_models[name].predict(self.X_test)
            
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
            mse = np.mean(np.sum((y_true_real - y_pred_real) ** 2, axis=1))
            mae = np.mean(np.sum(np.abs(y_true_real - y_pred_real), axis=1))
            
            # Calcul du nombre moyen de numéros correctement prédits
            correct_numbers = []
            for i in range(len(y_true_real)):
                correct = len(set(y_true_real[i]) & set(y_pred_real[i]))
                correct_numbers.append(correct)
            
            avg_correct = np.mean(correct_numbers)
            
            # Sauvegarde des résultats
            evaluation_results[name] = {
                "mse": float(mse),
                "mae": float(mae),
                "avg_correct_numbers": float(avg_correct)
            }
            
            print(f"   - MSE: {mse:.4f}")
            print(f"   - MAE: {mae:.4f}")
            print(f"   - Nombre moyen de numéros correctement prédits: {avg_correct:.2f}/5")
        
        # Évaluation des modèles pour les étoiles
        for name in [model_name for model_name in self.best_models.keys() if "stars" in model_name]:
            print(f"Évaluation du modèle {name}...")
            
            # Prédiction sur l'ensemble de test
            y_pred = self.best_models[name].predict(self.X_test)
            
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
            mse = np.mean(np.sum((y_true_real - y_pred_real) ** 2, axis=1))
            mae = np.mean(np.sum(np.abs(y_true_real - y_pred_real), axis=1))
            
            # Calcul du nombre moyen d'étoiles correctement prédites
            correct_stars = []
            for i in range(len(y_true_real)):
                correct = len(set(y_true_real[i]) & set(y_pred_real[i]))
                correct_stars.append(correct)
            
            avg_correct = np.mean(correct_stars)
            
            # Sauvegarde des résultats
            evaluation_results[name] = {
                "mse": float(mse),
                "mae": float(mae),
                "avg_correct_stars": float(avg_correct)
            }
            
            print(f"   - MSE: {mse:.4f}")
            print(f"   - MAE: {mae:.4f}")
            print(f"   - Nombre moyen d'étoiles correctement prédites: {avg_correct:.2f}/2")
        
        # Sauvegarde des résultats d'évaluation
        self.results["evaluation"] = evaluation_results
        
        # Sauvegarde des résultats dans un fichier
        with open("results/auto_ml/evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        print(f"✅ Résultats d'évaluation sauvegardés dans results/auto_ml/evaluation_results.json")
        
        return evaluation_results
    
    def generate_predictions(self):
        """
        Génère des prédictions pour le prochain tirage avec les modèles optimisés.
        """
        print("Génération de prédictions pour le prochain tirage...")
        
        # Vérification de l'existence des modèles
        if not self.best_models:
            print("❌ Aucun modèle disponible pour générer des prédictions.")
            return None
        
        # Préparation des données pour la prédiction
        # On utilise les dernières séquences de l'ensemble de test
        last_sequence = self.X_test[-1:]
        
        # Dictionnaire pour stocker les prédictions
        predictions = {}
        
        # Génération des prédictions pour les numéros principaux
        for name in [model_name for model_name in self.best_models.keys() if "main" in model_name]:
            print(f"Génération de prédictions avec le modèle {name}...")
            
            # Prédiction
            y_pred = self.best_models[name].predict(last_sequence)
            
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
        for name in [model_name for model_name in self.best_models.keys() if "stars" in model_name]:
            print(f"Génération de prédictions avec le modèle {name}...")
            
            # Prédiction
            y_pred = self.best_models[name].predict(last_sequence)
            
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
        
        # Sauvegarde des prédictions
        self.results["predictions"] = predictions
        
        # Sauvegarde des prédictions dans un fichier
        with open("results/auto_ml/predictions.json", 'w') as f:
            json.dump(predictions, f, indent=4)
        
        # Création d'un fichier de prédiction plus lisible
        with open("results/auto_ml/prediction.txt", 'w') as f:
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
        
        print(f"✅ Prédictions sauvegardées dans results/auto_ml/predictions.json et results/auto_ml/prediction.txt")
        
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
    
    def run_full_pipeline(self, quick_mode=True):
        """
        Exécute le pipeline complet d'optimisation des hyperparamètres et d'auto-adaptation.
        """
        print("Démarrage du pipeline d'optimisation des hyperparamètres et d'auto-adaptation...")
        
        # 1. Chargement des données
        self.load_data()
        
        # 2. Préparation des données
        self.prepare_data()
        
        # 3. Optimisation des hyperparamètres
        if quick_mode:
            # Mode rapide : moins d'essais
            self.optimize_transformer(target_type="main", n_trials=5)
            self.optimize_transformer(target_type="stars", n_trials=5)
        else:
            # Mode complet : tous les modèles
            self.optimize_transformer(target_type="main", n_trials=50)
            self.optimize_transformer(target_type="stars", n_trials=50)
            self.optimize_lstm_transformer(target_type="main", n_trials=50)
            self.optimize_lstm_transformer(target_type="stars", n_trials=50)
        
        # 4. Évaluation des modèles
        self.evaluate_models()
        
        # 5. Génération de prédictions
        self.generate_predictions()
        
        print("✅ Pipeline d'optimisation des hyperparamètres et d'auto-adaptation terminé avec succès!")
        
        return self.results

class EuromillionsUltraOptimized:
    """
    Classe pour l'intégration finale et l'évaluation du système ultra-optimisé
    de prédiction des numéros de l'Euromillions.
    """
    
    def __init__(self):
        """
        Initialise la classe pour l'intégration finale.
        """
        self.reinforcement_learning = None
        self.auto_ml = None
        self.predictions = {}
        self.results = {}
    
    def initialize_components(self, data_path="data_enriched/euromillions_enhanced_dataset.csv"):
        """
        Initialise les composants du système ultra-optimisé.
        """
        print("Initialisation des composants du système ultra-optimisé...")
        
        # Initialisation de l'apprentissage par renforcement
        self.reinforcement_learning = EuromillionsReinforcementLearning(data_path=data_path)
        
        # Initialisation de l'auto-ML
        self.auto_ml = EuromillionsAutoML(data_path=data_path)
        
        print("✅ Composants du système ultra-optimisé initialisés.")
        
        return True
    
    def load_models(self):
        """
        Charge les modèles entraînés.
        """
        print("Chargement des modèles entraînés...")
        
        # Chargement des modèles d'apprentissage par renforcement
        try:
            # Création de l'environnement
            self.reinforcement_learning.create_environment()
            
            # Chargement du modèle PPO
            if os.path.exists("models/reinforcement/ppo_euromillions.zip"):
                from stable_baselines3 import PPO
                model = PPO.load("models/reinforcement/ppo_euromillions")
                self.reinforcement_learning.models["ppo"] = model
                print("✅ Modèle PPO chargé avec succès.")
            else:
                print("⚠️ Modèle PPO non trouvé.")
            
            # Chargement du modèle A2C
            if os.path.exists("models/reinforcement/a2c_euromillions.zip"):
                from stable_baselines3 import A2C
                model = A2C.load("models/reinforcement/a2c_euromillions")
                self.reinforcement_learning.models["a2c"] = model
                print("✅ Modèle A2C chargé avec succès.")
            else:
                print("⚠️ Modèle A2C non trouvé.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles d'apprentissage par renforcement : {e}")
        
        # Chargement des modèles auto-ML
        try:
            # Préparation des données
            self.auto_ml.load_data()
            self.auto_ml.prepare_data()
            
            # Chargement du modèle Transformer pour les numéros principaux
            if os.path.exists("models/auto_ml/transformer_main_final.h5"):
                model = keras.models.load_model("models/auto_ml/transformer_main_final.h5")
                self.auto_ml.best_models["transformer_main"] = model
                print("✅ Modèle Transformer pour les numéros principaux chargé avec succès.")
            else:
                print("⚠️ Modèle Transformer pour les numéros principaux non trouvé.")
            
            # Chargement du modèle Transformer pour les étoiles
            if os.path.exists("models/auto_ml/transformer_stars_final.h5"):
                model = keras.models.load_model("models/auto_ml/transformer_stars_final.h5")
                self.auto_ml.best_models["transformer_stars"] = model
                print("✅ Modèle Transformer pour les étoiles chargé avec succès.")
            else:
                print("⚠️ Modèle Transformer pour les étoiles non trouvé.")
            
            # Chargement du modèle LSTM-Transformer pour les numéros principaux
            if os.path.exists("models/auto_ml/lstm_transformer_main_final.h5"):
                model = keras.models.load_model("models/auto_ml/lstm_transformer_main_final.h5")
                self.auto_ml.best_models["lstm_transformer_main"] = model
                print("✅ Modèle LSTM-Transformer pour les numéros principaux chargé avec succès.")
            else:
                print("⚠️ Modèle LSTM-Transformer pour les numéros principaux non trouvé.")
            
            # Chargement du modèle LSTM-Transformer pour les étoiles
            if os.path.exists("models/auto_ml/lstm_transformer_stars_final.h5"):
                model = keras.models.load_model("models/auto_ml/lstm_transformer_stars_final.h5")
                self.auto_ml.best_models["lstm_transformer_stars"] = model
                print("✅ Modèle LSTM-Transformer pour les étoiles chargé avec succès.")
            else:
                print("⚠️ Modèle LSTM-Transformer pour les étoiles non trouvé.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles auto-ML : {e}")
        
        print("✅ Chargement des modèles terminé.")
        
        return True
    
    def generate_ensemble_predictions(self, n_predictions=10):
        """
        Génère des prédictions avec l'ensemble des modèles.
        """
        print("Génération de prédictions avec l'ensemble des modèles...")
        
        # Dictionnaire pour stocker les prédictions
        predictions = {}
        
        # Génération de prédictions avec les modèles d'apprentissage par renforcement
        if self.reinforcement_learning and self.reinforcement_learning.models:
            for model_name in self.reinforcement_learning.models:
                try:
                    model_predictions = self.reinforcement_learning.generate_predictions(model_name=model_name, n_predictions=n_predictions)
                    predictions[f"rl_{model_name}"] = {
                        "main_numbers": model_predictions[0]["main_numbers"],
                        "stars": model_predictions[0]["stars"]
                    }
                    print(f"✅ Prédictions générées avec le modèle d'apprentissage par renforcement {model_name}.")
                except Exception as e:
                    print(f"❌ Erreur lors de la génération de prédictions avec le modèle {model_name} : {e}")
        
        # Génération de prédictions avec les modèles auto-ML
        if self.auto_ml and self.auto_ml.best_models:
            # Génération de prédictions
            auto_ml_predictions = self.auto_ml.generate_predictions()
            
            # Ajout des prédictions au dictionnaire
            for model_name, pred in auto_ml_predictions.items():
                if "main" in model_name:
                    if "transformer_main" not in predictions:
                        predictions["transformer_main"] = {"main_numbers": pred}
                    if "lstm_transformer_main" not in predictions and "lstm" in model_name:
                        predictions["lstm_transformer_main"] = {"main_numbers": pred}
                elif "stars" in model_name:
                    if "transformer_stars" not in predictions:
                        predictions["transformer_stars"] = {"stars": pred}
                    if "lstm_transformer_stars" not in predictions and "lstm" in model_name:
                        predictions["lstm_transformer_stars"] = {"stars": pred}
            
            print(f"✅ Prédictions générées avec les modèles auto-ML.")
        
        # Sauvegarde des prédictions
        self.predictions = predictions
        
        # Sauvegarde des prédictions dans un fichier
        with open("results/ensemble_predictions.json", 'w') as f:
            json.dump(predictions, f, indent=4)
        
        # Création d'un fichier de prédiction plus lisible
        with open("results/ensemble_prediction.txt", 'w') as f:
            f.write("Prédictions pour le prochain tirage de l'Euromillions\n")
            f.write("=================================================\n\n")
            
            f.write("Date de génération: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            f.write("Prédictions des numéros principaux:\n")
            main_predictions = []
            for name, pred in predictions.items():
                if "main_numbers" in pred:
                    main_numbers = pred["main_numbers"]
                    f.write(f"   - Modèle {name}: {', '.join(map(str, main_numbers))}\n")
                    main_predictions.append(main_numbers)
            
            f.write("\nPrédictions des étoiles:\n")
            stars_predictions = []
            for name, pred in predictions.items():
                if "stars" in pred:
                    stars = pred["stars"]
                    f.write(f"   - Modèle {name}: {', '.join(map(str, stars))}\n")
                    stars_predictions.append(stars)
            
            f.write("\nPrédiction finale (consensus):\n")
            
            # Calcul du consensus pour les numéros principaux
            main_consensus = self.calculate_consensus(main_predictions, 5, 50)
            
            # Calcul du consensus pour les étoiles
            stars_consensus = self.calculate_consensus(stars_predictions, 2, 12)
            
            f.write(f"   - Numéros principaux: {', '.join(map(str, main_consensus))}\n")
            f.write(f"   - Étoiles: {', '.join(map(str, stars_consensus))}\n")
        
        print(f"✅ Prédictions d'ensemble sauvegardées dans results/ensemble_predictions.json et results/ensemble_prediction.txt")
        
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
    
    def run_full_pipeline(self, quick_mode=True):
        """
        Exécute le pipeline complet d'intégration et d'évaluation du système ultra-optimisé.
        """
        print("Démarrage du pipeline d'intégration et d'évaluation du système ultra-optimisé...")
        
        # 1. Initialisation des composants
        self.initialize_components()
        
        # 2. Chargement des modèles
        self.load_models()
        
        # 3. Génération de prédictions d'ensemble
        if quick_mode:
            # Mode rapide : moins de prédictions
            self.generate_ensemble_predictions(n_predictions=5)
        else:
            # Mode complet : plus de prédictions
            self.generate_ensemble_predictions(n_predictions=10)
        
        print("✅ Pipeline d'intégration et d'évaluation du système ultra-optimisé terminé avec succès!")
        
        return self.predictions

# Exécution du pipeline
if __name__ == "__main__":
    # Vérification de l'existence du fichier de données enrichies
    if os.path.exists("data_enriched/euromillions_enhanced_dataset.csv"):
        # Mode complet
        print("Exécution du pipeline complet d'apprentissage par renforcement et d'auto-adaptation...")
        
        # Apprentissage par renforcement
        rl = EuromillionsReinforcementLearning()
        rl.run_full_pipeline(quick_mode=True)
        
        # Auto-ML
        auto_ml = EuromillionsAutoML()
        auto_ml.run_full_pipeline(quick_mode=True)
        
        # Intégration finale
        ultra = EuromillionsUltraOptimized()
        ultra.run_full_pipeline(quick_mode=True)
    else:
        # Mode rapide avec données synthétiques
        print("⚠️ Fichier de données enrichies non trouvé. Exécution en mode rapide avec données synthétiques.")
        
        # Apprentissage par renforcement
        rl = EuromillionsReinforcementLearning()
        rl.run_full_pipeline(quick_mode=True)
        
        # Auto-ML
        auto_ml = EuromillionsAutoML()
        auto_ml.run_full_pipeline(quick_mode=True)
        
        # Intégration finale
        ultra = EuromillionsUltraOptimized()
        ultra.run_full_pipeline(quick_mode=True)

