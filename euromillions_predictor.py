#!/usr/bin/env python3
"""
Interface utilisateur pour le système de prédiction Euromillions ultra-optimisé.
Ce script permet de générer facilement des prédictions et de visualiser les résultats.
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

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class EuromillionsPredictor:
    """
    Interface utilisateur pour le système de prédiction Euromillions ultra-optimisé.
    """
    
    def __init__(self):
        """
        Initialise l'interface utilisateur.
        """
        self.models_dir = "models/advanced"
        self.results_dir = "results/advanced"
        self.data_path = "euromillions_enhanced_dataset.csv"
        
        # Création des répertoires si nécessaires
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Vérification de l'existence des modèles
        self.models_exist = self.check_models_exist()
        
        # Chargement des prédictions si elles existent
        self.predictions = self.load_predictions()
    
    def check_models_exist(self):
        """
        Vérifie si les modèles existent.
        """
        required_models = [
            "transformer_main_final.h5",
            "transformer_stars_final.h5"
        ]
        
        for model_file in required_models:
            if not os.path.exists(os.path.join(self.models_dir, model_file)):
                return False
        
        return True
    
    def load_predictions(self):
        """
        Charge les prédictions existantes.
        """
        predictions_path = os.path.join(self.results_dir, "predictions.json")
        
        if os.path.exists(predictions_path):
            try:
                with open(predictions_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement des prédictions : {e}")
                return None
        
        return None
    
    def generate_quick_prediction(self):
        """
        Génère une prédiction rapide basée sur des heuristiques et l'analyse statistique.
        """
        print("Génération d'une prédiction rapide...")
        
        # Vérification de l'existence du fichier de données
        if not os.path.exists(self.data_path):
            print(f"❌ Fichier de données {self.data_path} non trouvé.")
            print("⚠️ Génération d'une prédiction aléatoire.")
            
            # Génération de numéros aléatoires
            main_numbers = sorted(random.sample(range(1, 51), 5))
            stars = sorted(random.sample(range(1, 13), 2))
            
            prediction = {
                "main_numbers": main_numbers,
                "stars": stars,
                "confidence": "Faible (prédiction aléatoire)",
                "method": "Aléatoire",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return prediction
        
        # Chargement des données
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Données chargées avec succès : {len(df)} tirages.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            return None
        
        # Analyse statistique simple
        # 1. Fréquence des numéros principaux
        main_numbers_freq = {}
        for i in range(1, 6):
            col = f'N{i}'
            for num in df[col]:
                if num in main_numbers_freq:
                    main_numbers_freq[num] += 1
                else:
                    main_numbers_freq[num] = 1
        
        # 2. Fréquence des étoiles
        stars_freq = {}
        for i in range(1, 3):
            col = f'E{i}'
            for num in df[col]:
                if num in stars_freq:
                    stars_freq[num] += 1
                else:
                    stars_freq[num] = 1
        
        # 3. Numéros récents (derniers 10 tirages)
        recent_draws = df.tail(10)
        recent_main_numbers = []
        for i in range(1, 6):
            col = f'N{i}'
            recent_main_numbers.extend(recent_draws[col].tolist())
        
        recent_stars = []
        for i in range(1, 3):
            col = f'E{i}'
            recent_stars.extend(recent_draws[col].tolist())
        
        # 4. Numéros "dus" (non sortis depuis longtemps)
        all_main_numbers = set(range(1, 51))
        all_stars = set(range(1, 13))
        
        recent_main_set = set(recent_main_numbers)
        recent_stars_set = set(recent_stars)
        
        due_main_numbers = all_main_numbers - recent_main_set
        due_stars = all_stars - recent_stars_set
        
        # 5. Génération de la prédiction
        # Mélange de numéros fréquents et de numéros "dus"
        # 60% fréquents, 40% dus
        
        # Tri des numéros par fréquence
        sorted_main_freq = sorted(main_numbers_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_stars_freq = sorted(stars_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection des numéros fréquents
        frequent_main = [num for num, _ in sorted_main_freq[:10]]
        frequent_stars = [num for num, _ in sorted_stars_freq[:5]]
        
        # Mélange de numéros fréquents et dus
        selected_main = []
        selected_main.extend(random.sample(frequent_main, 3))  # 3 numéros fréquents
        selected_main.extend(random.sample(list(due_main_numbers), 2))  # 2 numéros dus
        
        selected_stars = []
        selected_stars.extend(random.sample(frequent_stars, 1))  # 1 étoile fréquente
        selected_stars.extend(random.sample(list(due_stars), 1))  # 1 étoile due
        
        # Tri des numéros
        selected_main.sort()
        selected_stars.sort()
        
        # Création de la prédiction
        prediction = {
            "main_numbers": selected_main,
            "stars": selected_stars,
            "confidence": "Moyenne (analyse statistique simple)",
            "method": "Analyse statistique",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return prediction
    
    def display_prediction(self, prediction):
        """
        Affiche une prédiction de manière formatée.
        """
        print("\n" + "=" * 50)
        print("PRÉDICTION EUROMILLIONS")
        print("=" * 50)
        
        print(f"\nDate de génération : {prediction['timestamp']}")
        print(f"Méthode : {prediction['method']}")
        print(f"Niveau de confiance : {prediction['confidence']}")
        
        print("\nNUMÉROS PRINCIPAUX :")
        print(" ".join([f"[{num:2d}]" for num in prediction['main_numbers']]))
        
        print("\nÉTOILES :")
        print(" ".join([f"[{num:2d}]" for num in prediction['stars']]))
        
        print("\n" + "=" * 50)
        print("Bonne chance !")
        print("=" * 50 + "\n")
    
    def save_prediction(self, prediction, filename="quick_prediction.json"):
        """
        Sauvegarde une prédiction dans un fichier.
        """
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(prediction, f, indent=4)
            
            print(f"✅ Prédiction sauvegardée dans {filepath}")
            
            # Création d'un fichier texte plus lisible
            txt_filepath = os.path.join(self.results_dir, "quick_prediction.txt")
            
            with open(txt_filepath, 'w') as f:
                f.write("Prédiction pour le prochain tirage de l'Euromillions\n")
                f.write("=================================================\n\n")
                
                f.write(f"Date de génération : {prediction['timestamp']}\n")
                f.write(f"Méthode : {prediction['method']}\n")
                f.write(f"Niveau de confiance : {prediction['confidence']}\n\n")
                
                f.write("Numéros principaux :\n")
                f.write(", ".join(map(str, prediction['main_numbers'])) + "\n\n")
                
                f.write("Étoiles :\n")
                f.write(", ".join(map(str, prediction['stars'])) + "\n\n")
                
                f.write("Bonne chance !\n")
            
            print(f"✅ Prédiction sauvegardée en format texte dans {txt_filepath}")
            
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde de la prédiction : {e}")
            return False
    
    def visualize_predictions(self):
        """
        Visualise les prédictions existantes.
        """
        if not self.predictions:
            print("❌ Aucune prédiction disponible pour la visualisation.")
            return False
        
        try:
            # Création du répertoire pour les visualisations
            viz_dir = os.path.join(self.results_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Extraction des prédictions des numéros principaux
            main_predictions = {}
            for name, numbers in self.predictions.items():
                if "main" in name:
                    main_predictions[name] = numbers
            
            # Extraction des prédictions des étoiles
            stars_predictions = {}
            for name, numbers in self.predictions.items():
                if "stars" in name:
                    stars_predictions[name] = numbers
            
            # Visualisation des numéros principaux
            plt.figure(figsize=(12, 6))
            
            # Création d'un tableau pour les numéros principaux
            main_data = np.zeros((len(main_predictions), 50))
            
            for i, (name, numbers) in enumerate(main_predictions.items()):
                for num in numbers:
                    main_data[i, num-1] = 1
            
            # Création du heatmap
            ax = sns.heatmap(main_data, cmap="viridis", cbar=False)
            
            # Configuration des axes
            ax.set_yticks(np.arange(len(main_predictions)) + 0.5)
            ax.set_yticklabels(main_predictions.keys())
            
            ax.set_xticks(np.arange(0, 50, 5) + 0.5)
            ax.set_xticklabels([str(i+1) for i in range(0, 50, 5)])
            
            plt.title("Prédictions des numéros principaux")
            plt.xlabel("Numéros (1-50)")
            plt.ylabel("Modèles")
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "main_numbers_predictions.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Visualisation des étoiles
            plt.figure(figsize=(8, 4))
            
            # Création d'un tableau pour les étoiles
            stars_data = np.zeros((len(stars_predictions), 12))
            
            for i, (name, numbers) in enumerate(stars_predictions.items()):
                for num in numbers:
                    stars_data[i, num-1] = 1
            
            # Création du heatmap
            ax = sns.heatmap(stars_data, cmap="viridis", cbar=False)
            
            # Configuration des axes
            ax.set_yticks(np.arange(len(stars_predictions)) + 0.5)
            ax.set_yticklabels(stars_predictions.keys())
            
            ax.set_xticks(np.arange(12) + 0.5)
            ax.set_xticklabels([str(i+1) for i in range(12)])
            
            plt.title("Prédictions des étoiles")
            plt.xlabel("Numéros (1-12)")
            plt.ylabel("Modèles")
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "stars_predictions.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Visualisations sauvegardées dans {viz_dir}")
            
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la visualisation des prédictions : {e}")
            return False
    
    def run(self, args):
        """
        Exécute l'interface utilisateur en fonction des arguments.
        """
        if args.visualize:
            # Visualisation des prédictions existantes
            if self.predictions:
                self.visualize_predictions()
            else:
                print("❌ Aucune prédiction disponible pour la visualisation.")
        
        elif args.advanced:
            # Vérification de l'existence des modèles avancés
            if self.models_exist:
                print("Les modèles avancés existent. Pour générer une prédiction avancée, exécutez :")
                print("python euromillions_ultra_optimized.py")
            else:
                print("❌ Les modèles avancés n'existent pas encore.")
                print("Pour créer et entraîner les modèles avancés, exécutez :")
                print("python euromillions_ultra_optimized.py")
        
        else:
            # Génération d'une prédiction rapide
            prediction = self.generate_quick_prediction()
            
            if prediction:
                # Affichage de la prédiction
                self.display_prediction(prediction)
                
                # Sauvegarde de la prédiction
                self.save_prediction(prediction)

def main():
    """
    Fonction principale.
    """
    # Analyse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Interface utilisateur pour le système de prédiction Euromillions ultra-optimisé.")
    
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualise les prédictions existantes.")
    parser.add_argument("-a", "--advanced", action="store_true", help="Vérifie l'existence des modèles avancés.")
    
    args = parser.parse_args()
    
    # Création de l'interface utilisateur
    predictor = EuromillionsPredictor()
    
    # Exécution de l'interface utilisateur
    predictor.run(args)

if __name__ == "__main__":
    main()

