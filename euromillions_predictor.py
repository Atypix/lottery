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
# import matplotlib.pyplot as plt # Commented for CLI JSON output
# import seaborn as sns # Commented for CLI JSON output
from datetime import datetime, date as datetime_date # Added datetime_date
import random
import argparse
# import tensorflow as tf # Commented, TF not used in quick prediction
# from tensorflow import keras # Commented
from common.date_utils import get_next_euromillions_draw_date # Added
# json, os, sys are already imported by virtue of being used in the file.
# Explicitly adding for clarity if they were missing before, but they are not.

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
        # Updated data_path to prefer data/ subdirectory
        self.data_path_primary = "data/euromillions_enhanced_dataset.csv"
        self.data_path_fallback = "euromillions_enhanced_dataset.csv"
        
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
    
    def generate_quick_prediction(self, target_date_override_str=None): # Added target_date_override_str
        """
        Génère une prédiction rapide basée sur des heuristiques et l'analyse statistique.
        """
        # print("Génération d'une prédiction rapide...") # Suppressed for JSON output

        # Determine actual data path to use
        actual_data_file_to_use = None
        if os.path.exists(self.data_path_primary):
            actual_data_file_to_use = self.data_path_primary
        elif os.path.exists(self.data_path_fallback):
            actual_data_file_to_use = self.data_path_fallback
            print(f"ℹ️  Fichier de données trouvé dans le répertoire courant: {self.data_path_fallback}")

        if not actual_data_file_to_use:
            print(f"❌ Fichier de données non trouvé ({self.data_path_primary} ou {self.data_path_fallback}).")
            print("⚠️ Génération d'une prédiction aléatoire.")
            
            # Génération de numéros aléatoires
            main_numbers = sorted(random.sample(range(1, 51), 5))
            stars = sorted(random.sample(range(1, 13), 2))
            
            # Determine target_date_str for this prediction
            current_target_date_str = None
            if target_date_override_str:
                current_target_date_str = target_date_override_str
            else:
                data_file_for_next_date = None
                if hasattr(self, 'actual_data_file_to_use') and self.actual_data_file_to_use and os.path.exists(self.actual_data_file_to_use):
                    data_file_for_next_date = self.actual_data_file_to_use
                elif hasattr(self, 'data_path_primary') and os.path.exists(self.data_path_primary):
                    data_file_for_next_date = self.data_path_primary
                elif hasattr(self, 'data_path_fallback') and os.path.exists(self.data_path_fallback):
                    data_file_for_next_date = self.data_path_fallback

                if data_file_for_next_date:
                    current_target_date_str = get_next_euromillions_draw_date(data_file_for_next_date).strftime('%Y-%m-%d')
                else:
                    current_target_date_str = datetime.now().date().strftime('%Y-%m-%d')


            prediction = {
                "main_numbers": main_numbers,
                "stars": stars,
                "confidence": 0.2, # Default confidence for random
                # "method": "Aléatoire", # Not part of the final JSON schema
                # "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Not part of final schema
                "date_tirage_cible": current_target_date_str
            }
            
            return prediction
        
        # Chargement des données
        try:
            df = pd.read_csv(actual_data_file_to_use)
            print(f"✅ Données chargées avec succès depuis {actual_data_file_to_use}: {len(df)} tirages.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données depuis {actual_data_file_to_use}: {e}")
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
            "confidence": 0.5,
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
        print(f"Niveau de confiance : {prediction['confidence']:.2f}")
        
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
                f.write(f"Niveau de confiance : {prediction['confidence']:.2f}\n\n")
                
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
            # Default action: Generate quick prediction and output as JSON
            # Pass the date argument from CLI to generate_quick_prediction
            # The generate_quick_prediction method will be modified to accept args.date
            prediction_data = self.generate_quick_prediction(target_date_override_str=args.date if hasattr(args, 'date') else None)
            
            if prediction_data: # generate_quick_prediction now returns the dict
                output_dict = {
                    "nom_predicteur": "euromillions_predictor",
                    "numeros": prediction_data.get('main_numbers'),
                    "etoiles": prediction_data.get('stars'),
                    "date_tirage_cible": prediction_data.get('date_tirage_cible'),
                    "confidence": prediction_data.get('confidence', 5.0), # Default confidence
                    "categorie": "Scientifique"
                }
                print(json.dumps(output_dict))
                # self.display_prediction(prediction_data) # Suppressed for JSON output
                # self.save_prediction(prediction_data) # Suppressed for JSON output

def main():
    """
    Fonction principale.
    """
    # Analyse des arguments de la ligne de commande
    # Keep existing args, add --date
    parser = argparse.ArgumentParser(description="Euromillions Predictor CLI.")
    
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualise les prédictions existantes.")
    parser.add_argument("-a", "--advanced", action="store_true", help="Vérifie l'existence des modèles avancés.")
    parser.add_argument("--date", type=str, help="Target draw date in YYYY-MM-DD format for quick prediction.")
    
    args = parser.parse_args()
    
    # Création de l'interface utilisateur
    predictor = EuromillionsPredictor()
    
    # Exécution de l'interface utilisateur
    # The run method will now handle the args.date for the default case
    predictor.run(args)

if __name__ == "__main__":
    main()

