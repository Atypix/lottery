#!/usr/bin/env python3
"""
Script pour prédire les numéros de l'Euromillions en utilisant un modèle TensorFlow pré-entraîné.
"""

import os
import sys
import argparse
import tensorflow as tf # Keep for load_model, comment out if not directly used
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime # Already datetime.datetime
from common.date_utils import get_next_euromillions_draw_date, date as datetime_date # Added
import json # Added
# os, sys, argparse are already imported

def load_models(models_dir):
    """
    Charge les modèles pré-entraînés.
    
    Args:
        models_dir: Répertoire contenant les modèles
    
    Returns:
        Tuple (modèle pour les numéros principaux, modèle pour les étoiles)
    """
    # Define model paths directly
    main_model_path = os.path.join(models_dir, "tf_main_std", "final_model.h5")
    stars_model_path = os.path.join(models_dir, "tf_stars_std", "final_model.h5")
    
    if not os.path.exists(main_model_path) or not os.path.exists(stars_model_path):
        print("Erreur : Fichiers de modèle (tf_main_std/final_model.h5 or tf_stars_std/final_model.h5) non trouvés dans le répertoire des modèles. Veuillez d'abord entraîner les modèles.") # Modified error message
        sys.exit(1)
    
    main_model = tf.keras.models.load_model(main_model_path)
    stars_model = tf.keras.models.load_model(stars_model_path)
    
    return main_model, stars_model

def prepare_data(data_file, sequence_length=10):
    """
    Prépare les données pour la prédiction.
    
    Args:
        data_file: Chemin vers le fichier CSV contenant les données
        sequence_length: Longueur de la séquence pour la prédiction
    
    Returns:
        Tuple (dernière séquence de numéros principaux, dernière séquence d'étoiles, scalers)
    """
    # Charger les données
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Séparer les numéros principaux et les étoiles
    main_numbers = df[['N1', 'N2', 'N3', 'N4', 'N5']].values
    stars = df[['E1', 'E2']].values
    
    # Normaliser les données
    main_scaler = MinMaxScaler(feature_range=(0, 1))
    stars_scaler = MinMaxScaler(feature_range=(0, 1))
    
    main_normalized = main_scaler.fit_transform(main_numbers)
    stars_normalized = stars_scaler.fit_transform(stars)
    
    # Obtenir les dernières séquences
    X_main_last = main_normalized[-sequence_length:]
    X_stars_last = stars_normalized[-sequence_length:]
    
    return X_main_last, X_stars_last, main_scaler, stars_scaler

def predict_next_numbers(main_model, stars_model, X_main_last, X_stars_last, main_scaler, stars_scaler):
    """
    Prédit les numéros pour le prochain tirage.
    
    Args:
        main_model: Modèle pour les numéros principaux
        stars_model: Modèle pour les étoiles
        X_main_last: Dernière séquence de numéros principaux
        X_stars_last: Dernière séquence d'étoiles
        main_scaler: Scaler pour les numéros principaux
        stars_scaler: Scaler pour les étoiles
    
    Returns:
        Tuple (numéros principaux prédits, étoiles prédites)
    """
    # Prédire les numéros normalisés
    main_pred_normalized = main_model.predict(np.array([X_main_last]))
    stars_pred_normalized = stars_model.predict(np.array([X_stars_last]))
    
    # Inverser la normalisation
    main_pred = main_scaler.inverse_transform(main_pred_normalized)
    stars_pred = stars_scaler.inverse_transform(stars_pred_normalized)
    
    # Arrondir et trier les numéros
    main_numbers = np.round(main_pred[0]).astype(int)
    star_numbers = np.round(stars_pred[0]).astype(int)
    
    # S'assurer que les numéros sont dans les plages valides
    main_numbers = np.clip(main_numbers, 1, 50)
    star_numbers = np.clip(star_numbers, 1, 12)
    
    # Éliminer les doublons potentiels
    main_numbers = np.unique(main_numbers)
    star_numbers = np.unique(star_numbers)
    
    # Si nous avons moins de 5 numéros principaux ou 2 étoiles après élimination des doublons,
    # compléter avec des numéros aléatoires
    while len(main_numbers) < 5:
        new_num = np.random.randint(1, 51)
        if new_num not in main_numbers:
            main_numbers = np.append(main_numbers, new_num)
    
    while len(star_numbers) < 2:
        new_star = np.random.randint(1, 13)
        if new_star not in star_numbers:
            star_numbers = np.append(star_numbers, new_star)
    
    # Trier les numéros
    main_numbers = np.sort(main_numbers)[:5]  # Prendre les 5 premiers
    star_numbers = np.sort(star_numbers)[:2]  # Prendre les 2 premiers
    
    return main_numbers, star_numbers

def main():
    parser = argparse.ArgumentParser(description="Prédire les numéros de l'Euromillions")
    parser.add_argument("--data", default="data/euromillions_dataset.csv", help="Chemin vers le fichier CSV des données")
    parser.add_argument("--models-dir", default="models", help="Répertoire contenant les modèles")
    # parser.add_argument("--output", default="prediction.txt", help="Fichier de sortie pour la prédiction") # Commented out
    parser.add_argument("--date", type=str, help="Target draw date in YYYY-MM-DD format.") # Added
    args = parser.parse_args()
    
    # Vérifier si les fichiers et répertoires existent
    # Adjusted to check primary and fallback for data arg
    data_path_primary = args.data
    data_path_fallback = os.path.basename(args.data) # e.g. "euromillions_dataset.csv" if args.data was "data/..."

    actual_data_path = None
    if os.path.exists(data_path_primary):
        actual_data_path = data_path_primary
    elif os.path.exists(data_path_fallback):
        actual_data_path = data_path_fallback
        print(f"ℹ️ Fichier de données trouvé à {actual_data_path} (fallback)")
    else:
        print(f"Erreur : Fichier de données {data_path_primary} (ou {data_path_fallback}) non trouvé.")
        sys.exit(1)
    
    if not os.path.exists(args.models_dir):
        print(f"Erreur : Répertoire de modèles {args.models_dir} non trouvé.")
        sys.exit(1)
    
    # Charger les modèles
    print("Chargement des modèles...")
    main_model, stars_model = load_models(args.models_dir)
    
    # Préparer les données
    print("Préparation des données...")
    X_main_last, X_stars_last, main_scaler, stars_scaler = prepare_data(actual_data_path)
    
    # Prédire les prochains numéros
    print("Génération des prédictions...")
    main_numbers, star_numbers = predict_next_numbers(
        main_model, stars_model, X_main_last, X_stars_last, main_scaler, stars_scaler
    )
    
    # Afficher les résultats # Commented out
    # print("\nPrédiction pour le prochain tirage de l'Euromillions :")
    # print(f"Numéros principaux : {', '.join(map(str, main_numbers))}")
    # print(f"Étoiles : {', '.join(map(str, star_numbers))}")

    # Sauvegarder la prédiction dans un fichier # Commented out
    # with open(args.output, "w") as f:
    #     f.write(f"Prédiction pour le prochain tirage de l'Euromillions (générée le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}):\n")
    #     f.write(f"Numéros principaux : {', '.join(map(str, main_numbers))}\n")
    #     f.write(f"Étoiles : {', '.join(map(str, star_numbers))}\n")
    # print(f"\nLa prédiction a été sauvegardée dans le fichier '{args.output}'.")

    target_date_str = None
    if args.date:
        try:
            datetime.datetime.strptime(args.date, '%Y-%m-%d') # Validate
            target_date_str = args.date
        except ValueError:
            # print(f"Warning: Invalid date format for --date {args.date}. Using next logical date.", file=sys.stderr) # Suppressed
            target_date_obj = get_next_euromillions_draw_date(actual_data_path)
            target_date_str = target_date_obj.strftime('%Y-%m-%d')
    else:
        target_date_obj = get_next_euromillions_draw_date(actual_data_path)
        target_date_str = target_date_obj.strftime('%Y-%m-%d')

    output_dict = {
        "nom_predicteur": "predict_euromillions",
        "numeros": main_numbers.tolist(), # Ensure it's a list
        "etoiles": star_numbers.tolist(), # Ensure it's a list
        "date_tirage_cible": target_date_str,
        "confidence": 6.0, # Default confidence for TF model
        "categorie": "Scientifique"
    }
    print(json.dumps(output_dict))

if __name__ == "__main__":
    main()

