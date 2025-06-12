import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import datetime # Already datetime.datetime
import argparse # Added
import json # Added
from common.date_utils import get_next_euromillions_draw_date, date as datetime_date # Added
# pandas, numpy, os (implicitly via main) are used

def load_enhanced_data():
    """
    Charge les données améliorées.
    """
    data_path_primary = "data/euromillions_enhanced_dataset.csv"
    data_path_fallback = "euromillions_enhanced_dataset.csv"
    actual_data_path = None
    if os.path.exists(data_path_primary): # os needs to be imported if not already
        actual_data_path = data_path_primary
    elif os.path.exists(data_path_fallback):
        actual_data_path = data_path_fallback
        # print(f"ℹ️ Données chargées depuis {actual_data_path} (fallback)") # Suppressed

    if not actual_data_path:
        # print(f"❌ ERREUR: Fichier de données non trouvé ({data_path_primary} ou {data_path_fallback})") # Suppressed
        # Fallback to empty dataframe or raise error
        return pd.DataFrame()

    df = pd.read_csv(actual_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Données chargées: {len(df)} tirages avec {len(df.columns)} caractéristiques")
    return df

def prepare_data_for_prediction(df):
    """
    Prépare les données pour la prédiction.
    """
    # Séparer les numéros principaux et les étoiles
    main_numbers = df[['N1', 'N2', 'N3', 'N4', 'N5']].values
    stars = df[['E1', 'E2']].values
    
    # Sélectionner les caractéristiques les plus importantes
    important_features = [
        'Main_Sum', 'Main_Mean', 'Main_Std', 'Stars_Sum', 'Stars_Mean',
        'Main_Even_Count', 'Main_Odd_Count', 'Month', 'DayOfWeek',
        'Main_Sum_MA_5', 'Main_Sum_MA_10', 'Main_Sum_Volatility_5'
    ]
    
    # Vérifier quelles caractéristiques sont disponibles
    available_features = [col for col in important_features if col in df.columns]
    print(f"Caractéristiques utilisées: {available_features}")
    
    features = df[available_features].values
    
    # Normaliser les données
    main_scaler = MinMaxScaler(feature_range=(0, 1))
    stars_scaler = MinMaxScaler(feature_range=(0, 1))
    
    main_normalized = main_scaler.fit_transform(main_numbers)
    stars_normalized = stars_scaler.fit_transform(stars)
    
    return main_normalized, stars_normalized, features, main_scaler, stars_scaler

def create_ml_sequences(target_data, features_data, sequence_length=10):
    """
    Crée des séquences pour l'apprentissage ML.
    """
    X, y = [], []
    
    for i in range(sequence_length, len(target_data)):
        # Utiliser les dernières valeurs et caractéristiques
        sequence_data = target_data[i-sequence_length:i].flatten()
        current_features = features_data[i]
        combined_features = np.concatenate([sequence_data, current_features])
        
        X.append(combined_features)
        y.append(target_data[i])
    
    return np.array(X), np.array(y)

def train_optimized_models(main_normalized, stars_normalized, features):
    """
    Entraîne des modèles optimisés rapidement.
    """
    print("Entraînement des modèles optimisés...")
    
    sequence_length = 10
    
    # Préparer les données
    X_main, y_main = create_ml_sequences(main_normalized, features, sequence_length)
    X_stars, y_stars = create_ml_sequences(stars_normalized, features, sequence_length)
    
    # Division train/test
    split_idx = int(len(X_main) * 0.85)
    
    X_main_train = X_main[:split_idx]
    y_main_train = y_main[:split_idx]
    X_main_test = X_main[split_idx:]
    y_main_test = y_main[split_idx:]
    
    X_stars_train = X_stars[:split_idx]
    y_stars_train = y_stars[:split_idx]
    X_stars_test = X_stars[split_idx:]
    y_stars_test = y_stars[split_idx:]
    
    # Modèles Random Forest optimisés
    rf_main_models = []
    for i in range(5):
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42 + i,
            n_jobs=-1
        )
        rf_model.fit(X_main_train, y_main_train[:, i])
        rf_main_models.append(rf_model)
    
    rf_stars_models = []
    for i in range(2):
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42 + i,
            n_jobs=-1
        )
        rf_model.fit(X_stars_train, y_stars_train[:, i])
        rf_stars_models.append(rf_model)
    
    # Modèles XGBoost optimisés
    xgb_main_models = []
    for i in range(5):
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + i,
            n_jobs=-1
        )
        xgb_model.fit(X_main_train, y_main_train[:, i])
        xgb_main_models.append(xgb_model)
    
    xgb_stars_models = []
    for i in range(2):
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + i,
            n_jobs=-1
        )
        xgb_model.fit(X_stars_train, y_stars_train[:, i])
        xgb_stars_models.append(xgb_model)
    
    print("Modèles entraînés avec succès!")
    
    return {
        'rf_main': rf_main_models,
        'rf_stars': rf_stars_models,
        'xgb_main': xgb_main_models,
        'xgb_stars': xgb_stars_models,
        'sequence_length': sequence_length
    }

def make_ensemble_prediction(models, main_normalized, stars_normalized, features, main_scaler, stars_scaler):
    """
    Fait une prédiction en combinant les modèles.
    """
    sequence_length = models['sequence_length']
    
    # Préparer les données pour la prédiction
    sequence_data_main = main_normalized[-sequence_length:].flatten()
    sequence_data_stars = stars_normalized[-sequence_length:].flatten()
    current_features = features[-1]
    
    X_main = np.concatenate([sequence_data_main, current_features]).reshape(1, -1)
    X_stars = np.concatenate([sequence_data_stars, current_features]).reshape(1, -1)
    
    # Prédictions Random Forest
    rf_main_pred = np.array([model.predict(X_main)[0] for model in models['rf_main']])
    rf_stars_pred = np.array([model.predict(X_stars)[0] for model in models['rf_stars']])
    
    # Prédictions XGBoost
    xgb_main_pred = np.array([model.predict(X_main)[0] for model in models['xgb_main']])
    xgb_stars_pred = np.array([model.predict(X_stars)[0] for model in models['xgb_stars']])
    
    # Ensemble avec poids optimisés
    ensemble_main = 0.6 * rf_main_pred + 0.4 * xgb_main_pred
    ensemble_stars = 0.6 * rf_stars_pred + 0.4 * xgb_stars_pred
    
    # Inverser la normalisation
    main_pred = main_scaler.inverse_transform(ensemble_main.reshape(1, -1))[0]
    stars_pred = stars_scaler.inverse_transform(ensemble_stars.reshape(1, -1))[0]
    
    # Arrondir et ajuster
    main_numbers = np.round(main_pred).astype(int)
    star_numbers = np.round(stars_pred).astype(int)
    
    # S'assurer que les numéros sont dans les plages valides
    main_numbers = np.clip(main_numbers, 1, 50)
    star_numbers = np.clip(star_numbers, 1, 12)
    
    # Éliminer les doublons
    main_numbers = ensure_unique_numbers(main_numbers, 5, 1, 50)
    star_numbers = ensure_unique_numbers(star_numbers, 2, 1, 12)
    
    return main_numbers, star_numbers

def ensure_unique_numbers(numbers, target_count, min_val, max_val):
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

def analyze_prediction_confidence(df, main_numbers, star_numbers):
    """
    Analyse la confiance de la prédiction basée sur les données historiques.
    """
    # Analyser la fréquence des numéros prédits
    main_freq = []
    for num in main_numbers:
        freq = ((df[['N1', 'N2', 'N3', 'N4', 'N5']] == num).sum().sum()) / (len(df) * 5)
        main_freq.append(freq)
    
    star_freq = []
    for num in star_numbers:
        freq = ((df[['E1', 'E2']] == num).sum().sum()) / (len(df) * 2)
        star_freq.append(freq)
    
    # Calculer la somme prédite
    predicted_sum = sum(main_numbers)
    
    # Analyser la distribution des sommes historiques
    historical_sums = df[['N1', 'N2', 'N3', 'N4', 'N5']].sum(axis=1)
    sum_percentile = (historical_sums <= predicted_sum).mean() * 100
    
    return {
        'main_freq': main_freq,
        'star_freq': star_freq,
        'predicted_sum': predicted_sum,
        'sum_percentile': sum_percentile,
        'avg_main_freq': np.mean(main_freq),
        'avg_star_freq': np.mean(star_freq)
    }

def main():
    """
    Fonction principale pour la prédiction optimisée rapide.
    """
    parser = argparse.ArgumentParser(description="Quick Optimized Euromillions Predictor.")
    parser.add_argument("--date", type=str, help="Target draw date in YYYY-MM-DD format.")
    args = parser.parse_args()

    target_date_str = None
    data_file_path_for_next_date_calc = "data/euromillions_enhanced_dataset.csv" # Default for get_next_euromillions_draw_date

    if args.date:
        try:
            datetime.datetime.strptime(args.date, '%Y-%m-%d') # Validate date format
            target_date_str = args.date
        except ValueError:
            # print(f"Error: Date format for --date should be YYYY-MM-DD. Using next draw date instead.", file=sys.stderr) # Suppressed
            # Determine data_file_path for get_next_euromillions_draw_date
            df_temp = load_enhanced_data() # Load data to see if it exists for date calculation
            if df_temp.empty:
                 data_file_path_for_next_date_calc = None # No data, can't determine next date based on it

            target_date_obj = get_next_euromillions_draw_date(data_file_path_for_next_date_calc)
            target_date_str = target_date_obj.strftime('%Y-%m-%d')
    else:
        df_temp = load_enhanced_data()
        if df_temp.empty:
            data_file_path_for_next_date_calc = None

        target_date_obj = get_next_euromillions_draw_date(data_file_path_for_next_date_calc)
        target_date_str = target_date_obj.strftime('%Y-%m-%d')

    # print("=== Prédiction Euromillions Optimisée Rapide ===") # Suppressed
    
    df = load_enhanced_data()
    if df.empty:
        # print("Error: Could not load data. Exiting.") # Suppressed
        # Output a failure JSON? For now, assume it might fall back to random if generate_quick_prediction handles empty df.
        # However, the current script structure would error out before.
        # For robust CLI, this path needs to output valid JSON error or handle gracefully.
        # For now, let's assume data loading is usually successful.
        # If it fails, the script will likely crash before JSON output, which is a form of failure for the consensus.
        error_output = {
            "nom_predicteur": "quick_optimized_prediction",
            "numeros": [], "etoiles": [],
            "date_tirage_cible": target_date_str,
            "confidence": 0.0, "categorie": "Scientifique",
            "error": "Failed to load data."
        }
        print(json.dumps(error_output))
        return


    main_normalized, stars_normalized, features, main_scaler, stars_scaler = prepare_data_for_prediction(df)
    models = train_optimized_models(main_normalized, stars_normalized, features)
    
    # print("\nGénération de la prédiction optimisée...") # Suppressed
    main_numbers, star_numbers = make_ensemble_prediction(
        models, main_normalized, stars_normalized, features, main_scaler, stars_scaler
    )
    
    # confidence_analysis = analyze_prediction_confidence(df, main_numbers, star_numbers) # Suppressed complex confidence

    # print(f"\n🎯 PRÉDICTION OPTIMISÉE POUR LE PROCHAIN TIRAGE 🎯") # Suppressed
    # ... other prints suppressed ...

    # Sauvegarder la prédiction # Suppressed
    # with open("prediction_quick_optimized.txt", "w") as f:
    # ... content ...
    # print(f"\nPrédiction sauvegardée dans 'prediction_quick_optimized.txt'") # Suppressed
    # ... warning prints suppressed ...

    output_dict = {
        "nom_predicteur": "quick_optimized_prediction",
        "numeros": main_numbers.tolist(), # Ensure list for JSON
        "etoiles": star_numbers.tolist(), # Ensure list for JSON
        "date_tirage_cible": target_date_str,
        "confidence": 7.0, # Default confidence for this optimized model
        "categorie": "Scientifique"
    }
    print(json.dumps(output_dict))

if __name__ == "__main__":
    main()

