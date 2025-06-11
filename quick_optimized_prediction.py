import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import datetime

def load_enhanced_data():
    """
    Charge les données améliorées.
    """
    df = pd.read_csv("euromillions_enhanced_dataset.csv")
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
    print("=== Prédiction Euromillions Optimisée Rapide ===")
    
    # Charger les données
    df = load_enhanced_data()
    
    # Préparer les données
    main_normalized, stars_normalized, features, main_scaler, stars_scaler = prepare_data_for_prediction(df)
    
    # Entraîner les modèles
    models = train_optimized_models(main_normalized, stars_normalized, features)
    
    # Faire la prédiction
    print("\nGénération de la prédiction optimisée...")
    main_numbers, star_numbers = make_ensemble_prediction(
        models, main_normalized, stars_normalized, features, main_scaler, stars_scaler
    )
    
    # Analyser la confiance
    confidence = analyze_prediction_confidence(df, main_numbers, star_numbers)
    
    print(f"\n🎯 PRÉDICTION OPTIMISÉE POUR LE PROCHAIN TIRAGE 🎯")
    print("=" * 60)
    print(f"Numéros principaux: {', '.join(map(str, main_numbers))}")
    print(f"Étoiles: {', '.join(map(str, star_numbers))}")
    print("=" * 60)
    
    print(f"\n📊 ANALYSE DE CONFIANCE 📊")
    print(f"Somme des numéros principaux: {confidence['predicted_sum']}")
    print(f"Percentile de la somme: {confidence['sum_percentile']:.1f}%")
    print(f"Fréquence moyenne des numéros principaux: {confidence['avg_main_freq']:.3f}")
    print(f"Fréquence moyenne des étoiles: {confidence['avg_star_freq']:.3f}")
    
    print(f"\n📈 DÉTAILS DES FRÉQUENCES 📈")
    for i, (num, freq) in enumerate(zip(main_numbers, confidence['main_freq'])):
        print(f"Numéro {num}: {freq:.3f} ({freq*100:.1f}%)")
    
    for i, (num, freq) in enumerate(zip(star_numbers, confidence['star_freq'])):
        print(f"Étoile {num}: {freq:.3f} ({freq*100:.1f}%)")
    
    # Sauvegarder la prédiction
    with open("prediction_quick_optimized.txt", "w") as f:
        f.write(f"Prédiction Euromillions Optimisée Rapide\n")
        f.write(f"Générée le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Numéros principaux: {', '.join(map(str, main_numbers))}\n")
        f.write(f"Étoiles: {', '.join(map(str, star_numbers))}\n\n")
        f.write("Modèles utilisés:\n")
        f.write("- Random Forest optimisé (60% de poids)\n")
        f.write("- XGBoost optimisé (40% de poids)\n")
        f.write("- Ensemble learning avec données réelles\n\n")
        f.write(f"Données: {len(df)} tirages réels avec caractéristiques avancées\n")
        f.write(f"Période: {df['Date'].min()} à {df['Date'].max()}\n\n")
        f.write("Analyse de confiance:\n")
        f.write(f"- Somme des numéros principaux: {confidence['predicted_sum']}\n")
        f.write(f"- Percentile de la somme: {confidence['sum_percentile']:.1f}%\n")
        f.write(f"- Fréquence moyenne des numéros principaux: {confidence['avg_main_freq']:.3f}\n")
        f.write(f"- Fréquence moyenne des étoiles: {confidence['avg_star_freq']:.3f}\n")
    
    print(f"\nPrédiction sauvegardée dans 'prediction_quick_optimized.txt'")
    print("\n⚠️  AVERTISSEMENT ⚠️")
    print("Cette prédiction est basée sur l'analyse de données historiques.")
    print("L'Euromillions reste un jeu de hasard et aucune prédiction ne peut garantir des gains.")
    print("Jouez de manière responsable!")

if __name__ == "__main__":
    main()

