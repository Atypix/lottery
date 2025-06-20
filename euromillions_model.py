import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import datetime
from common.date_utils import get_next_euromillions_draw_date # Already added, good.

# Configuration des paramètres
SEQUENCE_LENGTH = 10  # Nombre de tirages précédents à considérer
BATCH_SIZE = 32
EPOCHS = 10  # Réduit de 100 à 10 pour accélérer l'entraînement
LEARNING_RATE = 0.001
# MODEL_NAME global variable is removed, will be handled in train_all_models_and_predict

# Fonction pour charger et préparer les données
def load_and_prepare_data(file_path): # Signature remains same, use_enhanced_specific_cols not needed
    """
    Charge et prépare les données pour l'entraînement du modèle.
    
    Args:
        file_path: Chemin vers le fichier CSV contenant les données
    
    Returns:
        Tuple (X_train, y_train, X_test, y_test, scalers)
    """
    # Charger les données
    df = pd.read_csv(file_path)
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
    
    # Créer des séquences pour l'apprentissage
    X_main, y_main = create_sequences(main_normalized, SEQUENCE_LENGTH)
    X_stars, y_stars = create_sequences(stars_normalized, SEQUENCE_LENGTH)
    
    # Diviser en ensembles d'entraînement et de test
    X_main_train, X_main_test, y_main_train, y_main_test = train_test_split(
        X_main, y_main, test_size=0.2, random_state=42
    )
    
    X_stars_train, X_stars_test, y_stars_train, y_stars_test = train_test_split(
        X_stars, y_stars, test_size=0.2, random_state=42
    )
    
    return (
        X_main_train, y_main_train, X_main_test, y_main_test,
        X_stars_train, y_stars_train, X_stars_test, y_stars_test,
        main_scaler, stars_scaler
    )

# Fonction pour créer des séquences
def create_sequences(data, seq_length):
    """
    Crée des séquences pour l'apprentissage.
    
    Args:
        data: Données normalisées
        seq_length: Longueur de la séquence
    
    Returns:
        Tuple (X, y) où X contient les séquences et y les valeurs cibles
    """
    X = []
    y = []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)

# Fonction pour créer le modèle des numéros principaux
def create_main_numbers_model(input_shape):
    """
    Crée un modèle pour prédire les numéros principaux.
    
    Args:
        input_shape: Forme des données d'entrée
    
    Returns:
        Modèle Keras
    """
    model = Sequential()
    
    # Couche LSTM avec normalisation par lots
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Deuxième couche LSTM
    model.add(LSTM(128, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Couches denses
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Couche de sortie (5 numéros principaux)
    model.add(Dense(5, activation='sigmoid'))
    
    # Compiler le modèle
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )
    
    return model

# Fonction pour créer le modèle des étoiles
def create_stars_model(input_shape):
    """
    Crée un modèle pour prédire les étoiles.
    
    Args:
        input_shape: Forme des données d'entrée
    
    Returns:
        Modèle Keras
    """
    model = Sequential()
    
    # Couche LSTM avec normalisation par lots
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Deuxième couche LSTM
    model.add(LSTM(64, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Couches denses
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Couche de sortie (2 étoiles)
    model.add(Dense(2, activation='sigmoid'))
    
    # Compiler le modèle
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )
    
    return model

# Fonction pour entraîner le modèle
def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Entraîne le modèle et sauvegarde les meilleurs poids.
    
    Args:
        model: Modèle Keras
        X_train: Données d'entraînement
        y_train: Cibles d'entraînement
        X_test: Données de test
        y_test: Cibles de test
        model_name: Nom du modèle pour la sauvegarde
    
    Returns:
        Historique d'entraînement
    """
    # Créer un répertoire pour les logs et les checkpoints
    log_dir = os.path.join("logs", model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join("models", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks
    tensorboard = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir, "model_{epoch:02d}_{val_loss:.4f}.h5"),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard, checkpoint]
    )
    
    # Sauvegarder le modèle final
    # model_name here is base name like "tf_main_std" or "tf_stars_enhanced"
    # Checkpoint dir is "models/tf_main_std", so final model is "models/tf_main_std/final_model.h5"
    model.save(os.path.join(checkpoint_dir, "final_model.h5"))
    
    return history

# Fonction pour visualiser l'historique d'entraînement
def plot_training_history(history, model_name):
    """
    Visualise l'historique d'entraînement.
    
    Args:
        history: Historique d'entraînement
        model_name: Nom du modèle
    """
    plt.figure(figsize=(12, 5))
    
    # Graphique de la perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte (entraînement)')
    plt.plot(history.history['val_loss'], label='Perte (validation)')
    plt.title('Évolution de la perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    # Graphique de l'erreur absolue moyenne
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE (entraînement)')
    plt.plot(history.history['val_mae'], label='MAE (validation)')
    plt.title('Évolution de l\'erreur absolue moyenne')
    plt.xlabel('Époque')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    
    # Créer un répertoire pour les graphiques
    plots_dir = os.path.join("plots", model_name)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Sauvegarder le graphique
    plt.savefig(os.path.join(plots_dir, "training_history.png"))
    plt.close()

# Fonction pour prédire les prochains numéros
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

# Fonction principale
def main():
    # Créer les répertoires nécessaires
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Charger et préparer les données
    # This main() function is effectively commented out by the new if __name__ == "__main__": block
    # However, if it were to be used, its data path would need adjustment.
    # For now, focusing on the active functions train_all_models_and_predict and predict_with_tensorflow_model
    (
        X_main_train, y_main_train, X_main_test, y_main_test,
        X_stars_train, y_stars_train, X_stars_test, y_stars_test,
        main_scaler, stars_scaler
    ) = load_and_prepare_data("data/euromillions_dataset.csv") # Added data/ prefix
    
    # Créer les modèles
    main_model = create_main_numbers_model((X_main_train.shape[1], X_main_train.shape[2]))
    stars_model = create_stars_model((X_stars_train.shape[1], X_stars_train.shape[2]))
    
    # Afficher les résumés des modèles
    print("Modèle pour les numéros principaux:")
    main_model.summary()
    
    print("\nModèle pour les étoiles:")
    stars_model.summary()
    
    # Entraîner les modèles
    print("\nEntraînement du modèle pour les numéros principaux...")
    main_history = train_model(
        main_model, X_main_train, y_main_train, X_main_test, y_main_test,
        f"{MODEL_NAME}_main"
    )
    
    print("\nEntraînement du modèle pour les étoiles...")
    stars_history = train_model(
        stars_model, X_stars_train, y_stars_train, X_stars_test, y_stars_test,
        f"{MODEL_NAME}_stars"
    )
    
    # Visualiser l'historique d'entraînement
    plot_training_history(main_history, f"{MODEL_NAME}_main")
    plot_training_history(stars_history, f"{MODEL_NAME}_stars")
    
    # Prédire les prochains numéros
    # Utiliser les dernières séquences disponibles
    X_main_last = X_main_test[-1]
    X_stars_last = X_stars_test[-1]
    
    main_numbers, star_numbers = predict_next_numbers(
        main_model, stars_model, X_main_last, X_stars_last, main_scaler, stars_scaler
    )
    
    print("\nPrédiction pour le prochain tirage:")
    print(f"Numéros principaux: {main_numbers}")
    print(f"Étoiles: {star_numbers}")
    
    # Sauvegarder la prédiction dans un fichier
    with open("prediction.txt", "w") as f:
        f.write("Prédiction pour le prochain tirage de l'Euromillions:\n")
        f.write(f"Numéros principaux: {', '.join(map(str, main_numbers))}\n")
        f.write(f"Étoiles: {', '.join(map(str, star_numbers))}\n")
    
    print("\nLa prédiction a été sauvegardée dans le fichier 'prediction.txt'.")

from common.date_utils import get_next_euromillions_draw_date # Added for tf_lstm
# --- Start of refactored functions ---

# Made load_trained_models a top-level function and added use_enhanced_data flag
def load_trained_models(use_enhanced_data=False):
    model_dir_suffix = "_enhanced" if use_enhanced_data else "_std"
    main_model_base_name = f"tf_main{model_dir_suffix}"
    stars_model_base_name = f"tf_stars{model_dir_suffix}"

    main_model_path = os.path.join("models", main_model_base_name, "final_model.h5")
    stars_model_path = os.path.join("models", stars_model_base_name, "final_model.h5")

    main_model = None
    stars_model = None
    try:
        if os.path.exists(main_model_path):
            main_model = tf.keras.models.load_model(main_model_path)
            print(f"Loaded main numbers model from {main_model_path}")
        else:
            print(f"Warning: Main numbers model not found at {main_model_path}")
        if os.path.exists(stars_model_path):
            stars_model = tf.keras.models.load_model(stars_model_path)
            print(f"Loaded stars model from {stars_model_path}")
        else:
            print(f"Warning: Stars model not found at {stars_model_path}")
    except Exception as e:
        print(f"Error loading models: {e}")
    return main_model, stars_model

def predict_with_tensorflow_model(use_enhanced_data=False): # Added use_enhanced_data
    main_model, stars_model = load_trained_models(use_enhanced_data=use_enhanced_data) # Pass flag

    current_model_name = f"tf_lstm{'_enhanced' if use_enhanced_data else '_std'}"

    if main_model is None or stars_model is None:
        return {
            'numbers': [], 'stars': [], 'confidence': None,
            'model_name': current_model_name, 'status': 'failure',
            'message': 'Models not found. Please train them or ensure dummy models exist.'
        }

    dataset_path_primary = f"data/euromillions{'_enhanced' if use_enhanced_data else ''}_dataset.csv"
    dataset_path_fallback = f"euromillions{'_enhanced' if use_enhanced_data else ''}_dataset.csv"

    actual_dataset_path = None
    if os.path.exists(dataset_path_primary):
        actual_dataset_path = dataset_path_primary
    elif os.path.exists(dataset_path_fallback):
        actual_dataset_path = dataset_path_fallback
        print(f"ℹ️ Using dataset from fallback: {actual_dataset_path}")

    if not actual_dataset_path:
        return {
            'numbers': [], 'stars': [], 'confidence': None,
            'model_name': current_model_name, 'status': 'failure',
            'message': f'{dataset_path_primary} (or {dataset_path_fallback}) not found.'
        }
    try:
        df = pd.read_csv(actual_dataset_path)
    except FileNotFoundError: # Should be caught by the check above, but as a safeguard
        return {
            'numbers': [], 'stars': [], 'confidence': None,
            'model_name': current_model_name, 'status': 'failure',
            'message': f'{actual_dataset_path} not found (secondary check).'
        }

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    main_numbers_data = df[['N1', 'N2', 'N3', 'N4', 'N5']].values
    stars_data = df[['E1', 'E2']].values

    main_scaler = MinMaxScaler(feature_range=(0, 1))
    stars_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scalers with all data, then transform
    main_normalized_full = main_scaler.fit_transform(main_numbers_data)
    stars_normalized_full = stars_scaler.fit_transform(stars_data)

    if len(main_normalized_full) < SEQUENCE_LENGTH or len(stars_normalized_full) < SEQUENCE_LENGTH:
        return {
            'numbers': [], 'stars': [], 'confidence': None,
            'model_name': current_model_name, 'status': 'failure',
            'message': f'Not enough data to form a sequence (need {SEQUENCE_LENGTH}, got {len(main_normalized_full)} from {actual_dataset_path}).'
        }

    X_main_last_sequence_data = main_normalized_full[-SEQUENCE_LENGTH:]
    X_stars_last_sequence_data = stars_normalized_full[-SEQUENCE_LENGTH:]

    X_main_last_sequence = np.reshape(X_main_last_sequence_data, (1, SEQUENCE_LENGTH, X_main_last_sequence_data.shape[1]))
    X_stars_last_sequence = np.reshape(X_stars_last_sequence_data, (1, SEQUENCE_LENGTH, X_stars_last_sequence_data.shape[1]))

    main_pred_norm = main_model.predict(X_main_last_sequence)
    stars_pred_norm = stars_model.predict(X_stars_last_sequence)

    main_pred = main_scaler.inverse_transform(main_pred_norm)
    stars_pred = stars_scaler.inverse_transform(stars_pred_norm)

    final_main_numbers = np.round(main_pred[0]).astype(int)
    final_stars = np.round(stars_pred[0]).astype(int)

    final_main_numbers = np.clip(final_main_numbers, 1, 50)
    final_stars = np.clip(final_stars, 1, 12)

    unique_main = sorted(list(set(final_main_numbers)))
    while len(unique_main) < 5:
        new_num = np.random.randint(1, 51)
        if new_num not in unique_main: unique_main.append(new_num)
        unique_main = sorted(list(set(unique_main))) # ensure sorted after append
    final_main_numbers = unique_main[:5]

    unique_stars = sorted(list(set(final_stars)))
    while len(unique_stars) < 2:
        new_star = np.random.randint(1, 13)
        if new_star not in unique_stars: unique_stars.append(new_star)
        unique_stars = sorted(list(set(unique_stars))) # ensure sorted after append
    final_stars = unique_stars[:2]


    next_draw_date_obj = get_next_euromillions_draw_date(actual_dataset_path) # Use chosen actual_dataset_path
    target_date_str = next_draw_date_obj.strftime('%Y-%m-%d')

    return {
        'numbers': final_main_numbers,
        'stars': final_stars,
        'confidence': None,
        'model_name': current_model_name,
        'status': 'success',
        'message': 'Prediction generated successfully.',
        'target_draw_date': target_date_str
    }

# Renaming original main function and adding use_enhanced_data flag
def train_all_models_and_predict(use_enhanced_data=False):
    os.makedirs("logs", exist_ok=True)
    # "models" directory itself is fine, subdirs are named specifically
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    dataset_path_primary = f"data/euromillions{'_enhanced' if use_enhanced_data else ''}_dataset.csv"
    dataset_path_fallback = f"euromillions{'_enhanced' if use_enhanced_data else ''}_dataset.csv"
    actual_dataset_path_train = None
    if os.path.exists(dataset_path_primary):
        actual_dataset_path_train = dataset_path_primary
    elif os.path.exists(dataset_path_fallback):
        actual_dataset_path_train = dataset_path_fallback
        print(f"ℹ️ Using dataset from fallback for training: {actual_dataset_path_train}")

    if not actual_dataset_path_train:
        print(f"❌ Training aborted: Dataset not found ({dataset_path_primary} or {dataset_path_fallback}).")
        return

    print(f"--- Training models using: {actual_dataset_path_train} ---")

    (
        X_main_train, y_main_train, X_main_test, y_main_test,
        X_stars_train, y_stars_train, X_stars_test, y_stars_test,
        main_scaler, stars_scaler
    ) = load_and_prepare_data(actual_dataset_path_train) # Pass actual_dataset_path_train

    main_model = create_main_numbers_model((X_main_train.shape[1], X_main_train.shape[2]))
    stars_model = create_stars_model((X_stars_train.shape[1], X_stars_train.shape[2]))

    print("Modèle pour les numéros principaux:")
    main_model.summary()
    print("\nModèle pour les étoiles:")
    stars_model.summary()

    model_dir_suffix = "_enhanced" if use_enhanced_data else "_std"
    # Define base names for model directories and logs, removing timestamp for consistent loading
    main_model_base_name = f"tf_main{model_dir_suffix}"
    stars_model_base_name = f"tf_stars{model_dir_suffix}"

    print(f"\nEntraînement du modèle pour les numéros principaux ({main_model_base_name})...")
    main_history = train_model(
        main_model, X_main_train, y_main_train, X_main_test, y_main_test,
        main_model_base_name
    )

    print(f"\nEntraînement du modèle pour les étoiles ({stars_model_base_name})...")
    stars_history = train_model(
        stars_model, X_stars_train, y_stars_train, X_stars_test, y_stars_test,
        stars_model_base_name
    )

    plot_training_history(main_history, main_model_base_name)
    plot_training_history(stars_history, stars_model_base_name)

    # Prédire les prochains numéros
    # Utiliser les dernières séquences disponibles
    X_main_last = X_main_test[-1]
    X_stars_last = X_stars_test[-1]

    # Note: The predict_next_numbers function uses the *trained* models from this session.
    # The new predict_with_tensorflow_model uses models loaded from "models/euromillions_model_tf_main/final_model.h5"
    main_numbers, star_numbers = predict_next_numbers(
        main_model, stars_model, X_main_last, X_stars_last, main_scaler, stars_scaler
    )

    print("\nPrédiction pour le prochain tirage (from training run):")
    print(f"Numéros principaux: {main_numbers}")
    print(f"Étoiles: {star_numbers}")

    # Sauvegarder la prédiction dans un fichier
    with open("prediction.txt", "w") as f:
        f.write("Prédiction pour le prochain tirage de l'Euromillions (from training run):\n")
        f.write(f"Numéros principaux: {', '.join(map(str, main_numbers))}\n")
        f.write(f"Étoiles: {', '.join(map(str, star_numbers))}\n")

    print("\nLa prédiction (from training run) a été sauvegardée dans le fichier 'prediction.txt'.")


if __name__ == "__main__":
    # To run prediction (assuming models are trained or dummy models exist):
    print("\n--- Attempting Prediction with Standard Models ---")
    prediction_result_std = predict_with_tensorflow_model(use_enhanced_data=False)
    if prediction_result_std['status'] == 'success':
        print(f"Numbers (std): {prediction_result_std['numbers']}")
        print(f"Stars (std): {prediction_result_std['stars']}")
        print(f"Target Date (std): {prediction_result_std['target_draw_date']}")
    else:
        print(f"Prediction failed (std): {prediction_result_std['message']}")
    print(f"Model: {prediction_result_std['model_name']}, Status: {prediction_result_std['status']}")

    print("\n--- Attempting Prediction with Enhanced Models ---")
    prediction_result_enhanced = predict_with_tensorflow_model(use_enhanced_data=True)
    if prediction_result_enhanced['status'] == 'success':
        print(f"Numbers (enhanced): {prediction_result_enhanced['numbers']}")
        print(f"Stars (enhanced): {prediction_result_enhanced['stars']}")
        print(f"Target Date (enhanced): {prediction_result_enhanced['target_draw_date']}")
    else:
        print(f"Prediction failed (enhanced): {prediction_result_enhanced['message']}")
    print(f"Model: {prediction_result_enhanced['model_name']}, Status: {prediction_result_enhanced['status']}")

    # Example calls for training (optional, can be commented out):
    # print("\n--- Training Standard TensorFlow Models (optional) ---")
    # train_all_models_and_predict(use_enhanced_data=False)
    # print("\n--- Training Enhanced TensorFlow Models (optional) ---")
    # train_all_models_and_predict(use_enhanced_data=True)
    # print("Training complete (if run).")

