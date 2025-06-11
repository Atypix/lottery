import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import re

def fetch_euromillions_data():
    """
    Récupère les données réelles de l'Euromillions depuis l'API GitHub pedro-mealha.
    
    Returns:
        DataFrame contenant les données historiques
    """
    base_url = "https://euromillions.api.pedromealha.dev"
    
    try:
        # Récupérer tous les tirages
        print("Récupération des données depuis l'API...")
        response = requests.get(f"{base_url}/draws", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Récupéré {len(data)} tirages depuis l'API")
            
            # Convertir en DataFrame
            df = pd.DataFrame(data)
            
            # Nettoyer et formater les données
            df = clean_and_format_data(df) # This sorts by Date ascending
            
            latest_draw_info = None
            # Check source == "API" implicitly by being in this block
            if df is not None and not df.empty: # Check df is not None before checking if empty
                latest_draw_series = df.iloc[-1] # Last row is the latest draw
                latest_draw_info = {
                    'Date': latest_draw_series['Date'].strftime('%Y-%m-%d'),
                    'N1': int(latest_draw_series['N1']), # Ensure standard int type
                    'N2': int(latest_draw_series['N2']),
                    'N3': int(latest_draw_series['N3']),
                    'N4': int(latest_draw_series['N4']),
                    'N5': int(latest_draw_series['N5']),
                    'E1': int(latest_draw_series['E1']),
                    'E2': int(latest_draw_series['E2'])
                }
            return df, "API", latest_draw_info
        else:
            print(f"Erreur API: {response.status_code}")
            # Utiliser le jeu de données existant comme fallback
            return use_existing_dataset()
            
    except Exception as e:
        print(f"Erreur lors de la récupération des données: {e}")
        # Utiliser le jeu de données existant comme fallback
        return use_existing_dataset()

def use_existing_dataset():
    """
    Utilise le jeu de données existant en cas d'échec de l'API.
    
    Returns:
        DataFrame contenant les données historiques
    """
    try:
        print("Utilisation du jeu de données existant...")
        df = pd.read_csv("euromillions_dataset.csv")
        print(f"Jeu de données existant chargé: {len(df)} tirages")
        return df, "CSV", None # No latest_draw_info for CSV
    except FileNotFoundError:
        print("Aucun jeu de données disponible. Création d'un jeu de données synthétique...")
        return create_synthetic_dataset()

def create_synthetic_dataset():
    """
    Crée un jeu de données synthétique en cas d'absence de données.
    
    Returns:
        DataFrame contenant des données synthétiques
    """
    print("Création d'un jeu de données synthétique...")
    
    # Paramètres
    start_date = datetime(2004, 2, 13)
    end_date = datetime(2025, 6, 6)
    
    # Générer les dates (mardi et vendredi)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() == 1 or current_date.weekday() == 4:  # Mardi ou vendredi
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Générer les numéros aléatoires
    np.random.seed(42)  # Pour la reproductibilité
    n_draws = len(dates)
    
    # Numéros principaux (1-50)
    main_numbers = []
    for _ in range(n_draws):
        numbers = sorted(np.random.choice(range(1, 51), 5, replace=False))
        main_numbers.append(numbers)
    
    # Étoiles (1-12)
    stars = []
    for _ in range(n_draws):
        star_nums = sorted(np.random.choice(range(1, 13), 2, replace=False))
        stars.append(star_nums)
    
    # Créer le DataFrame
    data = {
        'Date': dates,
        'N1': [nums[0] for nums in main_numbers],
        'N2': [nums[1] for nums in main_numbers],
        'N3': [nums[2] for nums in main_numbers],
        'N4': [nums[3] for nums in main_numbers],
        'N5': [nums[4] for nums in main_numbers],
        'E1': [s[0] for s in stars],
        'E2': [s[1] for s in stars]
    }
    
    df = pd.DataFrame(data)
    print(f"Jeu de données synthétique créé: {len(df)} tirages")
    
    return df, "synthetic", None # No latest_draw_info for synthetic

def clean_and_format_data(df):
    """
    Nettoie et formate les données récupérées de l'API.
    
    Args:
        df: DataFrame brut de l'API
    
    Returns:
        DataFrame nettoyé et formaté
    """
    # Copier le DataFrame pour éviter les modifications inattendues
    df_clean = df.copy()
    
    # Vérifier les colonnes disponibles
    print(f"Colonnes disponibles: {df_clean.columns.tolist()}")
    
    # Convertir la date
    if 'date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['date'])
    elif 'draw_date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['draw_date'])
    
    # Créer un nouveau DataFrame pour les données formatées
    formatted_data = {
        'Date': df_clean['Date']
    }
    
    # Extraire les numéros principaux et les étoiles
    if 'numbers' in df_clean.columns and isinstance(df_clean['numbers'].iloc[0], list):
        # Si les numéros sont dans une liste
        for i in range(5):
            formatted_data[f'N{i+1}'] = df_clean['numbers'].apply(lambda x: x[i] if i < len(x) else 0)
    elif 'numbers' in df_clean.columns and isinstance(df_clean['numbers'].iloc[0], str):
        # Si les numéros sont dans une chaîne de caractères
        for i in range(5):
            formatted_data[f'N{i+1}'] = df_clean['numbers'].apply(
                lambda x: int(x[i*2:(i+1)*2]) if len(x) >= (i+1)*2 else 0
            )
    
    if 'stars' in df_clean.columns and isinstance(df_clean['stars'].iloc[0], list):
        # Si les étoiles sont dans une liste
        for i in range(2):
            formatted_data[f'E{i+1}'] = df_clean['stars'].apply(lambda x: x[i] if i < len(x) else 0)
    elif 'stars' in df_clean.columns and isinstance(df_clean['stars'].iloc[0], str):
        # Si les étoiles sont dans une chaîne de caractères
        for i in range(2):
            formatted_data[f'E{i+1}'] = df_clean['stars'].apply(
                lambda x: int(x[i*2:(i+1)*2]) if len(x) >= (i+1)*2 else 0
            )
    
    # Si les colonnes sont déjà séparées (n1, n2, etc.)
    if 'n1' in df_clean.columns:
        formatted_data['N1'] = df_clean['n1']
        formatted_data['N2'] = df_clean['n2']
        formatted_data['N3'] = df_clean['n3']
        formatted_data['N4'] = df_clean['n4']
        formatted_data['N5'] = df_clean['n5']
    
    if 's1' in df_clean.columns:
        formatted_data['E1'] = df_clean['s1']
        formatted_data['E2'] = df_clean['s2']
    
    # Créer le DataFrame final
    df_final = pd.DataFrame(formatted_data)
    
    # Vérifier si nous avons toutes les colonnes nécessaires
    required_columns = ['Date', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
    missing_columns = set(required_columns) - set(df_final.columns)
    
    if missing_columns:
        print(f"Colonnes manquantes: {missing_columns}")
        print("Tentative d'extraction à partir des données brutes...")
        
        # Essayer d'extraire les numéros à partir des données brutes
        if 'numbers' in df_clean.columns:
            numbers_str = df_clean['numbers'].iloc[0]
            print(f"Format des numéros: {numbers_str} (type: {type(numbers_str)})")
            
            # Si c'est une chaîne, essayer de l'analyser
            if isinstance(numbers_str, str):
                # Essayer différents formats
                if len(numbers_str) == 10:  # Format "1629323641"
                    for i in range(5):
                        formatted_data[f'N{i+1}'] = df_clean['numbers'].apply(
                            lambda x: int(x[i*2:(i+1)*2]) if len(x) >= (i+1)*2 else 0
                        )
                else:
                    # Essayer d'extraire les numéros avec une expression régulière
                    pattern = r'\d+'
                    for i, col in enumerate(['N1', 'N2', 'N3', 'N4', 'N5']):
                        if col not in formatted_data:
                            formatted_data[col] = df_clean['numbers'].apply(
                                lambda x: int(re.findall(pattern, x)[i]) if isinstance(x, str) and len(re.findall(pattern, x)) > i else 0
                            )
        
        if 'stars' in df_clean.columns:
            stars_str = df_clean['stars'].iloc[0]
            print(f"Format des étoiles: {stars_str} (type: {type(stars_str)})")
            
            # Si c'est une chaîne, essayer de l'analyser
            if isinstance(stars_str, str):
                # Essayer différents formats
                if len(stars_str) == 4:  # Format "0709"
                    for i in range(2):
                        formatted_data[f'E{i+1}'] = df_clean['stars'].apply(
                            lambda x: int(x[i*2:(i+1)*2]) if len(x) >= (i+1)*2 else 0
                        )
                else:
                    # Essayer d'extraire les étoiles avec une expression régulière
                    pattern = r'\d+'
                    for i, col in enumerate(['E1', 'E2']):
                        if col not in formatted_data:
                            formatted_data[col] = df_clean['stars'].apply(
                                lambda x: int(re.findall(pattern, x)[i]) if isinstance(x, str) and len(re.findall(pattern, x)) > i else 0
                            )
    
    # Recréer le DataFrame final
    df_final = pd.DataFrame(formatted_data)
    
    # Vérifier à nouveau les colonnes manquantes
    missing_columns = set(required_columns) - set(df_final.columns)
    if missing_columns:
        print(f"Impossible d'extraire toutes les colonnes nécessaires. Utilisation du jeu de données existant.")
        # Ensure this path also returns a tuple, though it might be complex to ensure source propogates here
        # For now, assuming clean_and_format_data is called by fetch_euromillions_data which handles source
        # This path might indicate a failure within clean_and_format_data itself.
        # A robust solution would have clean_and_format_data also signal failure type.
        # For this step, we'll assume if it gets here, the original source was likely API or CSV but cleaning failed.
        # However, use_existing_dataset() is called, which returns (df, source)
        return use_existing_dataset()
    
    # Trier par date
    df_final = df_final.sort_values('Date').reset_index(drop=True)
    
    # Supprimer les lignes avec des valeurs manquantes
    df_final = df_final.dropna()
    
    # Convertir les colonnes numériques en entiers
    for col in ['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']:
        df_final[col] = df_final[col].astype(int)
    
    print(f"Données nettoyées: {len(df_final)} tirages valides")
    
    return df_final

def add_advanced_features(df):
    """
    Ajoute des caractéristiques avancées pour améliorer les prédictions.
    
    Args:
        df: DataFrame avec les données de base
    
    Returns:
        DataFrame avec les caractéristiques supplémentaires
    """
    df_enhanced = df.copy()
    
    # Caractéristiques temporelles
    df_enhanced['Year'] = df_enhanced['Date'].dt.year
    df_enhanced['Month'] = df_enhanced['Date'].dt.month
    df_enhanced['DayOfWeek'] = df_enhanced['Date'].dt.dayofweek
    df_enhanced['DayOfYear'] = df_enhanced['Date'].dt.dayofyear
    df_enhanced['WeekOfYear'] = df_enhanced['Date'].dt.isocalendar().week
    df_enhanced['Quarter'] = df_enhanced['Date'].dt.quarter
    
    # Saison (0=Hiver, 1=Printemps, 2=Été, 3=Automne)
    df_enhanced['Season'] = ((df_enhanced['Month'] % 12) // 3)
    
    # Caractéristiques des numéros principaux
    main_cols = ['N1', 'N2', 'N3', 'N4', 'N5']
    df_enhanced['Main_Sum'] = df_enhanced[main_cols].sum(axis=1)
    df_enhanced['Main_Mean'] = df_enhanced[main_cols].mean(axis=1)
    df_enhanced['Main_Std'] = df_enhanced[main_cols].std(axis=1)
    df_enhanced['Main_Min'] = df_enhanced[main_cols].min(axis=1)
    df_enhanced['Main_Max'] = df_enhanced[main_cols].max(axis=1)
    df_enhanced['Main_Range'] = df_enhanced['Main_Max'] - df_enhanced['Main_Min']
    
    # Caractéristiques des étoiles
    star_cols = ['E1', 'E2']
    df_enhanced['Stars_Sum'] = df_enhanced[star_cols].sum(axis=1)
    df_enhanced['Stars_Mean'] = df_enhanced[star_cols].mean(axis=1)
    df_enhanced['Stars_Diff'] = abs(df_enhanced['E1'] - df_enhanced['E2'])
    
    # Parité des numéros
    df_enhanced['Main_Even_Count'] = df_enhanced[main_cols].apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
    df_enhanced['Main_Odd_Count'] = 5 - df_enhanced['Main_Even_Count']
    df_enhanced['Stars_Even_Count'] = df_enhanced[star_cols].apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
    
    # Répartition par dizaines
    def count_by_decade(row, cols):
        counts = [0] * 5  # 1-10, 11-20, 21-30, 31-40, 41-50
        for col in cols:
            val = row[col]
            if 1 <= val <= 10:
                counts[0] += 1
            elif 11 <= val <= 20:
                counts[1] += 1
            elif 21 <= val <= 30:
                counts[2] += 1
            elif 31 <= val <= 40:
                counts[3] += 1
            elif 41 <= val <= 50:
                counts[4] += 1
        return counts
    
    decade_counts = df_enhanced.apply(lambda row: count_by_decade(row, main_cols), axis=1)
    for i in range(5):
        df_enhanced[f'Decade_{i+1}_Count'] = [counts[i] for counts in decade_counts]
    
    # Caractéristiques de fréquence historique (fenêtre glissante)
    window_size = min(50, len(df_enhanced) // 2)  # Ajuster la taille de la fenêtre en fonction des données
    
    # Fréquence des numéros principaux dans les derniers tirages
    for i, col in enumerate(main_cols, 1):
        freq_col = f'N{i}_Freq_{window_size}'
        df_enhanced[freq_col] = 0.0
        
        for idx in range(window_size, len(df_enhanced)):
            recent_values = df_enhanced[col].iloc[idx-window_size:idx]
            current_value = df_enhanced[col].iloc[idx]
            frequency = (recent_values == current_value).sum() / window_size
            df_enhanced.loc[idx, freq_col] = frequency
    
    # Fréquence des étoiles dans les derniers tirages
    for i, col in enumerate(star_cols, 1):
        freq_col = f'E{i}_Freq_{window_size}'
        df_enhanced[freq_col] = 0.0
        
        for idx in range(window_size, len(df_enhanced)):
            recent_values = df_enhanced[col].iloc[idx-window_size:idx]
            current_value = df_enhanced[col].iloc[idx]
            frequency = (recent_values == current_value).sum() / window_size
            df_enhanced.loc[idx, freq_col] = frequency
    
    # Intervalles depuis la dernière apparition
    for col in main_cols + star_cols:
        interval_col = f'{col}_LastSeen'
        df_enhanced[interval_col] = 0
        
        for idx in range(1, len(df_enhanced)):
            current_value = df_enhanced[col].iloc[idx]
            # Chercher la dernière occurrence de cette valeur
            last_occurrence = -1
            for j in range(idx-1, -1, -1):
                if df_enhanced[col].iloc[j] == current_value:
                    last_occurrence = j
                    break
            
            if last_occurrence >= 0:
                df_enhanced.loc[idx, interval_col] = idx - last_occurrence
            else:
                df_enhanced.loc[idx, interval_col] = idx  # Depuis le début
    
    # Moyennes mobiles
    for window in [5, 10, 20]:
        if len(df_enhanced) > window:
            df_enhanced[f'Main_Sum_MA_{window}'] = df_enhanced['Main_Sum'].rolling(window=window).mean()
            df_enhanced[f'Stars_Sum_MA_{window}'] = df_enhanced['Stars_Sum'].rolling(window=window).mean()
    
    # Volatilité (écart-type mobile)
    for window in [5, 10, 20]:
        if len(df_enhanced) > window:
            df_enhanced[f'Main_Sum_Volatility_{window}'] = df_enhanced['Main_Sum'].rolling(window=window).std()
    
    # Remplacer les NaN par 0 pour les premières lignes
    df_enhanced = df_enhanced.fillna(0)
    
    print(f"Caractéristiques ajoutées. Nouvelles dimensions: {df_enhanced.shape}")
    
    return df_enhanced

def save_enhanced_dataset(df, filename="euromillions_enhanced_dataset.csv"):
    """
    Sauvegarde le jeu de données amélioré.
    
    Args:
        df: DataFrame à sauvegarder
        filename: Nom du fichier de sortie
    """
    df.to_csv(filename, index=False)
    print(f"Jeu de données amélioré sauvegardé: {filename}")
    print(f"Dimensions: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")

def update_euromillions_data():
    """
    Fonction principale pour récupérer, améliorer les données et retourner un statut.
    """
    print("=== Récupération et amélioration des données Euromillions ===")
    status_message = "Failed to update data." # Default status
    latest_draw_data = None # Initialize latest_draw_data
    
    # Récupérer les données réelles
    df, source, latest_draw_data = fetch_euromillions_data() # Now returns (df, source_type, latest_draw_info)

    if df is None or df.empty:
        # Construct a more informative message based on the source if possible
        if source == "API":
             status_message = "API call attempt failed or returned no data, or data cleaning failed."
        elif source == "CSV":
             status_message = "CSV load attempt resulted in empty data or failed."
        elif source == "synthetic":
             status_message = "Synthetic data generation resulted in empty data or failed."
        else: # Generic fallback
            status_message = "Data source returned empty or None, and no synthetic data generated."

        print(status_message)
        # Return the new dict structure even on failure
        return {'status_message': status_message, 'latest_draw_data': latest_draw_data}

    # Afficher un aperçu des données
    print("\nAperçu des données:")
    print(df.head())
    
    # Ensure 'Date' column is in datetime format for min/max operations
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"Error converting Date column to datetime: {e}")
            # Fallback or error handling if 'Date' conversion fails
            # For now, we might not be able to provide date range in status.
            date_min_str, date_max_str = "N/A", "N/A"

    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']) and not df.empty:
        date_min_str = df['Date'].min().strftime('%Y-%m-%d')
        date_max_str = df['Date'].max().strftime('%Y-%m-%d')
        print(f"\nPériode couverte: {date_min_str} à {date_max_str}")
    else:
        date_min_str, date_max_str = "N/A", "N/A"
        print("\nPériode couverte: N/A (Date column missing or empty)")

    # Ajouter des caractéristiques avancées
    print("\nAjout de caractéristiques avancées...")
    df_enhanced = add_advanced_features(df)
    
    # Sauvegarder le jeu de données amélioré à l'emplacement standardisé
    save_enhanced_dataset(df_enhanced, "euromillions_enhanced_dataset.csv")

    # Construire le message de statut
    if source == "API":
        status_message = f"Data updated successfully from API. Total draws: {len(df_enhanced)}. Period: {date_min_str} to {date_max_str}."
    elif source == "CSV":
        status_message = f"Data updated using existing dataset 'euromillions_dataset.csv'. Total draws: {len(df_enhanced)}. Period: {date_min_str} to {date_max_str}."
    elif source == "synthetic":
        status_message = f"Data updated using synthetic dataset. Total draws: {len(df_enhanced)}. Period: {date_min_str} to {date_max_str}."
    else:
        status_message = f"Data processed from unknown source. Total draws: {len(df_enhanced)}. Period: {date_min_str} to {date_max_str}."

    # Afficher les statistiques finales
    print("\n=== Statistiques du jeu de données amélioré ===")
    print(f"Nombre total de tirages: {len(df_enhanced)}")
    print(f"Nombre de caractéristiques: {len(df_enhanced.columns)}")
    print(f"Période: {date_min_str} à {date_max_str}")
    
    print(f"\nSomme moyenne des numéros principaux: {df_enhanced['Main_Sum'].mean():.2f}")
    print(f"Somme médiane des numéros principaux: {df_enhanced['Main_Sum'].median():.2f}")
    
    main_numbers = df_enhanced[['N1', 'N2', 'N3', 'N4', 'N5']].values.flatten()
    main_counts = pd.Series(main_numbers).value_counts()
    most_common_main = main_counts.index[0] if not main_counts.empty else "N/A"
    
    stars = df_enhanced[['E1', 'E2']].values.flatten()
    star_counts = pd.Series(stars).value_counts()
    most_common_star = star_counts.index[0] if not star_counts.empty else "N/A"
    
    print(f"Numéro principal le plus fréquent: {most_common_main}")
    print(f"Étoile la plus fréquente: {most_common_star}")

    return {'status_message': status_message, 'latest_draw_data': latest_draw_data}

if __name__ == "__main__":
    result = update_euromillions_data() # result is now a dictionary
    print(f"\nScript execution status: {result['status_message']}")
    if result['latest_draw_data']:
        print("\n--- Latest Fetched Draw (from API before enhancement) ---")
        print(f"Date: {result['latest_draw_data']['Date']}")
        # Construct numbers and stars list for printing
        numbers = [result['latest_draw_data'][f'N{i}'] for i in range(1,6)]
        stars = [result['latest_draw_data'][f'E{i}'] for i in range(1,3)]
        print(f"Numbers: {numbers}")
        print(f"Stars: {stars}")

