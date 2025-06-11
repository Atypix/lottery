#!/usr/bin/env python3
"""
Créateur de données Euromillions françaises récentes
Utilise les données UK comme base et les adapte au format français
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_french_euromillions_data():
    """Crée un dataset Euromillions français récent"""
    
    print("🇫🇷 CRÉATION DU DATASET EUROMILLIONS FRANÇAIS")
    print("=" * 50)
    
    # Lecture des données UK comme référence
    uk_data = pd.read_csv('/home/ubuntu/downloads/csv.csv')
    print(f"✅ Données UK chargées : {len(uk_data)} tirages")
    
    # Conversion au format français
    french_data = []
    
    for _, row in uk_data.iterrows():
        # Conversion de la date
        date_str = row['DrawDate']
        date_obj = datetime.strptime(date_str, '%d-%b-%Y')
        french_date = date_obj.strftime('%d/%m/%Y')
        
        # Extraction des numéros et étoiles
        numbers = [row['Ball 1'], row['Ball 2'], row['Ball 3'], row['Ball 4'], row['Ball 5']]
        stars = [row['Lucky Star 1'], row['Lucky Star 2']]
        
        # Tri des numéros et étoiles
        numbers.sort()
        stars.sort()
        
        french_data.append({
            'Date': french_date,
            'Numero_1': numbers[0],
            'Numero_2': numbers[1], 
            'Numero_3': numbers[2],
            'Numero_4': numbers[3],
            'Numero_5': numbers[4],
            'Etoile_1': stars[0],
            'Etoile_2': stars[1],
            'Tirage': row['DrawNumber']
        })
    
    # Création du DataFrame français
    df_french = pd.DataFrame(french_data)
    
    # Tri par date (plus récent en premier)
    df_french['Date_obj'] = pd.to_datetime(df_french['Date'], format='%d/%m/%Y')
    df_french = df_french.sort_values('Date_obj', ascending=False)
    df_french = df_french.drop('Date_obj', axis=1)
    
    # Sauvegarde
    output_path = '/home/ubuntu/euromillions_france_recent.csv'
    df_french.to_csv(output_path, index=False)
    
    print(f"✅ Dataset français créé : {output_path}")
    print(f"📊 Nombre de tirages : {len(df_french)}")
    print(f"📅 Période : {df_french['Date'].iloc[-1]} à {df_french['Date'].iloc[0]}")
    
    # Vérification du tirage de référence
    reference_draw = df_french[df_french['Date'] == '06/06/2025']
    if not reference_draw.empty:
        print("\n🎯 TIRAGE DE RÉFÉRENCE TROUVÉ :")
        row = reference_draw.iloc[0]
        print(f"Date : {row['Date']}")
        print(f"Numéros : {row['Numero_1']}, {row['Numero_2']}, {row['Numero_3']}, {row['Numero_4']}, {row['Numero_5']}")
        print(f"Étoiles : {row['Etoile_1']}, {row['Etoile_2']}")
    
    # Statistiques rapides
    print("\n📈 STATISTIQUES RAPIDES :")
    all_numbers = []
    all_stars = []
    
    for _, row in df_french.iterrows():
        all_numbers.extend([row['Numero_1'], row['Numero_2'], row['Numero_3'], row['Numero_4'], row['Numero_5']])
        all_stars.extend([row['Etoile_1'], row['Etoile_2']])
    
    # Top 5 numéros les plus fréquents
    from collections import Counter
    number_counts = Counter(all_numbers)
    star_counts = Counter(all_stars)
    
    print("🔢 Top 5 numéros les plus fréquents :")
    for num, count in number_counts.most_common(5):
        print(f"   {num}: {count} fois ({count/len(df_french)*100:.1f}%)")
    
    print("⭐ Top 5 étoiles les plus fréquentes :")
    for star, count in star_counts.most_common(5):
        print(f"   {star}: {count} fois ({count/len(df_french)*100:.1f}%)")
    
    return output_path

if __name__ == "__main__":
    dataset_path = create_french_euromillions_data()
    print(f"\n🎉 DATASET FRANÇAIS PRÊT : {dataset_path}")

