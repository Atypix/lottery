import pandas as pd
import numpy as np
import datetime
import random
import csv
import os

# Fonction pour générer des dates de tirage
def generate_draw_dates(start_date, end_date, frequency=2):
    """
    Génère des dates de tirage pour l'Euromillions.
    
    Args:
        start_date: Date de début (format 'YYYY-MM-DD')
        end_date: Date de fin (format 'YYYY-MM-DD')
        frequency: Nombre de tirages par semaine (par défaut 2 pour mardi et vendredi)
    
    Returns:
        Liste de dates de tirage
    """
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Jours de tirage: 1 = lundi, 2 = mardi, ..., 7 = dimanche
    # Pour Euromillions: 2 = mardi et 5 = vendredi
    draw_days = [2, 5] if frequency == 2 else [5]  # Avant 2011, seulement le vendredi
    
    dates = []
    current = start
    while current <= end:
        if current.weekday() + 1 in draw_days:
            dates.append(current.strftime('%Y-%m-%d'))
        current += datetime.timedelta(days=1)
    
    return dates

# Fonction pour générer un tirage aléatoire
def generate_random_draw():
    """
    Génère un tirage aléatoire pour l'Euromillions.
    
    Returns:
        Tuple (5 numéros principaux, 2 étoiles)
    """
    # 5 numéros principaux entre 1 et 50
    numbers = sorted(random.sample(range(1, 51), 5))
    
    # 2 étoiles entre 1 et 12
    stars = sorted(random.sample(range(1, 13), 2))
    
    return numbers, stars

# Fonction principale pour créer le dataset
def create_euromillions_dataset(output_file, start_date='2004-02-13', end_date='2025-06-07'):
    """
    Crée un dataset synthétique des tirages de l'Euromillions.
    
    Args:
        output_file: Chemin du fichier CSV de sortie
        start_date: Date de début (format 'YYYY-MM-DD')
        end_date: Date de fin (format 'YYYY-MM-DD')
    """
    # Générer les dates de tirage
    # Avant 2011, il n'y avait qu'un tirage par semaine (vendredi)
    dates_before_2011 = generate_draw_dates('2004-02-13', '2011-05-09', frequency=1)
    # Après 2011, il y a deux tirages par semaine (mardi et vendredi)
    dates_after_2011 = generate_draw_dates('2011-05-10', end_date, frequency=2)
    
    all_dates = dates_before_2011 + dates_after_2011
    
    # Créer le dataset
    data = []
    for date in all_dates:
        numbers, stars = generate_random_draw()
        row = [date] + numbers + stars
        data.append(row)
    
    # Créer un DataFrame pandas
    columns = ['Date', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
    df = pd.DataFrame(data, columns=columns)
    
    # Sauvegarder en CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset créé avec succès: {output_file}")
    print(f"Nombre total de tirages: {len(df)}")

if __name__ == "__main__":
    output_file = "euromillions_dataset.csv"
    create_euromillions_dataset(output_file)

