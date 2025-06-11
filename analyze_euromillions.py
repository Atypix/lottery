import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def load_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path: Chemin vers le fichier CSV
    
    Returns:
        DataFrame pandas contenant les données
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def analyze_number_frequency(df):
    """
    Analyse la fréquence d'apparition de chaque numéro.
    
    Args:
        df: DataFrame contenant les données
    
    Returns:
        Tuple (fréquences des numéros principaux, fréquences des étoiles)
    """
    # Analyser les numéros principaux (1-50)
    main_numbers = []
    for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
        main_numbers.extend(df[col].tolist())
    
    main_freq = Counter(main_numbers)
    main_freq = {k: v for k, v in sorted(main_freq.items())}
    
    # Analyser les étoiles (1-12)
    stars = []
    for col in ['E1', 'E2']:
        stars.extend(df[col].tolist())
    
    star_freq = Counter(stars)
    star_freq = {k: v for k, v in sorted(star_freq.items())}
    
    return main_freq, star_freq

def plot_number_frequency(main_freq, star_freq, output_dir):
    """
    Crée des graphiques pour visualiser la fréquence des numéros.
    
    Args:
        main_freq: Dictionnaire des fréquences des numéros principaux
        star_freq: Dictionnaire des fréquences des étoiles
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Graphique pour les numéros principaux
    plt.figure(figsize=(16, 8))
    plt.bar(main_freq.keys(), main_freq.values(), color='royalblue')
    plt.title('Fréquence d\'apparition des numéros principaux (1-50)', fontsize=16)
    plt.xlabel('Numéro', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.xticks(range(1, 51))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_numbers_frequency.png'))
    
    # Graphique pour les étoiles
    plt.figure(figsize=(12, 6))
    plt.bar(star_freq.keys(), star_freq.values(), color='gold')
    plt.title('Fréquence d\'apparition des étoiles (1-12)', fontsize=16)
    plt.xlabel('Étoile', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.xticks(range(1, 13))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stars_frequency.png'))

def analyze_number_trends(df, output_dir):
    """
    Analyse les tendances des numéros au fil du temps.
    
    Args:
        df: DataFrame contenant les données
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Créer un DataFrame avec une ligne par année et les fréquences moyennes
    years = df['Date'].dt.year.unique()
    yearly_data = []
    
    for year in years:
        year_df = df[df['Date'].dt.year == year]
        
        # Calculer la moyenne des numéros pour cette année
        avg_n1 = year_df['N1'].mean()
        avg_n2 = year_df['N2'].mean()
        avg_n3 = year_df['N3'].mean()
        avg_n4 = year_df['N4'].mean()
        avg_n5 = year_df['N5'].mean()
        avg_e1 = year_df['E1'].mean()
        avg_e2 = year_df['E2'].mean()
        
        yearly_data.append({
            'Year': year,
            'Avg_N1': avg_n1,
            'Avg_N2': avg_n2,
            'Avg_N3': avg_n3,
            'Avg_N4': avg_n4,
            'Avg_N5': avg_n5,
            'Avg_E1': avg_e1,
            'Avg_E2': avg_e2
        })
    
    yearly_df = pd.DataFrame(yearly_data)
    
    # Graphique des tendances pour les numéros principaux
    plt.figure(figsize=(16, 8))
    for col in ['Avg_N1', 'Avg_N2', 'Avg_N3', 'Avg_N4', 'Avg_N5']:
        plt.plot(yearly_df['Year'], yearly_df[col], marker='o', label=col.replace('Avg_', ''))
    
    plt.title('Évolution de la moyenne des numéros principaux par année', fontsize=16)
    plt.xlabel('Année', fontsize=14)
    plt.ylabel('Moyenne', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_numbers_trends.png'))
    
    # Graphique des tendances pour les étoiles
    plt.figure(figsize=(16, 8))
    for col in ['Avg_E1', 'Avg_E2']:
        plt.plot(yearly_df['Year'], yearly_df[col], marker='o', label=col.replace('Avg_', ''))
    
    plt.title('Évolution de la moyenne des étoiles par année', fontsize=16)
    plt.xlabel('Année', fontsize=14)
    plt.ylabel('Moyenne', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stars_trends.png'))

def analyze_number_combinations(df, output_dir):
    """
    Analyse les combinaisons de numéros les plus fréquentes.
    
    Args:
        df: DataFrame contenant les données
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Analyser les paires de numéros principaux les plus fréquentes
    pairs = []
    for i, row in df.iterrows():
        numbers = [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']]
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                pairs.append(tuple(sorted([numbers[i], numbers[j]])))
    
    pair_freq = Counter(pairs)
    most_common_pairs = pair_freq.most_common(20)
    
    # Graphique des paires les plus fréquentes
    plt.figure(figsize=(16, 8))
    pair_labels = [f"{p[0][0]}-{p[0][1]}" for p in most_common_pairs]
    pair_values = [p[1] for p in most_common_pairs]
    
    plt.bar(pair_labels, pair_values, color='royalblue')
    plt.title('Les 20 paires de numéros principaux les plus fréquentes', fontsize=16)
    plt.xlabel('Paire de numéros', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_number_pairs.png'))
    
    # Analyser les paires d'étoiles
    star_pairs = []
    for i, row in df.iterrows():
        star_pairs.append(tuple(sorted([row['E1'], row['E2']])))
    
    star_pair_freq = Counter(star_pairs)
    most_common_star_pairs = star_pair_freq.most_common(20)
    
    # Graphique des paires d'étoiles les plus fréquentes
    plt.figure(figsize=(16, 8))
    star_pair_labels = [f"{p[0][0]}-{p[0][1]}" for p in most_common_star_pairs]
    star_pair_values = [p[1] for p in most_common_star_pairs]
    
    plt.bar(star_pair_labels, star_pair_values, color='gold')
    plt.title('Les paires d\'étoiles les plus fréquentes', fontsize=16)
    plt.xlabel('Paire d\'étoiles', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'star_pairs.png'))

def analyze_sum_distribution(df, output_dir):
    """
    Analyse la distribution de la somme des numéros.
    
    Args:
        df: DataFrame contenant les données
        output_dir: Répertoire de sortie pour les graphiques
    """
    # Calculer la somme des numéros principaux pour chaque tirage
    df['Sum_Main'] = df['N1'] + df['N2'] + df['N3'] + df['N4'] + df['N5']
    df['Sum_Stars'] = df['E1'] + df['E2']
    
    # Graphique de la distribution de la somme des numéros principaux
    plt.figure(figsize=(16, 8))
    sns.histplot(df['Sum_Main'], bins=30, kde=True, color='royalblue')
    plt.title('Distribution de la somme des numéros principaux', fontsize=16)
    plt.xlabel('Somme des numéros principaux', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_sum_distribution.png'))
    
    # Graphique de la distribution de la somme des étoiles
    plt.figure(figsize=(16, 8))
    sns.histplot(df['Sum_Stars'], bins=15, kde=True, color='gold')
    plt.title('Distribution de la somme des étoiles', fontsize=16)
    plt.xlabel('Somme des étoiles', fontsize=14)
    plt.ylabel('Fréquence', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stars_sum_distribution.png'))

def main():
    # Créer un répertoire pour les graphiques
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    df = load_data('euromillions_dataset.csv')
    
    # Analyser la fréquence des numéros
    main_freq, star_freq = analyze_number_frequency(df)
    plot_number_frequency(main_freq, star_freq, output_dir)
    
    # Analyser les tendances au fil du temps
    analyze_number_trends(df, output_dir)
    
    # Analyser les combinaisons de numéros
    analyze_number_combinations(df, output_dir)
    
    # Analyser la distribution de la somme des numéros
    analyze_sum_distribution(df, output_dir)
    
    print(f"Analyse terminée. Les résultats ont été sauvegardés dans le répertoire '{output_dir}'.")
    
    # Afficher les numéros les plus et moins fréquents
    print("\nNuméros principaux les plus fréquents:")
    most_common_main = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    for num, freq in most_common_main:
        print(f"Numéro {num}: {freq} occurrences")
    
    print("\nNuméros principaux les moins fréquents:")
    least_common_main = sorted(main_freq.items(), key=lambda x: x[1])[:5]
    for num, freq in least_common_main:
        print(f"Numéro {num}: {freq} occurrences")
    
    print("\nÉtoiles les plus fréquentes:")
    most_common_stars = sorted(star_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    for num, freq in most_common_stars:
        print(f"Étoile {num}: {freq} occurrences")
    
    print("\nÉtoiles les moins fréquentes:")
    least_common_stars = sorted(star_freq.items(), key=lambda x: x[1])[:3]
    for num, freq in least_common_stars:
        print(f"Étoile {num}: {freq} occurrences")

if __name__ == "__main__":
    main()

