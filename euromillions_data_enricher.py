import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Création du répertoire pour les données enrichies
os.makedirs("data_enriched", exist_ok=True)

class EuromillionsDataEnricher:
    """
    Classe pour récupérer et enrichir les données de l'Euromillions
    avec des caractéristiques avancées et des sources externes.
    """
    
    def __init__(self):
        self.api_base_url = "https://euromillions.api.pedromealha.dev/v1"
        self.draws_data = None
        self.df = None
        self.df_enriched = None
    
    def fetch_all_draws(self):
        """
        Récupère tous les tirages de l'Euromillions depuis l'API.
        """
        print("Récupération des données de l'API Euromillions...")
        response = requests.get(f"{self.api_base_url}/draws")
        
        if response.status_code == 200:
            self.draws_data = response.json()
            print(f"✅ {len(self.draws_data)} tirages récupérés avec succès.")
            return self.draws_data
        else:
            print(f"❌ Erreur lors de la récupération des données: {response.status_code}")
            return None
    
    def convert_to_dataframe(self):
        """
        Convertit les données JSON en DataFrame pandas.
        """
        if self.draws_data is None:
            print("❌ Aucune donnée à convertir. Veuillez d'abord récupérer les données.")
            return None
        
        print("Conversion des données en DataFrame...")
        
        # Initialisation des listes pour stocker les données
        data = []
        
        # Parcours des tirages
        for draw in self.draws_data:
            # Extraction des numéros principaux et des étoiles
            numbers = [int(n) for n in draw['numbers']]
            stars = [int(s) for s in draw['stars']]
            
            # Création d'une ligne de données
            row = {
                'date': datetime.strptime(draw['date'], '%Y-%m-%d'),
                'draw_id': draw['draw_id'],
                'has_winner': draw['has_winner'],
                'N1': numbers[0],
                'N2': numbers[1],
                'N3': numbers[2],
                'N4': numbers[3],
                'N5': numbers[4],
                'E1': stars[0],
                'E2': stars[1]
            }
            
            # Ajout des informations sur les prix
            total_winners = 0
            total_prize_pool = 0
            jackpot = 0
            
            for prize_info in draw['prizes']:
                matched_numbers = prize_info['matched_numbers']
                matched_stars = prize_info['matched_stars']
                winners = prize_info['winners']
                prize = prize_info['prize']
                
                # Calcul du total des gagnants et du prize pool
                total_winners += winners
                total_prize_pool += winners * prize
                
                # Identification du jackpot (5 numéros + 2 étoiles)
                if matched_numbers == 5 and matched_stars == 2:
                    jackpot = prize
                
                # Ajout des informations spécifiques sur les prix
                key = f"prize_{matched_numbers}_{matched_stars}"
                row[key] = prize
                
                key = f"winners_{matched_numbers}_{matched_stars}"
                row[key] = winners
            
            # Ajout des totaux
            row['total_winners'] = total_winners
            row['total_prize_pool'] = total_prize_pool
            row['jackpot'] = jackpot
            
            # Ajout de la ligne au tableau de données
            data.append(row)
        
        # Création du DataFrame
        self.df = pd.DataFrame(data)
        
        # Tri par date
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print(f"✅ DataFrame créé avec {len(self.df)} lignes et {len(self.df.columns)} colonnes.")
        return self.df
    
    def add_basic_features(self):
        """
        Ajoute des caractéristiques de base au DataFrame.
        """
        if self.df is None:
            print("❌ Aucun DataFrame à enrichir. Veuillez d'abord convertir les données.")
            return None
        
        print("Ajout des caractéristiques de base...")
        
        # Copie du DataFrame original
        self.df_enriched = self.df.copy()
        
        # Extraction des composantes de date
        self.df_enriched['Year'] = self.df_enriched['date'].dt.year
        self.df_enriched['Month'] = self.df_enriched['date'].dt.month
        self.df_enriched['Day'] = self.df_enriched['date'].dt.day
        self.df_enriched['DayOfWeek'] = self.df_enriched['date'].dt.dayofweek
        self.df_enriched['DayOfYear'] = self.df_enriched['date'].dt.dayofyear
        self.df_enriched['WeekOfYear'] = self.df_enriched['date'].dt.isocalendar().week
        self.df_enriched['Quarter'] = self.df_enriched['date'].dt.quarter
        
        # Ajout de la saison
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Hiver
            elif month in [3, 4, 5]:
                return 1  # Printemps
            elif month in [6, 7, 8]:
                return 2  # Été
            else:
                return 3  # Automne
        
        self.df_enriched['Season'] = self.df_enriched['Month'].apply(get_season)
        
        # Statistiques sur les numéros principaux
        self.df_enriched['Main_Sum'] = self.df_enriched[['N1', 'N2', 'N3', 'N4', 'N5']].sum(axis=1)
        self.df_enriched['Main_Mean'] = self.df_enriched[['N1', 'N2', 'N3', 'N4', 'N5']].mean(axis=1)
        self.df_enriched['Main_Std'] = self.df_enriched[['N1', 'N2', 'N3', 'N4', 'N5']].std(axis=1)
        self.df_enriched['Main_Min'] = self.df_enriched[['N1', 'N2', 'N3', 'N4', 'N5']].min(axis=1)
        self.df_enriched['Main_Max'] = self.df_enriched[['N1', 'N2', 'N3', 'N4', 'N5']].max(axis=1)
        self.df_enriched['Main_Range'] = self.df_enriched['Main_Max'] - self.df_enriched['Main_Min']
        
        # Statistiques sur les étoiles
        self.df_enriched['Stars_Sum'] = self.df_enriched[['E1', 'E2']].sum(axis=1)
        self.df_enriched['Stars_Mean'] = self.df_enriched[['E1', 'E2']].mean(axis=1)
        self.df_enriched['Stars_Diff'] = self.df_enriched['E2'] - self.df_enriched['E1']
        
        # Nombre de numéros pairs/impairs
        self.df_enriched['Main_Even_Count'] = self.df_enriched[['N1', 'N2', 'N3', 'N4', 'N5']].apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
        self.df_enriched['Main_Odd_Count'] = 5 - self.df_enriched['Main_Even_Count']
        self.df_enriched['Stars_Even_Count'] = self.df_enriched[['E1', 'E2']].apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
        
        # Répartition par dizaines
        def count_in_range(row, start, end):
            return sum(1 for n in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']] if start <= n <= end)
        
        self.df_enriched['Decade_1_Count'] = self.df_enriched.apply(lambda row: count_in_range(row, 1, 10), axis=1)
        self.df_enriched['Decade_2_Count'] = self.df_enriched.apply(lambda row: count_in_range(row, 11, 20), axis=1)
        self.df_enriched['Decade_3_Count'] = self.df_enriched.apply(lambda row: count_in_range(row, 21, 30), axis=1)
        self.df_enriched['Decade_4_Count'] = self.df_enriched.apply(lambda row: count_in_range(row, 31, 40), axis=1)
        self.df_enriched['Decade_5_Count'] = self.df_enriched.apply(lambda row: count_in_range(row, 41, 50), axis=1)
        
        print(f"✅ {len(self.df_enriched.columns) - len(self.df.columns)} caractéristiques de base ajoutées.")
        return self.df_enriched
    
    def add_frequency_features(self, window_size=50):
        """
        Ajoute des caractéristiques de fréquence au DataFrame.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi. Veuillez d'abord ajouter les caractéristiques de base.")
            return None
        
        print(f"Ajout des caractéristiques de fréquence (fenêtre de {window_size} tirages)...")
        
        # Pour chaque numéro principal (1-50)
        for num in range(1, 51):
            # Fréquence dans les derniers window_size tirages
            col_name = f'N{num}_Freq_{window_size}'
            self.df_enriched[col_name] = 0
            
            # Dernière apparition
            col_name_last = f'N{num}_LastSeen'
            self.df_enriched[col_name_last] = 0
            
            # Parcours des tirages
            for i in range(len(self.df_enriched)):
                # Fréquence
                if i < window_size:
                    # Pour les premiers tirages, on prend tous les tirages disponibles
                    previous_draws = self.df_enriched.iloc[:i+1]
                else:
                    # Sinon, on prend les window_size derniers tirages
                    previous_draws = self.df_enriched.iloc[i-window_size+1:i+1]
                
                # Calcul de la fréquence
                count = sum(1 for col in ['N1', 'N2', 'N3', 'N4', 'N5'] if (previous_draws[col] == num).any())
                self.df_enriched.loc[i, f'N{num}_Freq_{window_size}'] = count / len(previous_draws)
                
                # Dernière apparition
                if i > 0:
                    # On cherche le dernier tirage où le numéro est apparu
                    last_seen = 0
                    for j in range(i, 0, -1):
                        row = self.df_enriched.iloc[j]
                        if num in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']]:
                            break
                        last_seen += 1
                    
                    self.df_enriched.loc[i, f'N{num}_LastSeen'] = last_seen
        
        # Pour chaque étoile (1-12)
        for num in range(1, 13):
            # Fréquence dans les derniers window_size tirages
            col_name = f'E{num}_Freq_{window_size}'
            self.df_enriched[col_name] = 0
            
            # Dernière apparition
            col_name_last = f'E{num}_LastSeen'
            self.df_enriched[col_name_last] = 0
            
            # Parcours des tirages
            for i in range(len(self.df_enriched)):
                # Fréquence
                if i < window_size:
                    # Pour les premiers tirages, on prend tous les tirages disponibles
                    previous_draws = self.df_enriched.iloc[:i+1]
                else:
                    # Sinon, on prend les window_size derniers tirages
                    previous_draws = self.df_enriched.iloc[i-window_size+1:i+1]
                
                # Calcul de la fréquence
                count = sum(1 for col in ['E1', 'E2'] if (previous_draws[col] == num).any())
                self.df_enriched.loc[i, f'E{num}_Freq_{window_size}'] = count / len(previous_draws)
                
                # Dernière apparition
                if i > 0:
                    # On cherche le dernier tirage où l'étoile est apparue
                    last_seen = 0
                    for j in range(i, 0, -1):
                        row = self.df_enriched.iloc[j]
                        if num in [row['E1'], row['E2']]:
                            break
                        last_seen += 1
                    
                    self.df_enriched.loc[i, f'E{num}_LastSeen'] = last_seen
        
        print(f"✅ Caractéristiques de fréquence ajoutées pour {50} numéros principaux et {12} étoiles.")
        return self.df_enriched
    
    def add_trend_features(self):
        """
        Ajoute des caractéristiques de tendance au DataFrame.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi. Veuillez d'abord ajouter les caractéristiques de base.")
            return None
        
        print("Ajout des caractéristiques de tendance...")
        
        # Moyennes mobiles pour la somme des numéros principaux
        self.df_enriched['Main_Sum_MA_5'] = self.df_enriched['Main_Sum'].rolling(window=5, min_periods=1).mean()
        self.df_enriched['Main_Sum_MA_10'] = self.df_enriched['Main_Sum'].rolling(window=10, min_periods=1).mean()
        self.df_enriched['Main_Sum_MA_20'] = self.df_enriched['Main_Sum'].rolling(window=20, min_periods=1).mean()
        
        # Moyennes mobiles pour la somme des étoiles
        self.df_enriched['Stars_Sum_MA_5'] = self.df_enriched['Stars_Sum'].rolling(window=5, min_periods=1).mean()
        self.df_enriched['Stars_Sum_MA_10'] = self.df_enriched['Stars_Sum'].rolling(window=10, min_periods=1).mean()
        self.df_enriched['Stars_Sum_MA_20'] = self.df_enriched['Stars_Sum'].rolling(window=20, min_periods=1).mean()
        
        # Volatilité (écart-type mobile)
        self.df_enriched['Main_Sum_Volatility_5'] = self.df_enriched['Main_Sum'].rolling(window=5, min_periods=1).std()
        self.df_enriched['Main_Sum_Volatility_10'] = self.df_enriched['Main_Sum'].rolling(window=10, min_periods=1).std()
        self.df_enriched['Main_Sum_Volatility_20'] = self.df_enriched['Main_Sum'].rolling(window=20, min_periods=1).std()
        
        print(f"✅ Caractéristiques de tendance ajoutées.")
        return self.df_enriched
    
    def add_external_features(self):
        """
        Ajoute des caractéristiques externes au DataFrame.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi. Veuillez d'abord ajouter les caractéristiques de base.")
            return None
        
        print("Ajout des caractéristiques externes...")
        
        # Jours fériés (approximation simplifiée)
        holidays = [
            # Nouvel An
            '01-01',
            # Pâques (approximation)
            '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '04-07', '04-08', '04-09', '04-10',
            '04-11', '04-12', '04-13', '04-14', '04-15', '04-16', '04-17', '04-18', '04-19', '04-20',
            '04-21', '04-22', '04-23', '04-24', '04-25',
            # Fête du Travail
            '05-01',
            # Victoire 1945
            '05-08',
            # Ascension (approximation)
            '05-15', '05-16', '05-17', '05-18', '05-19', '05-20', '05-21', '05-22', '05-23', '05-24',
            '05-25', '05-26', '05-27', '05-28', '05-29', '05-30',
            # Pentecôte (approximation)
            '06-01', '06-02', '06-03', '06-04', '06-05', '06-06', '06-07', '06-08', '06-09', '06-10',
            # Fête Nationale
            '07-14',
            # Assomption
            '08-15',
            # Toussaint
            '11-01',
            # Armistice
            '11-11',
            # Noël
            '12-25'
        ]
        
        # Ajout de l'indicateur de jour férié
        self.df_enriched['IsHoliday'] = self.df_enriched['date'].dt.strftime('%m-%d').isin(holidays)
        
        # Indicateur de fin de mois (derniers 5 jours)
        self.df_enriched['IsMonthEnd'] = self.df_enriched['date'].dt.is_month_end
        self.df_enriched['DaysToMonthEnd'] = self.df_enriched['date'].dt.days_in_month - self.df_enriched['date'].dt.day
        self.df_enriched['IsMonthEnd5Days'] = self.df_enriched['DaysToMonthEnd'] <= 5
        
        # Indicateur de début de mois (premiers 5 jours)
        self.df_enriched['IsMonthStart'] = self.df_enriched['date'].dt.is_month_start
        self.df_enriched['IsMonthStart5Days'] = self.df_enriched['date'].dt.day <= 5
        
        # Indicateur de fin d'année (derniers 15 jours)
        self.df_enriched['IsYearEnd'] = self.df_enriched['date'].dt.is_year_end
        self.df_enriched['DaysToYearEnd'] = 365 - self.df_enriched['date'].dt.dayofyear
        self.df_enriched['IsYearEnd15Days'] = self.df_enriched['DaysToYearEnd'] <= 15
        
        # Indicateur de début d'année (premiers 15 jours)
        self.df_enriched['IsYearStart'] = self.df_enriched['date'].dt.is_year_start
        self.df_enriched['IsYearStart15Days'] = self.df_enriched['date'].dt.dayofyear <= 15
        
        print(f"✅ Caractéristiques externes ajoutées.")
        return self.df_enriched
    
    def add_jackpot_features(self):
        """
        Ajoute des caractéristiques liées au jackpot au DataFrame.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi. Veuillez d'abord ajouter les caractéristiques de base.")
            return None
        
        print("Ajout des caractéristiques liées au jackpot...")
        
        # Nombre de tirages consécutifs sans gagnant du jackpot
        self.df_enriched['ConsecutiveDrawsWithoutJackpotWinner'] = 0
        
        count = 0
        for i in range(len(self.df_enriched)):
            if not self.df_enriched.loc[i, 'has_winner']:
                count += 1
            else:
                count = 0
            
            self.df_enriched.loc[i, 'ConsecutiveDrawsWithoutJackpotWinner'] = count
        
        # Estimation du jackpot pour le prochain tirage
        self.df_enriched['EstimatedNextJackpot'] = 0
        
        for i in range(1, len(self.df_enriched)):
            if not self.df_enriched.loc[i-1, 'has_winner']:
                # Si pas de gagnant au tirage précédent, le jackpot augmente
                self.df_enriched.loc[i, 'EstimatedNextJackpot'] = self.df_enriched.loc[i-1, 'jackpot'] * 1.2
            else:
                # Si gagnant au tirage précédent, le jackpot est réinitialisé
                self.df_enriched.loc[i, 'EstimatedNextJackpot'] = 17000000  # Jackpot de base (17 millions d'euros)
        
        print(f"✅ Caractéristiques liées au jackpot ajoutées.")
        return self.df_enriched
    
    def add_advanced_statistical_features(self):
        """
        Ajoute des caractéristiques statistiques avancées au DataFrame.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi. Veuillez d'abord ajouter les caractéristiques de base.")
            return None
        
        print("Ajout des caractéristiques statistiques avancées...")
        
        # Écart entre les numéros consécutifs
        self.df_enriched['N1_N2_Gap'] = self.df_enriched['N2'] - self.df_enriched['N1']
        self.df_enriched['N2_N3_Gap'] = self.df_enriched['N3'] - self.df_enriched['N2']
        self.df_enriched['N3_N4_Gap'] = self.df_enriched['N4'] - self.df_enriched['N3']
        self.df_enriched['N4_N5_Gap'] = self.df_enriched['N5'] - self.df_enriched['N4']
        
        # Écart moyen entre les numéros
        self.df_enriched['Mean_Gap'] = (self.df_enriched['N1_N2_Gap'] + self.df_enriched['N2_N3_Gap'] + 
                                        self.df_enriched['N3_N4_Gap'] + self.df_enriched['N4_N5_Gap']) / 4
        
        # Écart-type des écarts
        self.df_enriched['Std_Gap'] = self.df_enriched[['N1_N2_Gap', 'N2_N3_Gap', 'N3_N4_Gap', 'N4_N5_Gap']].std(axis=1)
        
        # Somme des carrés des numéros
        self.df_enriched['Sum_Squares'] = (self.df_enriched['N1']**2 + self.df_enriched['N2']**2 + 
                                          self.df_enriched['N3']**2 + self.df_enriched['N4']**2 + 
                                          self.df_enriched['N5']**2)
        
        # Somme des cubes des numéros
        self.df_enriched['Sum_Cubes'] = (self.df_enriched['N1']**3 + self.df_enriched['N2']**3 + 
                                        self.df_enriched['N3']**3 + self.df_enriched['N4']**3 + 
                                        self.df_enriched['N5']**3)
        
        # Coefficient de variation
        self.df_enriched['Coef_Variation'] = self.df_enriched['Main_Std'] / self.df_enriched['Main_Mean']
        
        # Skewness (asymétrie)
        def skewness(row):
            nums = [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']]
            mean = sum(nums) / 5
            std = np.std(nums)
            if std == 0:
                return 0
            return sum(((n - mean) / std) ** 3 for n in nums) / 5
        
        self.df_enriched['Skewness'] = self.df_enriched.apply(skewness, axis=1)
        
        # Kurtosis (aplatissement)
        def kurtosis(row):
            nums = [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']]
            mean = sum(nums) / 5
            std = np.std(nums)
            if std == 0:
                return 0
            return sum(((n - mean) / std) ** 4 for n in nums) / 5 - 3
        
        self.df_enriched['Kurtosis'] = self.df_enriched.apply(kurtosis, axis=1)
        
        print(f"✅ Caractéristiques statistiques avancées ajoutées.")
        return self.df_enriched
    
    def add_pattern_features(self):
        """
        Ajoute des caractéristiques basées sur des patterns au DataFrame.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi. Veuillez d'abord ajouter les caractéristiques de base.")
            return None
        
        print("Ajout des caractéristiques basées sur des patterns...")
        
        # Nombre de numéros consécutifs
        def count_consecutive(row):
            nums = sorted([row['N1'], row['N2'], row['N3'], row['N4'], row['N5']])
            count = 0
            for i in range(1, len(nums)):
                if nums[i] == nums[i-1] + 1:
                    count += 1
            return count
        
        self.df_enriched['Consecutive_Count'] = self.df_enriched.apply(count_consecutive, axis=1)
        
        # Nombre de numéros premiers
        def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        def count_primes(row):
            return sum(1 for n in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']] if is_prime(n))
        
        self.df_enriched['Prime_Count'] = self.df_enriched.apply(count_primes, axis=1)
        
        # Nombre de numéros de Fibonacci
        fibonacci_numbers = [1, 2, 3, 5, 8, 13, 21, 34]
        
        def count_fibonacci(row):
            return sum(1 for n in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']] if n in fibonacci_numbers)
        
        self.df_enriched['Fibonacci_Count'] = self.df_enriched.apply(count_fibonacci, axis=1)
        
        # Nombre de numéros carrés parfaits
        perfect_squares = [1, 4, 9, 16, 25, 36, 49]
        
        def count_perfect_squares(row):
            return sum(1 for n in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']] if n in perfect_squares)
        
        self.df_enriched['Perfect_Square_Count'] = self.df_enriched.apply(count_perfect_squares, axis=1)
        
        # Patterns de parité (alternance pair/impair)
        def has_parity_pattern(row):
            nums = [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']]
            parities = [n % 2 for n in nums]
            
            # Vérification de l'alternance
            alternating = True
            for i in range(1, len(parities)):
                if parities[i] == parities[i-1]:
                    alternating = False
                    break
            
            return 1 if alternating else 0
        
        self.df_enriched['Has_Parity_Pattern'] = self.df_enriched.apply(has_parity_pattern, axis=1)
        
        print(f"✅ Caractéristiques basées sur des patterns ajoutées.")
        return self.df_enriched
    
    def save_data(self, filename='euromillions_enhanced_dataset.csv'):
        """
        Sauvegarde le DataFrame enrichi au format CSV.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi à sauvegarder.")
            return None
        
        filepath = os.path.join('data_enriched', filename)
        self.df_enriched.to_csv(filepath, index=False)
        print(f"✅ Données enrichies sauvegardées dans {filepath}")
        
        # Sauvegarde également au format JSON pour une utilisation plus facile
        json_filepath = os.path.join('data_enriched', filename.replace('.csv', '.json'))
        self.df_enriched.to_json(json_filepath, orient='records', date_format='iso')
        print(f"✅ Données enrichies sauvegardées au format JSON dans {json_filepath}")
        
        return filepath
    
    def generate_summary_statistics(self):
        """
        Génère des statistiques récapitulatives sur les données enrichies.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi pour générer des statistiques.")
            return None
        
        print("Génération des statistiques récapitulatives...")
        
        # Création du répertoire pour les statistiques
        os.makedirs("data_enriched/stats", exist_ok=True)
        
        # Statistiques de base
        stats = {
            'total_draws': len(self.df_enriched),
            'date_range': {
                'start': self.df_enriched['date'].min().strftime('%Y-%m-%d'),
                'end': self.df_enriched['date'].max().strftime('%Y-%m-%d')
            },
            'main_numbers': {
                'mean_sum': self.df_enriched['Main_Sum'].mean(),
                'median_sum': self.df_enriched['Main_Sum'].median(),
                'min_sum': self.df_enriched['Main_Sum'].min(),
                'max_sum': self.df_enriched['Main_Sum'].max()
            },
            'stars': {
                'mean_sum': self.df_enriched['Stars_Sum'].mean(),
                'median_sum': self.df_enriched['Stars_Sum'].median(),
                'min_sum': self.df_enriched['Stars_Sum'].min(),
                'max_sum': self.df_enriched['Stars_Sum'].max()
            },
            'jackpot': {
                'mean': self.df_enriched['jackpot'].mean(),
                'median': self.df_enriched['jackpot'].median(),
                'min': self.df_enriched['jackpot'].min(),
                'max': self.df_enriched['jackpot'].max(),
                'total_winners': self.df_enriched['has_winner'].sum()
            }
        }
        
        # Fréquence des numéros principaux
        main_numbers_freq = {}
        for num in range(1, 51):
            count = 0
            for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
                count += (self.df_enriched[col] == num).sum()
            main_numbers_freq[num] = count
        
        # Tri par fréquence
        main_numbers_freq = {k: v for k, v in sorted(main_numbers_freq.items(), key=lambda item: item[1], reverse=True)}
        
        # Top 10 numéros principaux les plus fréquents
        stats['main_numbers']['most_frequent'] = list(main_numbers_freq.keys())[:10]
        
        # Top 10 numéros principaux les moins fréquents
        stats['main_numbers']['least_frequent'] = list(main_numbers_freq.keys())[-10:]
        
        # Fréquence des étoiles
        stars_freq = {}
        for num in range(1, 13):
            count = 0
            for col in ['E1', 'E2']:
                count += (self.df_enriched[col] == num).sum()
            stars_freq[num] = count
        
        # Tri par fréquence
        stars_freq = {k: v for k, v in sorted(stars_freq.items(), key=lambda item: item[1], reverse=True)}
        
        # Top 5 étoiles les plus fréquentes
        stats['stars']['most_frequent'] = list(stars_freq.keys())[:5]
        
        # Top 5 étoiles les moins fréquentes
        stats['stars']['least_frequent'] = list(stars_freq.keys())[-5:]
        
        # Sauvegarde des statistiques au format JSON
        with open('data_enriched/stats/summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"✅ Statistiques récapitulatives sauvegardées dans data_enriched/stats/summary_statistics.json")
        
        # Génération de graphiques
        self.generate_visualizations()
        
        return stats
    
    def generate_visualizations(self):
        """
        Génère des visualisations à partir des données enrichies.
        """
        if self.df_enriched is None:
            print("❌ Aucun DataFrame enrichi pour générer des visualisations.")
            return None
        
        print("Génération des visualisations...")
        
        # Création du répertoire pour les visualisations
        os.makedirs("data_enriched/visualizations", exist_ok=True)
        
        # Configuration de Matplotlib
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # 1. Distribution de la somme des numéros principaux
        plt.figure(figsize=(12, 8))
        sns.histplot(self.df_enriched['Main_Sum'], kde=True, bins=30)
        plt.title('Distribution de la somme des numéros principaux')
        plt.xlabel('Somme')
        plt.ylabel('Fréquence')
        plt.savefig('data_enriched/visualizations/main_sum_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution de la somme des étoiles
        plt.figure(figsize=(12, 8))
        sns.histplot(self.df_enriched['Stars_Sum'], kde=True, bins=15)
        plt.title('Distribution de la somme des étoiles')
        plt.xlabel('Somme')
        plt.ylabel('Fréquence')
        plt.savefig('data_enriched/visualizations/stars_sum_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Fréquence des numéros principaux
        main_numbers_freq = {}
        for num in range(1, 51):
            count = 0
            for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
                count += (self.df_enriched[col] == num).sum()
            main_numbers_freq[num] = count
        
        plt.figure(figsize=(15, 8))
        plt.bar(main_numbers_freq.keys(), main_numbers_freq.values())
        plt.title('Fréquence d\'apparition des numéros principaux')
        plt.xlabel('Numéro')
        plt.ylabel('Fréquence')
        plt.xticks(range(1, 51))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('data_enriched/visualizations/main_numbers_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Fréquence des étoiles
        stars_freq = {}
        for num in range(1, 13):
            count = 0
            for col in ['E1', 'E2']:
                count += (self.df_enriched[col] == num).sum()
            stars_freq[num] = count
        
        plt.figure(figsize=(12, 8))
        plt.bar(stars_freq.keys(), stars_freq.values())
        plt.title('Fréquence d\'apparition des étoiles')
        plt.xlabel('Étoile')
        plt.ylabel('Fréquence')
        plt.xticks(range(1, 13))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('data_enriched/visualizations/stars_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Évolution du jackpot au fil du temps
        plt.figure(figsize=(15, 8))
        plt.plot(self.df_enriched['date'], self.df_enriched['jackpot'] / 1000000, marker='o', linestyle='-', alpha=0.7)
        plt.title('Évolution du jackpot au fil du temps')
        plt.xlabel('Date')
        plt.ylabel('Jackpot (millions €)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('data_enriched/visualizations/jackpot_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Heatmap des corrélations entre les caractéristiques numériques
        numeric_cols = self.df_enriched.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df_enriched[numeric_cols].corr()
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Matrice de corrélation des caractéristiques numériques')
        plt.tight_layout()
        plt.savefig('data_enriched/visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Tendances des numéros principaux au fil du temps
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(['N1', 'N2', 'N3', 'N4', 'N5']):
            plt.subplot(5, 1, i+1)
            plt.plot(self.df_enriched['date'], self.df_enriched[col], marker='.', linestyle='-', alpha=0.7)
            plt.title(f'Évolution du numéro {col} au fil du temps')
            plt.ylabel('Numéro')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('data_enriched/visualizations/main_numbers_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Tendances des étoiles au fil du temps
        plt.figure(figsize=(15, 6))
        for i, col in enumerate(['E1', 'E2']):
            plt.subplot(2, 1, i+1)
            plt.plot(self.df_enriched['date'], self.df_enriched[col], marker='.', linestyle='-', alpha=0.7)
            plt.title(f'Évolution de l\'étoile {col} au fil du temps')
            plt.ylabel('Étoile')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('data_enriched/visualizations/stars_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualisations générées dans data_enriched/visualizations/")
        
        return True
    
    def run_full_pipeline(self):
        """
        Exécute le pipeline complet d'enrichissement des données.
        """
        print("Démarrage du pipeline d'enrichissement des données Euromillions...")
        
        # 1. Récupération des données
        self.fetch_all_draws()
        
        # 2. Conversion en DataFrame
        self.convert_to_dataframe()
        
        # 3. Ajout des caractéristiques de base
        self.add_basic_features()
        
        # 4. Ajout des caractéristiques de fréquence
        self.add_frequency_features()
        
        # 5. Ajout des caractéristiques de tendance
        self.add_trend_features()
        
        # 6. Ajout des caractéristiques externes
        self.add_external_features()
        
        # 7. Ajout des caractéristiques liées au jackpot
        self.add_jackpot_features()
        
        # 8. Ajout des caractéristiques statistiques avancées
        self.add_advanced_statistical_features()
        
        # 9. Ajout des caractéristiques basées sur des patterns
        self.add_pattern_features()
        
        # 10. Sauvegarde des données
        self.save_data()
        
        # 11. Génération des statistiques récapitulatives
        self.generate_summary_statistics()
        
        print("✅ Pipeline d'enrichissement des données Euromillions terminé avec succès!")
        
        return self.df_enriched

# Exécution du pipeline
if __name__ == "__main__":
    enricher = EuromillionsDataEnricher()
    enricher.run_full_pipeline()

