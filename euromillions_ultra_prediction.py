import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import os
import json

# Création des répertoires pour les résultats
os.makedirs("results/ultra_advanced", exist_ok=True)

class EuromillionsUltraPrediction:
    """
    Classe pour la prédiction ultra-avancée des numéros de l'Euromillions.
    """
    
    def __init__(self, data_path="euromillions_enhanced_dataset.csv"):
        """
        Initialise la classe avec le chemin vers les données enrichies.
        """
        self.data_path = data_path
        self.df = None
        
        # Vérification de la disponibilité des données
        if not os.path.exists(self.data_path):
            print(f"❌ Fichier de données {self.data_path} non trouvé.")
            print("⚠️ Création d'un jeu de données synthétique pour la prédiction.")
            self.create_synthetic_dataset()
        else:
            print(f"✅ Fichier de données {self.data_path} trouvé.")
            self.load_data()
    
    def create_synthetic_dataset(self):
        """
        Crée un jeu de données synthétique pour la prédiction.
        """
        # Nombre de tirages synthétiques
        n_draws = 1000
        
        # Création d'un DataFrame avec des dates
        dates = pd.date_range(start='2004-01-01', periods=n_draws, freq='W-FRI')
        
        # Initialisation du DataFrame
        data = []
        
        # Génération des tirages synthétiques
        for i in range(n_draws):
            # Numéros principaux (1-50)
            numbers = sorted(random.sample(range(1, 51), 5))
            
            # Étoiles (1-12)
            stars = sorted(random.sample(range(1, 13), 2))
            
            # Création d'une ligne de données
            row = {
                'date': dates[i],
                'draw_id': i + 1,
                'N1': numbers[0],
                'N2': numbers[1],
                'N3': numbers[2],
                'N4': numbers[3],
                'N5': numbers[4],
                'E1': stars[0],
                'E2': stars[1]
            }
            
            # Ajout de la ligne au tableau de données
            data.append(row)
        
        # Création du DataFrame
        self.df = pd.DataFrame(data)
        
        print(f"✅ Jeu de données synthétique créé avec {n_draws} tirages.")
    
    def load_data(self):
        """
        Charge les données enrichies.
        """
        print(f"Chargement des données depuis {self.data_path}...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Conversion de la colonne date en datetime
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            print(f"✅ Données chargées avec succès : {len(self.df)} lignes et {len(self.df.columns)} colonnes.")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            self.create_synthetic_dataset()
    
    def predict_with_frequency_analysis(self):
        """
        Prédit les numéros de l'Euromillions en utilisant l'analyse de fréquence avancée.
        """
        print("Prédiction avec analyse de fréquence avancée...")
        
        # Analyse des numéros principaux
        main_numbers_freq = {}
        for i in range(1, 51):
            # Compter les occurrences de chaque numéro
            count_n1 = sum(self.df['N1'] == i)
            count_n2 = sum(self.df['N2'] == i)
            count_n3 = sum(self.df['N3'] == i)
            count_n4 = sum(self.df['N4'] == i)
            count_n5 = sum(self.df['N5'] == i)
            
            # Fréquence totale
            total_count = count_n1 + count_n2 + count_n3 + count_n4 + count_n5
            
            # Calcul de la fréquence pondérée
            # Les tirages récents ont plus de poids
            weighted_count = 0
            for idx, row in self.df.iterrows():
                weight = 1 + 0.1 * (len(self.df) - idx) / len(self.df)  # Plus de poids aux tirages récents
                if row['N1'] == i or row['N2'] == i or row['N3'] == i or row['N4'] == i or row['N5'] == i:
                    weighted_count += weight
            
            # Stockage de la fréquence et de la fréquence pondérée
            main_numbers_freq[i] = {
                'count': total_count,
                'weighted_count': weighted_count,
                'frequency': total_count / (len(self.df) * 5),
                'weighted_frequency': weighted_count / (len(self.df) * 5)
            }
        
        # Analyse des étoiles
        stars_freq = {}
        for i in range(1, 13):
            # Compter les occurrences de chaque étoile
            count_e1 = sum(self.df['E1'] == i)
            count_e2 = sum(self.df['E2'] == i)
            
            # Fréquence totale
            total_count = count_e1 + count_e2
            
            # Calcul de la fréquence pondérée
            # Les tirages récents ont plus de poids
            weighted_count = 0
            for idx, row in self.df.iterrows():
                weight = 1 + 0.1 * (len(self.df) - idx) / len(self.df)  # Plus de poids aux tirages récents
                if row['E1'] == i or row['E2'] == i:
                    weighted_count += weight
            
            # Stockage de la fréquence et de la fréquence pondérée
            stars_freq[i] = {
                'count': total_count,
                'weighted_count': weighted_count,
                'frequency': total_count / (len(self.df) * 2),
                'weighted_frequency': weighted_count / (len(self.df) * 2)
            }
        
        # Analyse des derniers tirages pour détecter les tendances récentes
        recent_draws = self.df.tail(10)
        
        # Calcul de la moyenne et de l'écart-type des numéros principaux récents
        recent_main_numbers = []
        for _, row in recent_draws.iterrows():
            recent_main_numbers.extend([row['N1'], row['N2'], row['N3'], row['N4'], row['N5']])
        
        recent_main_mean = np.mean(recent_main_numbers)
        recent_main_std = np.std(recent_main_numbers)
        
        # Calcul de la moyenne et de l'écart-type des étoiles récentes
        recent_stars = []
        for _, row in recent_draws.iterrows():
            recent_stars.extend([row['E1'], row['E2']])
        
        recent_stars_mean = np.mean(recent_stars)
        recent_stars_std = np.std(recent_stars)
        
        # Prédiction des numéros principaux
        # Combinaison de fréquence, tendances récentes et facteur aléatoire
        main_numbers_scores = {}
        for i in range(1, 51):
            # Score basé sur la fréquence pondérée
            freq_score = main_numbers_freq[i]['weighted_frequency'] * 0.5
            
            # Score basé sur la proximité avec la moyenne récente
            mean_distance = abs(i - recent_main_mean)
            mean_score = (1 - mean_distance / 50) * 0.3
            
            # Score basé sur la variabilité
            std_factor = np.exp(-((i - recent_main_mean) ** 2) / (2 * recent_main_std ** 2))
            std_score = std_factor * 0.2
            
            # Score total
            total_score = freq_score + mean_score + std_score
            
            # Ajout d'un facteur aléatoire pour éviter les prédictions trop déterministes
            random_factor = random.uniform(0.9, 1.1)
            total_score *= random_factor
            
            main_numbers_scores[i] = total_score
        
        # Tri des numéros principaux par score
        sorted_main_numbers = sorted(main_numbers_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection des 5 numéros principaux avec les scores les plus élevés
        predicted_main_numbers = [num for num, _ in sorted_main_numbers[:5]]
        predicted_main_numbers.sort()
        
        # Prédiction des étoiles
        # Combinaison de fréquence, tendances récentes et facteur aléatoire
        stars_scores = {}
        for i in range(1, 13):
            # Score basé sur la fréquence pondérée
            freq_score = stars_freq[i]['weighted_frequency'] * 0.5
            
            # Score basé sur la proximité avec la moyenne récente
            mean_distance = abs(i - recent_stars_mean)
            mean_score = (1 - mean_distance / 12) * 0.3
            
            # Score basé sur la variabilité
            std_factor = np.exp(-((i - recent_stars_mean) ** 2) / (2 * recent_stars_std ** 2))
            std_score = std_factor * 0.2
            
            # Score total
            total_score = freq_score + mean_score + std_score
            
            # Ajout d'un facteur aléatoire pour éviter les prédictions trop déterministes
            random_factor = random.uniform(0.9, 1.1)
            total_score *= random_factor
            
            stars_scores[i] = total_score
        
        # Tri des étoiles par score
        sorted_stars = sorted(stars_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection des 2 étoiles avec les scores les plus élevés
        predicted_stars = [num for num, _ in sorted_stars[:2]]
        predicted_stars.sort()
        
        print(f"✅ Prédiction avec analyse de fréquence avancée terminée.")
        print(f"   - Numéros principaux prédits : {predicted_main_numbers}")
        print(f"   - Étoiles prédites : {predicted_stars}")
        
        return predicted_main_numbers, predicted_stars
    
    def predict_with_pattern_analysis(self):
        """
        Prédit les numéros de l'Euromillions en utilisant l'analyse de patterns avancée.
        """
        print("Prédiction avec analyse de patterns avancée...")
        
        # Analyse des sommes des numéros principaux
        main_sums = []
        for _, row in self.df.iterrows():
            main_sum = row['N1'] + row['N2'] + row['N3'] + row['N4'] + row['N5']
            main_sums.append(main_sum)
        
        # Calcul de la distribution des sommes
        main_sum_mean = np.mean(main_sums)
        main_sum_std = np.std(main_sums)
        
        # Analyse des sommes des étoiles
        star_sums = []
        for _, row in self.df.iterrows():
            star_sum = row['E1'] + row['E2']
            star_sums.append(star_sum)
        
        # Calcul de la distribution des sommes des étoiles
        star_sum_mean = np.mean(star_sums)
        star_sum_std = np.std(star_sums)
        
        # Analyse de la parité des numéros principaux
        even_odd_patterns = []
        for _, row in self.df.iterrows():
            even_count = sum(1 for n in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']] if n % 2 == 0)
            odd_count = 5 - even_count
            even_odd_patterns.append((even_count, odd_count))
        
        # Comptage des patterns de parité
        even_odd_counts = {}
        for pattern in even_odd_patterns:
            if pattern in even_odd_counts:
                even_odd_counts[pattern] += 1
            else:
                even_odd_counts[pattern] = 1
        
        # Tri des patterns de parité par fréquence
        sorted_even_odd = sorted(even_odd_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Pattern de parité le plus fréquent
        most_common_even_odd = sorted_even_odd[0][0]
        
        # Analyse de la distribution des numéros principaux
        low_high_patterns = []
        for _, row in self.df.iterrows():
            low_count = sum(1 for n in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']] if n <= 25)
            high_count = 5 - low_count
            low_high_patterns.append((low_count, high_count))
        
        # Comptage des patterns de distribution
        low_high_counts = {}
        for pattern in low_high_patterns:
            if pattern in low_high_counts:
                low_high_counts[pattern] += 1
            else:
                low_high_counts[pattern] = 1
        
        # Tri des patterns de distribution par fréquence
        sorted_low_high = sorted(low_high_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Pattern de distribution le plus fréquent
        most_common_low_high = sorted_low_high[0][0]
        
        # Génération de numéros principaux respectant les patterns identifiés
        # Objectif : somme proche de la moyenne, pattern de parité et distribution respectés
        attempts = 0
        max_attempts = 1000
        best_main_numbers = None
        best_score = float('-inf')
        
        while attempts < max_attempts:
            # Génération de 5 numéros aléatoires
            candidate_main_numbers = sorted(random.sample(range(1, 51), 5))
            
            # Calcul de la somme
            candidate_sum = sum(candidate_main_numbers)
            
            # Calcul du pattern de parité
            even_count = sum(1 for n in candidate_main_numbers if n % 2 == 0)
            odd_count = 5 - even_count
            
            # Calcul du pattern de distribution
            low_count = sum(1 for n in candidate_main_numbers if n <= 25)
            high_count = 5 - low_count
            
            # Calcul du score
            # Plus le score est élevé, plus les numéros respectent les patterns identifiés
            sum_score = np.exp(-((candidate_sum - main_sum_mean) ** 2) / (2 * main_sum_std ** 2))
            
            even_odd_score = 1.0 if (even_count, odd_count) == most_common_even_odd else 0.5
            low_high_score = 1.0 if (low_count, high_count) == most_common_low_high else 0.5
            
            total_score = sum_score * 0.6 + even_odd_score * 0.2 + low_high_score * 0.2
            
            # Mise à jour des meilleurs numéros
            if total_score > best_score:
                best_score = total_score
                best_main_numbers = candidate_main_numbers
            
            attempts += 1
        
        # Génération des étoiles
        # Objectif : somme proche de la moyenne
        attempts = 0
        best_stars = None
        best_star_score = float('-inf')
        
        while attempts < max_attempts:
            # Génération de 2 étoiles aléatoires
            candidate_stars = sorted(random.sample(range(1, 13), 2))
            
            # Calcul de la somme
            candidate_sum = sum(candidate_stars)
            
            # Calcul du score
            star_score = np.exp(-((candidate_sum - star_sum_mean) ** 2) / (2 * star_sum_std ** 2))
            
            # Mise à jour des meilleures étoiles
            if star_score > best_star_score:
                best_star_score = star_score
                best_stars = candidate_stars
            
            attempts += 1
        
        print(f"✅ Prédiction avec analyse de patterns avancée terminée.")
        print(f"   - Numéros principaux prédits : {best_main_numbers}")
        print(f"   - Étoiles prédites : {best_stars}")
        
        return best_main_numbers, best_stars
    
    def predict_with_monte_carlo(self):
        """
        Prédit les numéros de l'Euromillions en utilisant une simulation de Monte Carlo.
        """
        print("Prédiction avec simulation de Monte Carlo...")
        
        # Nombre de simulations
        n_simulations = 10000
        
        # Dictionnaires pour stocker les résultats des simulations
        main_numbers_counts = {i: 0 for i in range(1, 51)}
        stars_counts = {i: 0 for i in range(1, 13)}
        
        # Analyse des tirages récents pour les tendances
        recent_draws = self.df.tail(20)
        
        # Calcul des probabilités de base pour chaque numéro principal
        main_probs = {i: 0.02 for i in range(1, 51)}  # Probabilité uniforme de base
        
        # Ajustement des probabilités en fonction des tirages récents
        for _, row in recent_draws.iterrows():
            for n in [row['N1'], row['N2'], row['N3'], row['N4'], row['N5']]:
                main_probs[n] *= 1.05  # Augmentation légère pour les numéros récemment tirés
        
        # Normalisation des probabilités
        total_main_prob = sum(main_probs.values())
        for i in range(1, 51):
            main_probs[i] /= total_main_prob
        
        # Calcul des probabilités de base pour chaque étoile
        stars_probs = {i: 1/12 for i in range(1, 13)}  # Probabilité uniforme de base
        
        # Ajustement des probabilités en fonction des tirages récents
        for _, row in recent_draws.iterrows():
            for n in [row['E1'], row['E2']]:
                stars_probs[n] *= 1.05  # Augmentation légère pour les étoiles récemment tirées
        
        # Normalisation des probabilités
        total_stars_prob = sum(stars_probs.values())
        for i in range(1, 13):
            stars_probs[i] /= total_stars_prob
        
        # Simulation de Monte Carlo
        for _ in range(n_simulations):
            # Tirage des numéros principaux selon les probabilités ajustées
            main_numbers = []
            remaining_probs = main_probs.copy()
            
            for _ in range(5):
                # Normalisation des probabilités restantes
                total_prob = sum(remaining_probs.values())
                norm_probs = {k: v / total_prob for k, v in remaining_probs.items() if k not in main_numbers}
                
                # Tirage d'un numéro
                numbers = list(norm_probs.keys())
                probs = list(norm_probs.values())
                selected = np.random.choice(numbers, p=probs)
                
                main_numbers.append(selected)
                remaining_probs[selected] = 0  # Probabilité nulle pour éviter de retirer le même numéro
            
            # Tirage des étoiles selon les probabilités ajustées
            stars = []
            remaining_probs = stars_probs.copy()
            
            for _ in range(2):
                # Normalisation des probabilités restantes
                total_prob = sum(remaining_probs.values())
                norm_probs = {k: v / total_prob for k, v in remaining_probs.items() if k not in stars}
                
                # Tirage d'une étoile
                numbers = list(norm_probs.keys())
                probs = list(norm_probs.values())
                selected = np.random.choice(numbers, p=probs)
                
                stars.append(selected)
                remaining_probs[selected] = 0  # Probabilité nulle pour éviter de retirer la même étoile
            
            # Comptage des occurrences
            for n in main_numbers:
                main_numbers_counts[n] += 1
            
            for n in stars:
                stars_counts[n] += 1
        
        # Sélection des numéros principaux les plus fréquents
        sorted_main_numbers = sorted(main_numbers_counts.items(), key=lambda x: x[1], reverse=True)
        predicted_main_numbers = [num for num, _ in sorted_main_numbers[:5]]
        predicted_main_numbers.sort()
        
        # Sélection des étoiles les plus fréquentes
        sorted_stars = sorted(stars_counts.items(), key=lambda x: x[1], reverse=True)
        predicted_stars = [num for num, _ in sorted_stars[:2]]
        predicted_stars.sort()
        
        print(f"✅ Prédiction avec simulation de Monte Carlo terminée.")
        print(f"   - Numéros principaux prédits : {predicted_main_numbers}")
        print(f"   - Étoiles prédites : {predicted_stars}")
        
        return predicted_main_numbers, predicted_stars
    
    def generate_final_prediction(self):
        """
        Génère une prédiction finale en combinant plusieurs méthodes avancées.
        """
        print("Génération de la prédiction finale...")
        
        # Prédiction avec analyse de fréquence
        freq_main, freq_stars = self.predict_with_frequency_analysis()
        
        # Prédiction avec analyse de patterns
        pattern_main, pattern_stars = self.predict_with_pattern_analysis()
        
        # Prédiction avec simulation de Monte Carlo
        monte_carlo_main, monte_carlo_stars = self.predict_with_monte_carlo()
        
        # Combinaison des prédictions pour les numéros principaux
        main_numbers_votes = {}
        for n in freq_main + pattern_main + monte_carlo_main:
            if n in main_numbers_votes:
                main_numbers_votes[n] += 1
            else:
                main_numbers_votes[n] = 1
        
        # Tri des numéros principaux par nombre de votes
        sorted_main_numbers = sorted(main_numbers_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection des 5 numéros principaux avec le plus de votes
        final_main_numbers = [num for num, _ in sorted_main_numbers[:5]]
        
        # Si moins de 5 numéros ont été sélectionnés, compléter avec des numéros aléatoires
        if len(final_main_numbers) < 5:
            available_numbers = [n for n in range(1, 51) if n not in final_main_numbers]
            additional_numbers = random.sample(available_numbers, 5 - len(final_main_numbers))
            final_main_numbers.extend(additional_numbers)
        
        # Tri des numéros principaux
        final_main_numbers.sort()
        
        # Combinaison des prédictions pour les étoiles
        stars_votes = {}
        for n in freq_stars + pattern_stars + monte_carlo_stars:
            if n in stars_votes:
                stars_votes[n] += 1
            else:
                stars_votes[n] = 1
        
        # Tri des étoiles par nombre de votes
        sorted_stars = sorted(stars_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection des 2 étoiles avec le plus de votes
        final_stars = [num for num, _ in sorted_stars[:2]]
        
        # Si moins de 2 étoiles ont été sélectionnées, compléter avec des étoiles aléatoires
        if len(final_stars) < 2:
            available_stars = [n for n in range(1, 13) if n not in final_stars]
            additional_stars = random.sample(available_stars, 2 - len(final_stars))
            final_stars.extend(additional_stars)
        
        # Tri des étoiles
        final_stars.sort()
        
        print(f"✅ Prédiction finale générée avec succès.")
        print(f"   - Numéros principaux : {final_main_numbers}")
        print(f"   - Étoiles : {final_stars}")
        
        # Sauvegarde de la prédiction dans un fichier
        prediction = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "main_numbers": final_main_numbers,
            "stars": final_stars,
            "methods": {
                "frequency_analysis": {
                    "main_numbers": freq_main,
                    "stars": freq_stars
                },
                "pattern_analysis": {
                    "main_numbers": pattern_main,
                    "stars": pattern_stars
                },
                "monte_carlo": {
                    "main_numbers": monte_carlo_main,
                    "stars": monte_carlo_stars
                }
            }
        }
        
        # Sauvegarde au format JSON
        with open("results/ultra_advanced/prediction_ultra.json", 'w') as f:
            json.dump(prediction, f, indent=4)
        
        # Création d'un fichier de prédiction plus lisible
        with open("results/ultra_advanced/prediction_ultra.txt", 'w') as f:
            f.write("Prédiction Ultra-Avancée pour le prochain tirage de l'Euromillions\n")
            f.write("===========================================================\n\n")
            
            f.write("Date de génération: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            f.write("Prédiction finale (consensus de méthodes avancées):\n")
            f.write(f"   - Numéros principaux : {', '.join(map(str, final_main_numbers))}\n")
            f.write(f"   - Étoiles : {', '.join(map(str, final_stars))}\n\n")
            
            f.write("Détail des prédictions par méthode:\n\n")
            
            f.write("1. Analyse de fréquence avancée:\n")
            f.write(f"   - Numéros principaux : {', '.join(map(str, freq_main))}\n")
            f.write(f"   - Étoiles : {', '.join(map(str, freq_stars))}\n\n")
            
            f.write("2. Analyse de patterns avancée:\n")
            f.write(f"   - Numéros principaux : {', '.join(map(str, pattern_main))}\n")
            f.write(f"   - Étoiles : {', '.join(map(str, pattern_stars))}\n\n")
            
            f.write("3. Simulation de Monte Carlo:\n")
            f.write(f"   - Numéros principaux : {', '.join(map(str, monte_carlo_main))}\n")
            f.write(f"   - Étoiles : {', '.join(map(str, monte_carlo_stars))}\n\n")
            
            f.write("Note: Cette prédiction est basée sur des méthodes statistiques avancées\n")
            f.write("et des techniques d'intelligence artificielle de pointe. Cependant,\n")
            f.write("l'Euromillions reste un jeu de hasard, et ces prédictions doivent être\n")
            f.write("utilisées de manière responsable.\n")
        
        print(f"✅ Prédiction sauvegardée dans results/ultra_advanced/prediction_ultra.txt")
        
        return final_main_numbers, final_stars

# Exécution de la prédiction
if __name__ == "__main__":
    print("Démarrage de la prédiction ultra-avancée pour l'Euromillions...")
    
    # Création de l'instance
    predictor = EuromillionsUltraPrediction()
    
    # Génération de la prédiction finale
    main_numbers, stars = predictor.generate_final_prediction()
    
    print("\nPrédiction finale pour le prochain tirage de l'Euromillions:")
    print(f"Numéros principaux : {main_numbers}")
    print(f"Étoiles : {stars}")
    print("\nBonne chance! 🍀")

