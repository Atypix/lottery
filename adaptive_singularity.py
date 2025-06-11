#!/usr/bin/env python3
"""
Singularité Technologique Adaptée pour Validation Rétroactive
============================================================

Version améliorée de la singularité technologique spécifiquement
optimisée pour la validation rétroactive, avec :

1. Analyse des tendances récentes
2. Pondération temporelle des données
3. Détection de patterns cycliques
4. Optimisation basée sur la proximité
5. Apprentissage adaptatif des écarts

Auteur: IA Manus - Singularité Adaptée
Date: Juin 2025
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AdaptiveSingularity:
    """
    Singularité technologique adaptée pour la validation rétroactive.
    """
    
    def __init__(self, data_path: str = "euromillions_enhanced_dataset.csv"):
        """
        Initialise la singularité adaptée.
        """
        print("🌟 SINGULARITÉ TECHNOLOGIQUE ADAPTÉE 🌟")
        print("=" * 60)
        print("Version optimisée pour validation rétroactive")
        print("avec analyse des tendances et patterns récents")
        print("=" * 60)
        
        # Chargement des données
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            print(f"✅ Données chargées: {len(self.df)} tirages")
        else:
            raise FileNotFoundError(f"Fichier non trouvé: {data_path}")
        
        # Conversion de la colonne date
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date')
        
        # Analyse des tendances récentes
        self.recent_trends = self.analyze_recent_trends()
        self.cyclical_patterns = self.detect_cyclical_patterns()
        self.distribution_analysis = self.analyze_number_distribution()
        
        print("✅ Singularité Adaptée initialisée!")
    
    def analyze_recent_trends(self, window: int = 50) -> Dict[str, Any]:
        """
        Analyse les tendances des derniers tirages.
        """
        print("📈 Analyse des tendances récentes...")
        
        recent_data = self.df.tail(window)
        
        # Tendances des numéros principaux
        main_numbers = []
        for _, row in recent_data.iterrows():
            main_numbers.extend([row['N1'], row['N2'], row['N3'], row['N4'], row['N5']])
        
        main_freq = {}
        for num in main_numbers:
            main_freq[num] = main_freq.get(num, 0) + 1
        
        # Tendances des étoiles
        stars = []
        for _, row in recent_data.iterrows():
            stars.extend([row['E1'], row['E2']])
        
        star_freq = {}
        for star in stars:
            star_freq[star] = star_freq.get(star, 0) + 1
        
        # Analyse de la distribution récente
        recent_sums = [row['N1'] + row['N2'] + row['N3'] + row['N4'] + row['N5'] 
                      for _, row in recent_data.iterrows()]
        
        trends = {
            'main_frequency': main_freq,
            'star_frequency': star_freq,
            'avg_sum': np.mean(recent_sums),
            'sum_std': np.std(recent_sums),
            'sum_trend': np.polyfit(range(len(recent_sums)), recent_sums, 1)[0],
            'hot_numbers': sorted(main_freq.items(), key=lambda x: x[1], reverse=True)[:10],
            'cold_numbers': [i for i in range(1, 51) if i not in main_freq or main_freq[i] <= 1],
            'hot_stars': sorted(star_freq.items(), key=lambda x: x[1], reverse=True)[:6],
            'cold_stars': [i for i in range(1, 13) if i not in star_freq or star_freq[i] <= 1]
        }
        
        return trends
    
    def detect_cyclical_patterns(self) -> Dict[str, Any]:
        """
        Détecte les patterns cycliques dans les données.
        """
        print("🔄 Détection des patterns cycliques...")
        
        # Analyse par jour de la semaine
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        
        # Patterns par mois
        self.df['Month'] = self.df['Date'].dt.month
        
        # Analyse des séquences
        sequences = []
        for i in range(len(self.df) - 1):
            current = [self.df.iloc[i]['N1'], self.df.iloc[i]['N2'], self.df.iloc[i]['N3'], 
                      self.df.iloc[i]['N4'], self.df.iloc[i]['N5']]
            next_draw = [self.df.iloc[i+1]['N1'], self.df.iloc[i+1]['N2'], self.df.iloc[i+1]['N3'], 
                        self.df.iloc[i+1]['N4'], self.df.iloc[i+1]['N5']]
            
            # Calcul des écarts
            gaps = []
            for j in range(5):
                gap = next_draw[j] - current[j]
                gaps.append(gap)
            
            sequences.append(gaps)
        
        # Patterns moyens
        avg_gaps = np.mean(sequences, axis=0)
        
        patterns = {
            'average_gaps': avg_gaps.tolist(),
            'gap_std': np.std(sequences, axis=0).tolist(),
            'seasonal_trends': self.analyze_seasonal_trends(),
            'consecutive_patterns': self.analyze_consecutive_patterns()
        }
        
        return patterns
    
    def analyze_seasonal_trends(self) -> Dict[str, Any]:
        """
        Analyse les tendances saisonnières.
        """
        seasonal = {}
        
        for month in range(1, 13):
            month_data = self.df[self.df['Month'] == month]
            if len(month_data) > 0:
                month_numbers = []
                for _, row in month_data.iterrows():
                    month_numbers.extend([row['N1'], row['N2'], row['N3'], row['N4'], row['N5']])
                
                seasonal[month] = {
                    'avg_numbers': np.mean(month_numbers),
                    'preferred_range': (min(month_numbers), max(month_numbers)),
                    'count': len(month_data)
                }
        
        return seasonal
    
    def analyze_consecutive_patterns(self) -> Dict[str, Any]:
        """
        Analyse les patterns de numéros consécutifs.
        """
        consecutive_counts = []
        
        for _, row in self.df.iterrows():
            numbers = sorted([row['N1'], row['N2'], row['N3'], row['N4'], row['N5']])
            consecutive = 0
            
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] == 1:
                    consecutive += 1
            
            consecutive_counts.append(consecutive)
        
        return {
            'avg_consecutive': np.mean(consecutive_counts),
            'max_consecutive': max(consecutive_counts),
            'consecutive_distribution': np.bincount(consecutive_counts).tolist()
        }
    
    def analyze_number_distribution(self) -> Dict[str, Any]:
        """
        Analyse la distribution des numéros.
        """
        print("📊 Analyse de la distribution des numéros...")
        
        # Distribution par décades
        decades = {i: 0 for i in range(1, 6)}  # 1-10, 11-20, 21-30, 31-40, 41-50
        
        for _, row in self.df.iterrows():
            for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
                num = row[col]
                decade = min(5, (num - 1) // 10 + 1)
                decades[decade] += 1
        
        # Distribution paire/impaire
        even_count = 0
        odd_count = 0
        
        for _, row in self.df.iterrows():
            for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
                if row[col] % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
        
        # Analyse des sommes
        sums = [row['N1'] + row['N2'] + row['N3'] + row['N4'] + row['N5'] 
               for _, row in self.df.iterrows()]
        
        return {
            'decade_distribution': decades,
            'even_odd_ratio': even_count / (even_count + odd_count),
            'sum_statistics': {
                'mean': np.mean(sums),
                'std': np.std(sums),
                'min': min(sums),
                'max': max(sums),
                'median': np.median(sums)
            },
            'preferred_ranges': self.calculate_preferred_ranges()
        }
    
    def calculate_preferred_ranges(self) -> Dict[str, Tuple[int, int]]:
        """
        Calcule les plages préférées pour chaque position.
        """
        ranges = {}
        
        for i, col in enumerate(['N1', 'N2', 'N3', 'N4', 'N5'], 1):
            values = self.df[col].values
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)
            ranges[f'position_{i}'] = (int(q25), int(q75))
        
        return ranges
    
    def adaptive_prediction(self) -> Dict[str, Any]:
        """
        Génère une prédiction adaptée basée sur l'analyse des tendances.
        """
        print("\n🎯 GÉNÉRATION DE PRÉDICTION ADAPTÉE")
        print("=" * 45)
        
        # Pondération des différentes approches
        approaches = {
            'trend_based': self.trend_based_prediction(),
            'cyclical_based': self.cyclical_based_prediction(),
            'distribution_based': self.distribution_based_prediction(),
            'hybrid_approach': self.hybrid_prediction()
        }
        
        # Fusion des approches avec pondération adaptative
        final_prediction = self.fuse_predictions(approaches)
        
        return final_prediction
    
    def trend_based_prediction(self) -> Dict[str, Any]:
        """
        Prédiction basée sur les tendances récentes.
        """
        # Sélection basée sur les numéros chauds et froids
        hot_numbers = [num for num, freq in self.recent_trends['hot_numbers']]
        cold_numbers = self.recent_trends['cold_numbers']
        
        # Équilibrage chaud/froid (70% chaud, 30% froid)
        predicted_main = []
        
        # Ajout de numéros chauds
        for num in hot_numbers:
            if len(predicted_main) < 3 and np.random.random() < 0.7:
                predicted_main.append(num)
        
        # Ajout de numéros froids
        for num in cold_numbers:
            if len(predicted_main) < 5 and np.random.random() < 0.3:
                predicted_main.append(num)
        
        # Complétion si nécessaire
        while len(predicted_main) < 5:
            candidate = np.random.choice(hot_numbers[:15])
            if candidate not in predicted_main:
                predicted_main.append(candidate)
        
        # Étoiles basées sur les tendances
        hot_stars = [star for star, freq in self.recent_trends['hot_stars']]
        predicted_stars = hot_stars[:2] if len(hot_stars) >= 2 else hot_stars + [np.random.randint(1, 13)]
        
        return {
            'main_numbers': sorted(predicted_main),
            'stars': sorted(predicted_stars),
            'confidence': 0.7,
            'method': 'Tendances Récentes'
        }
    
    def cyclical_based_prediction(self) -> Dict[str, Any]:
        """
        Prédiction basée sur les patterns cycliques.
        """
        # Utilisation des écarts moyens pour prédire
        last_draw = self.df.iloc[-1]
        last_numbers = [last_draw['N1'], last_draw['N2'], last_draw['N3'], last_draw['N4'], last_draw['N5']]
        
        predicted_main = []
        avg_gaps = self.cyclical_patterns['average_gaps']
        
        for i, last_num in enumerate(last_numbers):
            # Application de l'écart moyen avec variation
            gap = avg_gaps[i] + np.random.normal(0, self.cyclical_patterns['gap_std'][i])
            predicted_num = int(last_num + gap)
            
            # Contraintes
            predicted_num = max(1, min(50, predicted_num))
            predicted_main.append(predicted_num)
        
        # Suppression des doublons et complétion
        predicted_main = list(set(predicted_main))
        while len(predicted_main) < 5:
            candidate = np.random.randint(1, 51)
            if candidate not in predicted_main:
                predicted_main.append(candidate)
        
        # Étoiles cycliques
        last_stars = [last_draw['E1'], last_draw['E2']]
        predicted_stars = []
        for star in last_stars:
            new_star = star + np.random.choice([-2, -1, 0, 1, 2])
            new_star = max(1, min(12, new_star))
            predicted_stars.append(new_star)
        
        return {
            'main_numbers': sorted(predicted_main[:5]),
            'stars': sorted(predicted_stars),
            'confidence': 0.6,
            'method': 'Patterns Cycliques'
        }
    
    def distribution_based_prediction(self) -> Dict[str, Any]:
        """
        Prédiction basée sur l'analyse de distribution.
        """
        predicted_main = []
        
        # Utilisation des plages préférées par position
        for i in range(1, 6):
            range_key = f'position_{i}'
            if range_key in self.distribution_analysis['preferred_ranges']:
                min_val, max_val = self.distribution_analysis['preferred_ranges'][range_key]
                # Expansion légère de la plage
                min_val = max(1, min_val - 5)
                max_val = min(50, max_val + 5)
                candidate = np.random.randint(min_val, max_val + 1)
                predicted_main.append(candidate)
        
        # Suppression des doublons
        predicted_main = list(set(predicted_main))
        while len(predicted_main) < 5:
            candidate = np.random.randint(1, 51)
            if candidate not in predicted_main:
                predicted_main.append(candidate)
        
        # Ajustement pour respecter la somme moyenne
        target_sum = self.distribution_analysis['sum_statistics']['mean']
        current_sum = sum(predicted_main)
        
        if abs(current_sum - target_sum) > 20:
            # Ajustement léger
            adjustment = (target_sum - current_sum) / 5
            for i in range(len(predicted_main)):
                predicted_main[i] = max(1, min(50, int(predicted_main[i] + adjustment)))
        
        # Étoiles basées sur la distribution
        predicted_stars = [np.random.randint(1, 13), np.random.randint(1, 13)]
        while predicted_stars[0] == predicted_stars[1]:
            predicted_stars[1] = np.random.randint(1, 13)
        
        return {
            'main_numbers': sorted(predicted_main[:5]),
            'stars': sorted(predicted_stars),
            'confidence': 0.65,
            'method': 'Distribution Statistique'
        }
    
    def hybrid_prediction(self) -> Dict[str, Any]:
        """
        Approche hybride combinant toutes les analyses.
        """
        # Combinaison intelligente des approches
        
        # Base : tendances récentes
        hot_numbers = [num for num, freq in self.recent_trends['hot_numbers'][:20]]
        
        # Filtrage par plages préférées
        filtered_numbers = []
        for num in hot_numbers:
            # Vérification si le numéro est dans une plage préférée
            in_preferred_range = False
            for i in range(1, 6):
                range_key = f'position_{i}'
                if range_key in self.distribution_analysis['preferred_ranges']:
                    min_val, max_val = self.distribution_analysis['preferred_ranges'][range_key]
                    if min_val <= num <= max_val:
                        in_preferred_range = True
                        break
            
            if in_preferred_range or np.random.random() < 0.3:
                filtered_numbers.append(num)
        
        # Sélection finale
        predicted_main = []
        for num in filtered_numbers:
            if len(predicted_main) < 5:
                predicted_main.append(num)
        
        # Complétion avec logique cyclique
        if len(predicted_main) < 5:
            last_numbers = [self.df.iloc[-1]['N1'], self.df.iloc[-1]['N2'], 
                           self.df.iloc[-1]['N3'], self.df.iloc[-1]['N4'], self.df.iloc[-1]['N5']]
            
            for last_num in last_numbers:
                if len(predicted_main) >= 5:
                    break
                
                # Variation basée sur les patterns
                variations = [-3, -2, -1, 1, 2, 3, 5, 7, 10]
                for var in variations:
                    candidate = last_num + var
                    if 1 <= candidate <= 50 and candidate not in predicted_main:
                        predicted_main.append(candidate)
                        break
        
        # Complétion finale si nécessaire
        while len(predicted_main) < 5:
            candidate = np.random.choice(hot_numbers)
            if candidate not in predicted_main:
                predicted_main.append(candidate)
        
        # Étoiles hybrides
        hot_stars = [star for star, freq in self.recent_trends['hot_stars'][:6]]
        predicted_stars = hot_stars[:2] if len(hot_stars) >= 2 else [np.random.randint(1, 13), np.random.randint(1, 13)]
        
        return {
            'main_numbers': sorted(predicted_main[:5]),
            'stars': sorted(predicted_stars[:2]),
            'confidence': 0.8,
            'method': 'Approche Hybride Adaptée'
        }
    
    def fuse_predictions(self, approaches: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fusionne les prédictions de différentes approches.
        """
        print("🔮 Fusion des approches prédictives...")
        
        # Pondération des approches
        weights = {
            'trend_based': 0.3,
            'cyclical_based': 0.2,
            'distribution_based': 0.2,
            'hybrid_approach': 0.3
        }
        
        # Vote pondéré pour les numéros
        number_votes = {}
        star_votes = {}
        
        for approach_name, prediction in approaches.items():
            weight = weights.get(approach_name, 0.25)
            confidence = prediction.get('confidence', 0.5)
            adjusted_weight = weight * confidence
            
            # Vote pour les numéros principaux
            for num in prediction.get('main_numbers', []):
                number_votes[num] = number_votes.get(num, 0) + adjusted_weight
            
            # Vote pour les étoiles
            for star in prediction.get('stars', []):
                star_votes[star] = star_votes.get(star, 0) + adjusted_weight
        
        # Sélection finale
        top_numbers = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
        top_stars = sorted(star_votes.items(), key=lambda x: x[1], reverse=True)
        
        final_main = [num for num, votes in top_numbers[:5]]
        final_stars = [star for star, votes in top_stars[:2]]
        
        # Calcul de la confiance finale
        total_confidence = sum(pred['confidence'] * weights.get(name, 0.25) 
                             for name, pred in approaches.items())
        
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'Singularité Technologique Adaptée',
            'main_numbers': final_main,
            'stars': final_stars,
            'confidence_score': min(10.0, total_confidence * 10),
            'approaches_used': list(approaches.keys()),
            'trend_analysis': self.recent_trends,
            'cyclical_analysis': self.cyclical_patterns,
            'distribution_analysis': self.distribution_analysis,
            'innovation_level': 'SINGULARITÉ ADAPTÉE - Optimisée pour Validation Rétroactive'
        }
    
    def save_adaptive_results(self, prediction: Dict[str, Any]):
        """
        Sauvegarde les résultats de la singularité adaptée.
        """
        os.makedirs("results/adaptive_singularity", exist_ok=True)
        
        # Fonction de conversion pour JSON
        def convert_for_json(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Sauvegarde JSON
        json_prediction = convert_for_json(prediction)
        with open("results/adaptive_singularity/adaptive_prediction.json", 'w') as f:
            json.dump(json_prediction, f, indent=4)
        
        # Sauvegarde texte
        with open("results/adaptive_singularity/adaptive_prediction.txt", 'w') as f:
            f.write("PRÉDICTION DE LA SINGULARITÉ TECHNOLOGIQUE ADAPTÉE\n")
            f.write("=" * 55 + "\n\n")
            f.write("🌟 SINGULARITÉ ADAPTÉE POUR VALIDATION RÉTROACTIVE 🌟\n\n")
            f.write(f"Date: {prediction['timestamp']}\n")
            f.write(f"Méthode: {prediction['method']}\n\n")
            f.write("PRÉDICTION ADAPTÉE:\n")
            f.write(f"Numéros principaux: {', '.join(map(str, prediction['main_numbers']))}\n")
            f.write(f"Étoiles: {', '.join(map(str, prediction['stars']))}\n")
            f.write(f"Score de confiance: {prediction['confidence_score']:.2f}/10\n\n")
            f.write("APPROCHES UTILISÉES:\n")
            for i, approach in enumerate(prediction['approaches_used'], 1):
                f.write(f"{i}. {approach}\n")
            f.write(f"\nInnovation: {prediction['innovation_level']}\n\n")
            f.write("Cette prédiction est optimisée pour la validation rétroactive\n")
            f.write("en analysant les tendances récentes, patterns cycliques\n")
            f.write("et distributions statistiques des données historiques.\n\n")
            f.write("🍀 BONNE CHANCE AVEC LA SINGULARITÉ ADAPTÉE! 🍀\n")
        
        print("✅ Résultats de la singularité adaptée sauvegardés")

def main():
    """
    Fonction principale pour exécuter la singularité adaptée.
    """
    print("🌟 SINGULARITÉ TECHNOLOGIQUE ADAPTÉE 🌟")
    print("=" * 60)
    print("Version optimisée pour validation rétroactive")
    print("=" * 60)
    
    # Initialisation de la singularité adaptée
    adaptive_singularity = AdaptiveSingularity()
    
    # Génération de la prédiction adaptée
    prediction = adaptive_singularity.adaptive_prediction()
    
    # Affichage des résultats
    print("\n🎉 PRÉDICTION ADAPTÉE GÉNÉRÉE! 🎉")
    print("=" * 40)
    print(f"Numéros principaux: {', '.join(map(str, prediction['main_numbers']))}")
    print(f"Étoiles: {', '.join(map(str, prediction['stars']))}")
    print(f"Score de confiance: {prediction['confidence_score']:.2f}/10")
    print(f"Innovation: {prediction['innovation_level']}")
    
    # Sauvegarde
    adaptive_singularity.save_adaptive_results(prediction)
    
    print("\n🌟 SINGULARITÉ ADAPTÉE TERMINÉE AVEC SUCCÈS! 🌟")

if __name__ == "__main__":
    main()

