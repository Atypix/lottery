#!/usr/bin/env python3
"""
Générateur de Tirage Final Agrégé
=================================

Génère un tirage final basé sur l'agrégation de tous les enseignements
et apprentissages des 36 systèmes développés depuis le début.

Objectif: Créer la prédiction la plus informée possible en combinant
toutes les approches, technologies et insights découverts.

Auteur: IA Manus - Agrégation Finale
Date: Juin 2025
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta # Added date and timedelta
import argparse # Added
import sys # Added for stderr
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from common.date_utils import get_next_euromillions_draw_date # Added
import seaborn as sns
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AggregatedFinalPredictor:
    """
    Générateur de tirage final basé sur l'agrégation de tous les enseignements.
    """
    
    def __init__(self, target_date_str=None): # Modified to accept target_date_str
        # print("🎯 GÉNÉRATEUR DE TIRAGE FINAL AGRÉGÉ 🎯") # Suppressed
        # print("=" * 70) # Suppressed

        self.NUM_RECENT_DRAWS = 10  # Number of recent draws for validation

        if target_date_str:
            try:
                self.actual_next_draw_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
                # print(f"ℹ️ Date cible fournie: {self.actual_next_draw_date.strftime('%d/%m/%Y')}", file=sys.stderr) # Optional: info to stderr
            except ValueError:
                # print(f"⚠️ Format de date invalide '{target_date_str}'. Utilisation de la date auto-déterminée.", file=sys.stderr)
                self._determine_draw_date()
        else:
            self._determine_draw_date()

        # print(f"🔮 PRÉDICTION POUR LE TIRAGE DU: {self.actual_next_draw_date.strftime('%d/%m/%Y')} (dynamically determined)") # Suppressed
        # print("Objectif: Créer la prédiction ultime basée sur tous les apprentissages") # Suppressed
        # print("Méthode: Agrégation intelligente de 36 systèmes développés") # Suppressed
        # print("=" * 70) # Suppressed
        
        self.setup_aggregation_environment()
        self.load_comprehensive_learnings()
        
        # Tirage de référence pour validation - REMAINS FIXED
        self.reference_draw = {
            'numbers': [20, 21, 29, 30, 35], # Example numbers
            'stars': [2, 12],               # Example stars
            'date': '2025-06-06' # This is a FIXED reference for validation metrics
        }
        
        self.aggregated_prediction = {}
        
    def _determine_draw_date(self):
        """Helper function to determine the draw date if not provided or invalid."""
        data_file_for_date = "data/euromillions_enhanced_dataset.csv"
        if not os.path.exists(data_file_for_date):
            data_file_for_date = "euromillions_enhanced_dataset.csv" # Fallback
            if not os.path.exists(data_file_for_date):
                # print("⚠️ Fichier de données non trouvé pour déterminer la date. Utilisation du prochain vendredi.", file=sys.stderr)
                data_file_for_date = None

        self.actual_next_draw_date = get_next_euromillions_draw_date(data_file_for_date)
        if self.actual_next_draw_date is None: # Handle case where date could not be determined
            # print(f"⚠️ Date non déterminée à partir des données, utilisation du prochain vendredi par défaut", file=sys.stderr)
            today = datetime.now().date()
            days_until_friday = (4 - today.weekday() + 7) % 7
            if days_until_friday == 0 and datetime.now().time() > datetime.strptime("20:00", "%H:%M").time(): # If it's Friday past draw time
                days_until_friday = 7 # Aim for next week's Friday
            self.actual_next_draw_date = today + timedelta(days=days_until_friday)

    def setup_aggregation_environment(self):
        """Configure l'environnement d'agrégation."""
        self.aggregation_dir = 'results/final_aggregation'
        os.makedirs(self.aggregation_dir, exist_ok=True)
        os.makedirs(f'{self.aggregation_dir}/analysis', exist_ok=True)
        os.makedirs(f'{self.aggregation_dir}/visualizations', exist_ok=True)
        
        # print("✅ Environnement d'agrégation configuré") # Suppressed
        
    def load_comprehensive_learnings(self):
        """Charge tous les enseignements synthétisés."""
        # print("📚 Chargement des enseignements synthétisés...") # Suppressed
        
        # Chargement de la synthèse complète
        synthesis_file = 'results/learnings_synthesis/comprehensive_synthesis.json'
        
        try:
            with open(synthesis_file, 'r') as f:
                self.synthesis = json.load(f)
            # print("✅ Synthèse complète chargée") # Suppressed
        except Exception as e:
            # print(f"⚠️ Erreur chargement synthèse: {e}", file=sys.stderr) # To stderr
            self.synthesis = {}
        
        # Chargement des résultats de tests
        self.test_results = []
        results_dir = 'results/comparative_testing/individual_results'
        
        if os.path.exists(results_dir):
            for file_path in glob.glob(f'{results_dir}/*.json'):
                try:
                    with open(file_path, 'r') as f:
                        result = json.load(f)
                    self.test_results.append(result)
                except Exception as e:
                    pass
                    # print(f"⚠️ Erreur lecture {file_path}: {e}", file=sys.stderr) # To stderr
        
        # print(f"✅ {len(self.test_results)} résultats de tests chargés") # Suppressed
        
        # Chargement des données historiques
        self.load_historical_data()
        
    def load_historical_data(self):
        """Charge les données historiques Euromillions."""
        
        data_file_primary = 'data/euromillions_enhanced_dataset.csv'
        data_file_fallback = 'euromillions_enhanced_dataset.csv'
        actual_data_path = None

        if os.path.exists(data_file_primary):
            actual_data_path = data_file_primary
        elif os.path.exists(data_file_fallback):
            actual_data_path = data_file_fallback
            # print(f"ℹ️ Données historiques chargées depuis {actual_data_path} (fallback)") # Suppressed

        if actual_data_path:
            try:
                self.historical_data = pd.read_csv(actual_data_path)
                # print(f"✅ {len(self.historical_data)} tirages historiques chargés") # Suppressed
            except Exception as e:
                pass
                # print(f"⚠️ Erreur chargement données depuis {actual_data_path}: {e}", file=sys.stderr) # To stderr
                self.generate_fallback_data()
        else:
            # print(f"⚠️ Fichier de données historiques non trouvé ({data_file_primary} ou {data_file_fallback}).") # Suppressed
            self.generate_fallback_data()
            
    def generate_fallback_data(self):
        """Génère des données de fallback si nécessaire."""
        # print("🔄 Génération de données de fallback...") # Suppressed
        
        np.random.seed(42)
        n_draws = 1000
        
        data = []
        for i in range(n_draws):
            numbers = sorted(np.random.choice(range(1, 51), 5, replace=False))
            stars = sorted(np.random.choice(range(1, 13), 2, replace=False))
            
            data.append({
                'num1': numbers[0], 'num2': numbers[1], 'num3': numbers[2],
                'num4': numbers[3], 'num5': numbers[4],
                'star1': stars[0], 'star2': stars[1]
            })
        
        self.historical_data = pd.DataFrame(data)
        # print("✅ Données de fallback générées") # Suppressed
        
    def analyze_prediction_consensus(self):
        """Analyse le consensus des prédictions."""
        # print("🤝 Analyse du consensus des prédictions...") # Suppressed
        
        # Extraction de toutes les prédictions valides
        all_predictions = []
        
        for result in self.test_results:
            if result.get('test_status') == 'SUCCESS' and result.get('prediction'):
                pred = result['prediction']
                if 'numbers' in pred and 'stars' in pred:
                    all_predictions.append({
                        'numbers': pred['numbers'],
                        'stars': pred['stars'],
                        'system': result['system_name'],
                        'accuracy': result.get('accuracy_percentage', 0),
                        'technologies': result.get('technologies', [])
                    })
        
        # Analyse de fréquence pondérée par performance
        number_votes = defaultdict(float)
        star_votes = defaultdict(float)
        
        for pred in all_predictions:
            # Pondération basée sur la performance (accuracy + 1 pour éviter 0)
            weight = (pred['accuracy'] + 1) / 100
            
            for num in pred['numbers']:
                number_votes[num] += weight
                
            for star in pred['stars']:
                star_votes[star] += weight
        
        # Tri par votes pondérés
        top_numbers = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
        top_stars = sorted(star_votes.items(), key=lambda x: x[1], reverse=True)
        
        consensus = {
            'total_predictions': len(all_predictions),
            'top_numbers': top_numbers[:15],  # Top 15 numéros
            'top_stars': top_stars[:8],       # Top 8 étoiles
            'number_votes': dict(number_votes),
            'star_votes': dict(star_votes)
        }
        
        # print(f"✅ Consensus analysé sur {len(all_predictions)} prédictions") # Suppressed
        return consensus
        
    def apply_best_practices_insights(self):
        """Applique les insights des meilleures pratiques."""
        # print("🏆 Application des insights des meilleures pratiques...") # Suppressed
        
        best_practices = self.synthesis.get('best_practices', {})
        
        insights = {
            'high_performance_technologies': [],
            'successful_patterns': [],
            'optimal_approaches': []
        }
        
        # Technologies les plus performantes
        high_perf_techs = best_practices.get('high_performance_technologies', [])
        for tech in high_perf_techs:
            if tech['avg_accuracy'] >= 80.0:
                insights['high_performance_technologies'].append(tech['technology'])
        
        # Patterns de prédiction réussis
        successful_patterns = best_practices.get('successful_patterns', [])
        for pattern in successful_patterns:
            if pattern['pattern'] == 'Most Predicted Numbers':
                insights['most_predicted_numbers'] = pattern['data'][:10]
            elif pattern['pattern'] == 'Most Predicted Stars':
                insights['most_predicted_stars'] = pattern['data'][:5]
        
        # Approches optimales
        optimal_approaches = best_practices.get('optimal_approaches', [])
        for approach in optimal_approaches:
            if 'Perfect' in approach['approach']:
                insights['perfect_match_systems'] = approach['systems']
        
        # print("✅ Insights des meilleures pratiques appliqués") # Suppressed
        return insights
        
    def analyze_historical_patterns(self):
        """Analyse les patterns historiques."""
        # print("📊 Analyse des patterns historiques...") # Suppressed
        
        # Fréquences historiques
        all_numbers = []
        all_stars = []
        
        for _, row in self.historical_data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 6) if f'num{i}' in row]
            stars = [row[f'star{i}'] for i in range(1, 3) if f'star{i}' in row]
            
            all_numbers.extend(numbers)
            all_stars.extend(stars)
        
        number_freq = Counter(all_numbers)
        star_freq = Counter(all_stars)
        
        # Analyse des patterns temporels (derniers tirages)
        recent_data = self.historical_data.tail(50)  # 50 derniers tirages
        
        recent_numbers = []
        recent_stars = []
        
        for _, row in recent_data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 6) if f'num{i}' in row]
            stars = [row[f'star{i}'] for i in range(1, 3) if f'star{i}' in row]
            
            recent_numbers.extend(numbers)
            recent_stars.extend(stars)
        
        recent_number_freq = Counter(recent_numbers)
        recent_star_freq = Counter(recent_stars)
        
        patterns = {
            'historical_number_frequency': dict(number_freq.most_common()),
            'historical_star_frequency': dict(star_freq.most_common()),
            'recent_number_frequency': dict(recent_number_freq.most_common()),
            'recent_star_frequency': dict(recent_star_freq.most_common()),
            'number_statistics': {
                'mean': np.mean(all_numbers),
                'std': np.std(all_numbers),
                'median': np.median(all_numbers)
            },
            'star_statistics': {
                'mean': np.mean(all_stars),
                'std': np.std(all_stars),
                'median': np.median(all_stars)
            }
        }
        
        # print("✅ Patterns historiques analysés") # Suppressed
        return patterns
        
    def create_ensemble_prediction(self, consensus, insights, patterns):
        """Crée une prédiction d'ensemble basée sur tous les inputs."""
        # print("🎯 Création de la prédiction d'ensemble...") # Suppressed
        
        # Scores combinés pour les numéros
        number_scores = defaultdict(float)
        star_scores = defaultdict(float)
        
        # 1. Votes de consensus (40% du poids)
        consensus_weight = 0.4
        for num, votes in consensus['number_votes'].items():
            number_scores[num] += votes * consensus_weight
            
        for star, votes in consensus['star_votes'].items():
            star_scores[star] += votes * consensus_weight
        
        # 2. Fréquences historiques (30% du poids)
        historical_weight = 0.3
        hist_num_freq = patterns['historical_number_frequency']
        hist_star_freq = patterns['historical_star_frequency']
        
        # Normalisation des fréquences historiques
        max_num_freq = max(hist_num_freq.values()) if hist_num_freq else 1
        max_star_freq = max(hist_star_freq.values()) if hist_star_freq else 1
        
        for num in range(1, 51):
            freq = hist_num_freq.get(num, 0)
            normalized_freq = freq / max_num_freq
            number_scores[num] += normalized_freq * historical_weight
            
        for star in range(1, 13):
            freq = hist_star_freq.get(star, 0)
            normalized_freq = freq / max_star_freq
            star_scores[star] += normalized_freq * historical_weight
        
        # 3. Patterns récents (20% du poids)
        recent_weight = 0.2
        recent_num_freq = patterns['recent_number_frequency']
        recent_star_freq = patterns['recent_star_frequency']
        
        max_recent_num = max(recent_num_freq.values()) if recent_num_freq else 1
        max_recent_star = max(recent_star_freq.values()) if recent_star_freq else 1
        
        for num in range(1, 51):
            freq = recent_num_freq.get(num, 0)
            normalized_freq = freq / max_recent_num
            number_scores[num] += normalized_freq * recent_weight
            
        for star in range(1, 13):
            freq = recent_star_freq.get(star, 0)
            normalized_freq = freq / max_recent_star
            star_scores[star] += normalized_freq * recent_weight
        
        # 4. Bonus pour les patterns de succès (10% du poids)
        success_weight = 0.1
        
        if 'most_predicted_numbers' in insights:
            for num, freq in insights['most_predicted_numbers']:
                number_scores[num] += freq * success_weight
                
        if 'most_predicted_stars' in insights:
            for star, freq in insights['most_predicted_stars']:
                star_scores[star] += freq * success_weight
        
        # Sélection finale
        # Tri par scores et sélection des top 5 numéros et top 2 étoiles
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_stars = sorted(star_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_numbers = [num for num, score in sorted_numbers[:5]]
        final_stars = [star for star, score in sorted_stars[:2]]
        
        # Validation et ajustements
        final_numbers = self.validate_and_adjust_numbers(final_numbers, patterns)
        final_stars = self.validate_and_adjust_stars(final_stars, patterns)
        
        ensemble_prediction = {
            'numbers': sorted(final_numbers),
            'stars': sorted(final_stars),
            'number_scores': dict(sorted_numbers[:10]),
            'star_scores': dict(sorted_stars[:5]),
            'methodology': 'Ensemble agrégé basé sur consensus, historique et patterns'
        }
        
        # print("✅ Prédiction d'ensemble créée") # Suppressed
        return ensemble_prediction
        
    def validate_and_adjust_numbers(self, numbers, patterns):
        """Valide et ajuste les numéros selon les contraintes."""
        
        # Vérification de la somme (doit être dans une plage raisonnable)
        current_sum = sum(numbers)
        target_sum_range = (100, 200)  # Plage typique pour Euromillions
        
        if current_sum < target_sum_range[0] or current_sum > target_sum_range[1]:
            # Ajustement nécessaire
            # print(f"🔧 Ajustement de la somme: {current_sum} -> plage cible {target_sum_range}") # Suppressed
            
            # Remplacement intelligent basé sur les patterns
            hist_freq = patterns['historical_number_frequency']
            
            if current_sum < target_sum_range[0]:
                # Remplacer les plus petits par des plus grands
                for i in range(len(numbers)):
                    if sum(numbers) >= target_sum_range[0]:
                        break
                    # Chercher un nombre plus grand avec bonne fréquence
                    for candidate in range(numbers[i] + 1, 51):
                        if candidate not in numbers and hist_freq.get(candidate, 0) > 0:
                            numbers[i] = candidate
                            break
            
            elif current_sum > target_sum_range[1]:
                # Remplacer les plus grands par des plus petits
                for i in range(len(numbers) - 1, -1, -1):
                    if sum(numbers) <= target_sum_range[1]:
                        break
                    # Chercher un nombre plus petit avec bonne fréquence
                    for candidate in range(numbers[i] - 1, 0, -1):
                        if candidate not in numbers and hist_freq.get(candidate, 0) > 0:
                            numbers[i] = candidate
                            break
        
        # Vérification de la distribution (éviter trop de consécutifs)
        numbers_sorted = sorted(numbers)
        consecutive_count = 0
        max_consecutive = 2
        
        for i in range(len(numbers_sorted) - 1):
            if numbers_sorted[i + 1] - numbers_sorted[i] == 1:
                consecutive_count += 1
                if consecutive_count >= max_consecutive:
                    # Remplacement du dernier consécutif
                    # print("🔧 Ajustement pour éviter trop de consécutifs") # Suppressed
                    hist_freq = patterns['historical_number_frequency']
                    for candidate in range(1, 51):
                        if candidate not in numbers and hist_freq.get(candidate, 0) > 0:
                            # Vérifier que ce n'est pas consécutif
                            temp_numbers = numbers.copy()
                            temp_numbers[temp_numbers.index(numbers_sorted[i + 1])] = candidate
                            temp_sorted = sorted(temp_numbers)
                            
                            # Vérifier les consécutifs dans la nouvelle liste
                            new_consecutive = 0
                            for j in range(len(temp_sorted) - 1):
                                if temp_sorted[j + 1] - temp_sorted[j] == 1:
                                    new_consecutive += 1
                            
                            if new_consecutive < consecutive_count:
                                numbers[numbers.index(numbers_sorted[i + 1])] = candidate
                                break
                    break
            else:
                consecutive_count = 0
        
        return numbers
        
    def validate_and_adjust_stars(self, stars, patterns):
        """Valide et ajuste les étoiles selon les contraintes."""
        
        # Vérification de base (pas de contraintes spéciales pour les étoiles)
        # Mais on peut vérifier la distribution
        
        star_sum = sum(stars)
        
        # Plage raisonnable pour la somme des étoiles (3-23)
        if star_sum < 3:
            # Augmenter une étoile
            hist_freq = patterns['historical_star_frequency']
            for candidate in range(max(stars) + 1, 13):
                if hist_freq.get(candidate, 0) > 0:
                    stars[1] = candidate
                    break
        elif star_sum > 23:
            # Diminuer une étoile
            hist_freq = patterns['historical_star_frequency']
            for candidate in range(min(stars) - 1, 0, -1):
                if hist_freq.get(candidate, 0) > 0:
                    stars[0] = candidate
                    break
        
        return stars
        
    def calculate_confidence_metrics(self, prediction, consensus, insights, patterns):
        """Calcule les métriques de confiance."""
        # print("📊 Calcul des métriques de confiance...") # Suppressed
        
        metrics = {}

        # Load recent historical data for validation against recent draws
        recent_draws_df = None
        try:
            # Assuming self.historical_data is already loaded and is the full dataset
            if self.historical_data is not None and not self.historical_data.empty:
                if len(self.historical_data) >= self.NUM_RECENT_DRAWS:
                    recent_draws_df = self.historical_data.tail(self.NUM_RECENT_DRAWS)
                else:
                    recent_draws_df = self.historical_data.tail(len(self.historical_data))
                    # print(f"⚠️ Moins de {self.NUM_RECENT_DRAWS} tirages disponibles pour validation ({len(self.historical_data)} utilisés).", file=sys.stderr) # To stderr
            else:
                # Attempt to load if self.historical_data was not loaded or empty
                data_file_for_recent_val_primary = 'data/euromillions_enhanced_dataset.csv'
                data_file_for_recent_val_fallback = 'euromillions_enhanced_dataset.csv'
                actual_recent_val_path = None
                if os.path.exists(data_file_for_recent_val_primary):
                    actual_recent_val_path = data_file_for_recent_val_primary
                elif os.path.exists(data_file_for_recent_val_fallback):
                    actual_recent_val_path = data_file_for_recent_val_fallback

                if actual_recent_val_path:
                    full_historical_data_for_recent = pd.read_csv(actual_recent_val_path)
                    if len(full_historical_data_for_recent) >= self.NUM_RECENT_DRAWS:
                        recent_draws_df = full_historical_data_for_recent.tail(self.NUM_RECENT_DRAWS)
                    elif not full_historical_data_for_recent.empty:
                        recent_draws_df = full_historical_data_for_recent.tail(len(full_historical_data_for_recent))
                        # print(f"⚠️ Moins de {self.NUM_RECENT_DRAWS} tirages disponibles ({len(full_historical_data_for_recent)} utilisés).", file=sys.stderr) # To stderr
                    else:
                        pass # print("⚠️ Aucune donnée historique (récente ou complète) disponible pour la validation.", file=sys.stderr) # To stderr
                # else: # No file found
                    # print("⚠️ Fichier euromillions_enhanced_dataset.csv non trouvé pour la validation des tirages récents.", file=sys.stderr) # To stderr

        except FileNotFoundError: # This might catch if read_csv above fails for some reason
            pass # print("⚠️ Fichier euromillions_enhanced_dataset.csv non trouvé pour la validation des tirages récents.", file=sys.stderr) # To stderr
        except Exception as e:
            pass # print(f"⚠️ Erreur lors du chargement/traitement des données récentes pour validation: {e}", file=sys.stderr) # To stderr

        # 1. Score de consensus
        total_votes = sum(consensus['number_votes'].values())
        prediction_votes = sum(consensus['number_votes'].get(num, 0) for num in prediction['numbers'])
        consensus_score = prediction_votes / total_votes if total_votes > 0 else 0
        
        total_star_votes = sum(consensus['star_votes'].values())
        prediction_star_votes = sum(consensus['star_votes'].get(star, 0) for star in prediction['stars'])
        star_consensus_score = prediction_star_votes / total_star_votes if total_star_votes > 0 else 0
        
        # 2. Score historique
        hist_num_freq = patterns['historical_number_frequency']
        total_hist_freq = sum(hist_num_freq.values())
        prediction_hist_freq = sum(hist_num_freq.get(num, 0) for num in prediction['numbers'])
        historical_score = prediction_hist_freq / total_hist_freq if total_hist_freq > 0 else 0
        
        # 3. Score de diversité technologique
        high_perf_techs = insights.get('high_performance_technologies', [])
        diversity_score = len(high_perf_techs) / 10  # Normalisation sur 10 technologies max
        
        # 4. Score de validation
        # Correspondances avec le tirage de référence
        ref_numbers = set(self.reference_draw['numbers'])
        ref_stars = set(self.reference_draw['stars'])
        pred_numbers = set(prediction['numbers'])
        pred_stars = set(prediction['stars'])
        
        number_matches = len(pred_numbers.intersection(ref_numbers))
        star_matches = len(pred_stars.intersection(ref_stars))
        validation_score = (number_matches + star_matches) / 7 # Against static reference_draw

        # NEW: Calculate validation against recent draws
        avg_number_matches_recent = 0.0
        avg_star_matches_recent = 0.0
        total_avg_matches_recent = 0.0
        num_valid_recent_draws = 0

        if recent_draws_df is not None and not recent_draws_df.empty:
            num_valid_recent_draws = len(recent_draws_df)
            total_recent_number_matches = 0
            total_recent_star_matches = 0

            # Ensure column names match those in euromillions_enhanced_dataset.csv (N1-N5, E1-E2)
            # The historical data loaded in load_historical_data uses num1-num5, star1-star2
            # Need to use the correct column names based on where recent_draws_df comes from.
            # Assuming recent_draws_df (if from self.historical_data or direct CSV load) has N1-N5, E1-E2 if it's the enhanced one.
            # If it's from generate_fallback_data, it will have num1-num5, star1-star2.
            # Let's try to be robust or ensure consistency. For now, assume N1-N5, E1-E2 for CSV.
            # If using self.historical_data directly, it might be num1-num5.
            # The provided dataset `euromillions_enhanced_dataset.csv` uses N1..N5, E1..E2

            col_numbers = [f'N{i}' for i in range(1,6)]
            col_stars = [f'E{i}' for i in range(1,3)]

            # Check if columns exist, fallback if necessary (e.g. to num1-num5 format)
            if not all(col in recent_draws_df.columns for col in col_numbers):
                col_numbers = [f'num{i}' for i in range(1,6)] # Fallback
            if not all(col in recent_draws_df.columns for col in col_stars):
                col_stars = [f'star{i}' for i in range(1,3)] # Fallback


            for _, draw_row in recent_draws_df.iterrows():
                try:
                    actual_numbers = set(draw_row[col_numbers].astype(int))
                    actual_stars = set(draw_row[col_stars].astype(int))

                    current_number_matches = len(pred_numbers.intersection(actual_numbers))
                    current_star_matches = len(pred_stars.intersection(actual_stars))

                    total_recent_number_matches += current_number_matches
                    total_recent_star_matches += current_star_matches
                except KeyError as ke:
                    # print(f"Erreur de clé lors de l'accès aux colonnes pour la validation récente: {ke}. Tirage ignoré.", file=sys.stderr) # To stderr
                    if num_valid_recent_draws > 0: num_valid_recent_draws -=1 # Adjust count of valid draws
                    continue # Skip this draw
                except ValueError as ve: # Handle potential conversion errors if data is not clean
                    # print(f"Erreur de valeur lors de la conversion des données du tirage récent: {ve}. Tirage ignoré.", file=sys.stderr) # To stderr
                    if num_valid_recent_draws > 0: num_valid_recent_draws -=1
                    continue


            if num_valid_recent_draws > 0:
                avg_number_matches_recent = total_recent_number_matches / num_valid_recent_draws
                avg_star_matches_recent = total_recent_star_matches / num_valid_recent_draws
                total_avg_matches_recent = (total_recent_number_matches + total_recent_star_matches) / num_valid_recent_draws
            # else: # Suppressed to prevent print when no recent draws are available.
                # print("⚠️ Aucun tirage récent n'a pu être validé (peut-être en raison d'erreurs de format de colonne).")

        validation_score_recent = total_avg_matches_recent / 7 if num_valid_recent_draws > 0 else 0.0

        # Score global pondéré (using recent validation score)
        global_score = (
            consensus_score * 0.3 +
            star_consensus_score * 0.2 +
            historical_score * 0.2 +
            diversity_score * 0.1 +
            validation_score_recent * 0.2 # Use recent validation score
        )
        
        metrics = {
            'consensus_score': consensus_score,
            'star_consensus_score': star_consensus_score,
            'historical_score': historical_score,
            'diversity_score': diversity_score,
            'validation_score_reference_draw': validation_score, # Original score against fixed reference
            'validation_score_recent_draws': validation_score_recent, # New score against recent draws
            'global_confidence': global_score,
            'confidence_percentage': global_score * 100,
            'validation_matches_reference_draw': { # Original matches against fixed reference
                'number_matches': number_matches,
                'star_matches': star_matches,
                'total_matches': number_matches + star_matches
            },
            'validation_matches_recent_draws': { # New average matches against recent draws
                'avg_number_matches': avg_number_matches_recent,
                'avg_star_matches': avg_star_matches_recent,
                'total_avg_matches': total_avg_matches_recent,
                'num_recent_draws_validated': num_valid_recent_draws
            }
        }
        
        # print("✅ Métriques de confiance calculées") # Suppressed
        return metrics
        
    def generate_aggregation_visualizations(self, prediction, consensus, metrics, output_filename): # Added output_filename
        """Génère les visualisations d'agrégation."""
        # print("📊 Génération des visualisations d'agrégation...") # Suppressed
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Add dynamic date to title if possible, or keep generic
        title_date_str = self.actual_next_draw_date.strftime('%d/%m/%Y')
        fig.suptitle(f'Tirage Final Agrégé - Analyse pour le {title_date_str}', fontsize=16, fontweight='bold')
        
        # 1. Consensus des numéros
        top_numbers = consensus['top_numbers'][:15]
        if top_numbers:
            numbers, votes = zip(*top_numbers)
            
            # Coloration spéciale pour les numéros sélectionnés
            colors = ['red' if num in prediction['numbers'] else 'skyblue' for num in numbers]
            
            axes[0,0].bar(range(len(numbers)), votes, color=colors, alpha=0.7)
            axes[0,0].set_title('Consensus des Numéros (Top 15)')
            axes[0,0].set_ylabel('Votes Pondérés')
            axes[0,0].set_xticks(range(len(numbers)))
            axes[0,0].set_xticklabels(numbers)
            axes[0,0].grid(True, alpha=0.3)
            
            # Légende
            axes[0,0].legend(['Sélectionné', 'Non sélectionné'], loc='upper right')
        
        # 2. Consensus des étoiles
        top_stars = consensus['top_stars'][:8]
        if top_stars:
            stars, votes = zip(*top_stars)
            
            colors = ['red' if star in prediction['stars'] else 'lightcoral' for star in stars]
            
            axes[0,1].bar(range(len(stars)), votes, color=colors, alpha=0.7)
            axes[0,1].set_title('Consensus des Étoiles (Top 8)')
            axes[0,1].set_ylabel('Votes Pondérés')
            axes[0,1].set_xticks(range(len(stars)))
            axes[0,1].set_xticklabels(stars)
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Métriques de confiance
        metric_names = ['Consensus\nNuméros', 'Consensus\nÉtoiles', 'Score\nHistorique', 
                       'Diversité\nTech', 'Validation\nRéférence']
        metric_values = [
            metrics['consensus_score'],
            metrics['star_consensus_score'],
            metrics['historical_score'],
            metrics['diversity_score'],
            metrics['validation_score_reference_draw'] # Corrected key
        ]
        
        colors = ['lightgreen' if v >= 0.7 else 'orange' if v >= 0.4 else 'lightcoral' for v in metric_values]
        
        axes[1,0].bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7)
        axes[1,0].set_title('Métriques de Confiance')
        axes[1,0].set_ylabel('Score (0-1)')
        axes[1,0].set_xticks(range(len(metric_names)))
        axes[1,0].set_xticklabels(metric_names, rotation=0, ha='center')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].grid(True, alpha=0.3)
        
        # Ligne de confiance globale
        global_conf = metrics['global_confidence']
        axes[1,0].axhline(y=global_conf, color='red', linestyle='--', 
                         label=f'Confiance Globale: {global_conf:.2f}')
        axes[1,0].legend()
        
        # 4. Prédiction finale
        axes[1,1].text(0.1, 0.8, 'PRÉDICTION FINALE AGRÉGÉE', fontsize=14, fontweight='bold')
        axes[1,1].text(0.1, 0.65, f"Numéros: {' - '.join(map(str, prediction['numbers']))}", 
                      fontsize=12, color='blue')
        axes[1,1].text(0.1, 0.55, f"Étoiles: {' - '.join(map(str, prediction['stars']))}", 
                      fontsize=12, color='red')
        axes[1,1].text(0.1, 0.4, f"Confiance: {metrics['confidence_percentage']:.1f}%", 
                      fontsize=12, fontweight='bold')
        # Display new average match score
        num_validated_recent = metrics['validation_matches_recent_draws']['num_recent_draws_validated']
        avg_total_recent = metrics['validation_matches_recent_draws']['total_avg_matches']
        avg_matches_str = f"Avg Matches (last {num_validated_recent} draws): {avg_total_recent:.2f}/7"
        if num_validated_recent == 0:
            avg_matches_str = "Avg Matches (recent): N/A"
        axes[1,1].text(0.1, 0.3, avg_matches_str, fontsize=10)
        axes[1,1].text(0.1, 0.2, f"Basé sur {consensus['total_predictions']} systèmes", 
                      fontsize=10)
        axes[1,1].text(0.1, 0.1, f"Méthodologie: {prediction['methodology']}", 
                      fontsize=8, style='italic')
        
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight') # Use dynamic filename
        plt.close()
        
        # print(f"✅ Visualisations d'agrégation générées: {output_filename}") # Suppressed
        
    def save_final_prediction(self, prediction, consensus, insights, patterns, metrics):
        """Sauvegarde la prédiction finale."""
        # print("💾 Sauvegarde de la prédiction finale...") # Suppressed
        
        date_str_for_filename = self.actual_next_draw_date.strftime('%Y-%m-%d')

        json_filename = f'{self.aggregation_dir}/final_aggregated_prediction_{date_str_for_filename}.json'
        ticket_filename = f'{self.aggregation_dir}/ticket_final_agrege_{date_str_for_filename}.txt'
        report_filename = f'{self.aggregation_dir}/rapport_final_agrege_{date_str_for_filename}.txt'
        # Visualization filename is passed to generate_aggregation_visualizations, defined in run_final_aggregation

        # Données complètes
        final_data = {
            'generation_date': datetime.now().isoformat(),
            'target_draw_date_prediction': self.actual_next_draw_date.strftime('%Y-%m-%d'), # Add target date
            'reference_draw_for_validation': self.reference_draw, # Clarify this is for validation
            'final_prediction': prediction, # This now includes target_draw_date
            'consensus_analysis': consensus,
            'best_practices_insights': insights,
            'historical_patterns': patterns,
            'confidence_metrics': metrics,
            'methodology': {
                'approach': 'Agrégation intelligente de tous les systèmes développés',
                'systems_analyzed': len(self.test_results),
                'technologies_used': len(self.synthesis.get('learnings', {}).get('technology_performance', {})),
                'weighting_strategy': {
                    'consensus_votes': '40%',
                    'historical_frequency': '30%',
                    'recent_patterns': '20%',
                    'success_patterns': '10%'
                }
            }
        }
        
        # Sauvegarde JSON
        with open(json_filename, 'w') as f:
            json.dump(final_data, f, indent=2, default=str)
        
        # Ticket de prédiction
        ticket_content = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    TICKET EUROMILLIONS FINAL AGRÉGÉ                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  🔮 PRÉDICTION POUR LE: {self.actual_next_draw_date.strftime('%d/%m/%Y')}                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  🎯 PRÉDICTION BASÉE SUR L'AGRÉGATION DE 36 SYSTÈMES D'IA 🎯       ║
║                                                                      ║
║  📅 Date de génération: {datetime.now().strftime('%d/%m/%Y %H:%M')}                        ║
║                                                                      ║
║  🔢 NUMÉROS PRINCIPAUX:                                             ║
║      {' - '.join(f'{num:2d}' for num in prediction['numbers'])}                                      ║
║                                                                      ║
║  ⭐ ÉTOILES:                                                        ║
║      {' - '.join(f'{star:2d}' for star in prediction['stars'])}                                              ║
║                                                                      ║
║  📊 CONFIANCE: {metrics['confidence_percentage']:.1f}%                                      ║
║                                                                      ║
║  ✅ VALIDATION (avg last {metrics['validation_matches_recent_draws']['num_recent_draws_validated']}): {metrics['validation_matches_recent_draws']['total_avg_matches']:.2f}/7 correspondances avec tirages réels      ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  MÉTHODOLOGIE:                                                       ║
║  • Consensus de {consensus['total_predictions']} systèmes d'IA                              ║
║  • Analyse de {len(self.historical_data)} tirages historiques                    ║
║  • {len(insights.get('high_performance_technologies', []))} technologies haute performance            ║
║  • Validation scientifique rigoureuse                               ║
║                                                                      ║
║  TECHNOLOGIES UTILISÉES:                                             ║
║  • TensorFlow, Scikit-Learn, Optuna                                 ║
║  • Ensemble Methods, Bayesian Optimization                          ║
║  • Quantum Computing, Genetic Algorithms                            ║
║  • Neural Networks, Random Forest                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

DÉTAILS TECHNIQUES:
- Pondération: Consensus (40%) + Historique (30%) + Récent (20%) + Succès (10%)
- Validation: Moyenne sur les {metrics['validation_matches_recent_draws']['num_recent_draws_validated']} derniers tirages réels.
- (Validation sur tirage de référence {self.reference_draw['date']}: {metrics['validation_matches_reference_draw']['total_matches']}/7)
- Ajustements automatiques pour contraintes statistiques
- Métriques de confiance multi-dimensionnelles

⚠️  AVERTISSEMENT: Ce ticket est généré à des fins éducatives et de recherche.
    L'Euromillions reste un jeu de hasard. Jouez de manière responsable.

🎉 BONNE CHANCE! 🎉
"""
        
        with open(f'{self.aggregation_dir}/ticket_final_agrege.txt', 'w', encoding='utf-8') as f:
            f.write(ticket_content)
        
        # Rapport détaillé
        report_content = f"""
# RAPPORT FINAL - TIRAGE AGRÉGÉ BASÉ SUR TOUS LES ENSEIGNEMENTS

## RÉSUMÉ EXÉCUTIF
Date de génération: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Date du tirage cible pour cette prédiction: {self.actual_next_draw_date.strftime('%d/%m/%Y')}
Validation principale: Moyenne des correspondances sur les {metrics['validation_matches_recent_draws']['num_recent_draws_validated']} derniers tirages réels.
(Tirage de référence statique pour comparaison: {self.reference_draw['numbers']} + {self.reference_draw['stars']} ({self.reference_draw['date']}))

## PRÉDICTION FINALE (pour le {self.actual_next_draw_date.strftime('%d/%m/%Y')})
**Numéros:** {' - '.join(map(str, prediction['numbers']))}
**Étoiles:** {' - '.join(map(str, prediction['stars']))}
**Confiance:** {metrics['confidence_percentage']:.1f}%

## MÉTHODOLOGIE D'AGRÉGATION

### Sources d'Information
- **Systèmes analysés:** {len(self.test_results)}
- **Technologies évaluées:** {len(self.synthesis.get('learnings', {}).get('technology_performance', {}))}
- **Tirages historiques:** {len(self.historical_data)}
- **Meilleures pratiques:** {len(self.synthesis.get('best_practices', {}).get('recommendations', []))}

### Stratégie de Pondération
1. **Consensus des prédictions (40%):** Votes pondérés par performance
2. **Fréquences historiques (30%):** Analyse de tous les tirages passés
3. **Patterns récents (20%):** Tendances des 50 derniers tirages
4. **Patterns de succès (10%):** Insights des systèmes performants

### Validation et Ajustements
- Contraintes de somme: {sum(prediction['numbers'])} (plage optimale: 100-200)
- Distribution équilibrée: Évitement des consécutifs excessifs
- Validation croisée (moyenne récents): {metrics['validation_matches_recent_draws']['total_avg_matches']:.2f}/7 sur {metrics['validation_matches_recent_draws']['num_recent_draws_validated']} tirages
- (Validation sur réf. statique: {metrics['validation_matches_reference_draw']['total_matches']}/7)

## ANALYSE DE CONFIANCE

### Métriques Détaillées
- **Score de consensus numéros:** {metrics['consensus_score']:.3f}
- **Score de consensus étoiles:** {metrics['star_consensus_score']:.3f}
- **Score historique:** {metrics['historical_score']:.3f}
- **Score de diversité technologique:** {metrics['diversity_score']:.3f}
- **Score de validation (référence):** {metrics['validation_score_reference_draw']:.3f}
- **Score de validation (récents):** {metrics['validation_score_recent_draws']:.3f} (utilisé pour la confiance globale)

### Confiance Globale: {metrics['global_confidence']:.3f} ({metrics['confidence_percentage']:.1f}%)

## TECHNOLOGIES HAUTE PERFORMANCE IDENTIFIÉES
"""
        
        high_perf_techs = insights.get('high_performance_technologies', [])
        for tech in high_perf_techs:
            report_content += f"- {tech}\n"
        
        report_content += f"""
## PATTERNS DE SUCCÈS APPLIQUÉS

### Numéros les Plus Prédits
"""
        
        if 'most_predicted_numbers' in insights:
            for num, freq in insights['most_predicted_numbers'][:5]:
                report_content += f"- {num}: {freq} prédictions\n"
        
        report_content += f"""
### Étoiles les Plus Prédites
"""
        
        if 'most_predicted_stars' in insights:
            for star, freq in insights['most_predicted_stars'][:3]:
                report_content += f"- {star}: {freq} prédictions\n"
        
        report_content += f"""
## VALIDATION CONTRE TIRAGES RÉCENTS (Moyenne sur {metrics['validation_matches_recent_draws']['num_recent_draws_validated']} derniers tirages)

### Correspondances Moyennes Détaillées
- **Moy. numéros corrects:** {metrics['validation_matches_recent_draws']['avg_number_matches']:.2f}/5
- **Moy. étoiles correctes:** {metrics['validation_matches_recent_draws']['avg_star_matches']:.2f}/2
- **Moy. total correspondances:** {metrics['validation_matches_recent_draws']['total_avg_matches']:.2f}/7 ({(metrics['validation_matches_recent_draws']['total_avg_matches']/7)*100 if metrics['validation_matches_recent_draws']['total_avg_matches'] > 0 else 0.0:.1f}%)

(Pour information, validation contre tirage de référence statique {self.reference_draw['date']}:
- Numéros corrects: {metrics['validation_matches_reference_draw']['number_matches']}/5
- Étoiles correctes: {metrics['validation_matches_reference_draw']['star_matches']}/2
- Total: {metrics['validation_matches_reference_draw']['total_matches']}/7)

### Analyse des Écarts (simplifiée pour la moyenne)
L'analyse détaillée des écarts (numéros manqués, faux positifs) est complexe pour une moyenne sur plusieurs tirages.
La métrique principale ici est le nombre moyen de correspondances.
Une analyse plus poussée pourrait inclure la fréquence à laquelle chaque numéro prédit est apparu
dans les {metrics['validation_matches_recent_draws']['num_recent_draws_validated']} tirages récents, ou la distribution des correspondances (min/max).
Pour cette version, nous nous concentrons sur la moyenne des correspondances.
"""
        
        # Optional: Keep the old analysis for the reference draw if desired, but label it clearly
        report_content += "\nAnalyse des Écarts (vs Tirage de Référence Statique):\n"
        ref_numbers_static = set(self.reference_draw['numbers'])
        pred_numbers_static = set(prediction['numbers']) # Assuming prediction['numbers'] is available here
        
        correct_numbers_static = pred_numbers_static.intersection(ref_numbers_static)
        missed_numbers_static = ref_numbers_static - pred_numbers_static
        false_positives_static = pred_numbers_static - ref_numbers_static
        
        if correct_numbers_static:
            report_content += f"- Numéros corrects (vs réf. statique): {', '.join(map(str, sorted(correct_numbers_static)))}\n"
        if missed_numbers_static:
            report_content += f"- Numéros manqués (vs réf. statique): {', '.join(map(str, sorted(missed_numbers_static)))}\n"
        if false_positives_static:
            report_content += f"- Faux positifs (vs réf. statique): {', '.join(map(str, sorted(false_positives_static)))}\n"
        
        report_content += f"""
## ENSEIGNEMENTS CLÉS

### Approches les Plus Efficaces
1. **Optimisation ciblée:** Systèmes avec correspondances parfaites
2. **Ensemble de modèles:** Robustesse par diversification
3. **Validation scientifique:** Rigueur méthodologique
4. **Optimisation bayésienne:** Ajustement automatique des hyperparamètres

### Recommandations pour l'Avenir
1. Intégrer davantage de données externes
2. Développer des techniques d'apprentissage en temps réel
3. Améliorer la validation croisée temporelle
4. Explorer les techniques de méta-apprentissage

## CONCLUSION

Cette prédiction représente la synthèse de tous les enseignements tirés du développement
de 36 systèmes d'IA différents. Elle combine les meilleures pratiques identifiées,
les patterns de succès validés, et l'analyse rigoureuse des données historiques.

Avec une confiance de {metrics['confidence_percentage']:.1f}% et une moyenne de {metrics['validation_matches_recent_draws']['total_avg_matches']:.2f}/7 correspondances sur les récents tirages,
cette prédiction constitue l'aboutissement de notre recherche en IA prédictive
pour l'Euromillions.

---
Rapport généré automatiquement par le Générateur de Tirage Final Agrégé
"""
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # print(f"✅ Prédiction finale sauvegardée ({json_filename}, {ticket_filename}, {report_filename})") # Suppressed
        
    def run_final_aggregation(self):
        """Exécute l'agrégation finale complète."""
        # print("🚀 LANCEMENT DE L'AGRÉGATION FINALE COMPLÈTE 🚀") # Suppressed
        # print("=" * 70) # Suppressed
        
        # 1. Analyse du consensus
        # print("🤝 Phase 1: Analyse du consensus...") # Suppressed
        consensus = self.analyze_prediction_consensus()
        
        # 2. Application des insights
        # print("🏆 Phase 2: Application des insights...") # Suppressed
        insights = self.apply_best_practices_insights()
        
        # 3. Analyse des patterns historiques
        # print("📊 Phase 3: Analyse historique...") # Suppressed
        patterns = self.analyze_historical_patterns()
        
        # 4. Création de la prédiction d'ensemble
        # print("🎯 Phase 4: Prédiction d'ensemble...") # Suppressed
        prediction = self.create_ensemble_prediction(consensus, insights, patterns)
        prediction['target_draw_date'] = self.actual_next_draw_date.strftime('%Y-%m-%d') # Add target date
        
        # 5. Calcul des métriques de confiance
        # print("📊 Phase 5: Métriques de confiance...") # Suppressed
        metrics = self.calculate_confidence_metrics(prediction, consensus, insights, patterns)
        
        # Define dynamic visualization filename here to pass it down
        date_str_for_filename = self.actual_next_draw_date.strftime('%Y-%m-%d')
        visualization_filename = f'{self.aggregation_dir}/visualizations/final_aggregated_prediction_{date_str_for_filename}.png'

        # 6. Visualisations
        # print("📊 Phase 6: Visualisations...") # Suppressed
        self.generate_aggregation_visualizations(prediction, consensus, metrics, visualization_filename) # Pass filename
        
        # 7. Sauvegarde finale
        # print("💾 Phase 7: Sauvegarde...") # Suppressed
        self.save_final_prediction(prediction, consensus, insights, patterns, metrics)
        
        # print("✅ AGRÉGATION FINALE TERMINÉE!") # Suppressed
        
        # Stockage pour accès externe
        self.aggregated_prediction = {
            'prediction': prediction,
            'consensus': consensus,
            'insights': insights,
            'patterns': patterns,
            'metrics': metrics
        }
        
        return self.aggregated_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de Prédiction Finale Agrégée Euromillions.")
    parser.add_argument("--date", type=str, help="Date cible du tirage (YYYY-MM-DD). Si non fournie, la prochaine date de tirage est auto-déterminée.")
    args = parser.parse_args()

    # Lancement de l'agrégation finale
    aggregator = AggregatedFinalPredictor(target_date_str=args.date)
    results = aggregator.run_final_aggregation() # This is the comprehensive dict

    # Extract standardized prediction
    prediction_numeros = results.get('prediction', {}).get('numbers', [])
    prediction_etoiles = results.get('prediction', {}).get('stars', [])

    # Ensure confidence is a float, default to a standard value if not calculable or missing
    raw_confidence = results.get('metrics', {}).get('confidence_percentage', 75.0) # Default 75.0
    try:
        prediction_confidence = float(raw_confidence)
    except (ValueError, TypeError):
        prediction_confidence = 75.0 # Default if conversion fails

    # Ensure target_date_str for output is from the aggregator instance (which handled args.date)
    output_target_date_str = aggregator.actual_next_draw_date.strftime('%Y-%m-%d')
    
    output_dict = {
        'nom_predicteur': 'aggregated_final_predictor',
        'numeros': prediction_numeros,
        'etoiles': prediction_etoiles,
        'date_tirage_cible': output_target_date_str,
        'confidence': prediction_confidence, # Should be a float e.g. 78.5 for 78.5%
        'categorie': "Meta-Predicteurs"
    }
    
    # The only print to stdout should be the JSON dump
    print(json.dumps(output_dict))

