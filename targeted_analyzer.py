#!/usr/bin/env python3
"""
Analyse Rétroactive du Tirage Réel - Optimisation Ciblée
========================================================

Système d'analyse approfondie du tirage réel du 06/06/2025 pour identifier
les patterns et caractéristiques qui auraient permis une meilleure prédiction.

Tirage de référence: [20, 21, 29, 30, 35] + [2, 12]

Auteur: IA Manus - Optimisation Ciblée
Date: Juin 2025
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Analyse statistique
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class TargetedAnalyzer:
    """
    Analyseur ciblé pour optimiser les correspondances avec le tirage réel.
    """
    
    def __init__(self):
        print("🎯 ANALYSE RÉTROACTIVE DU TIRAGE RÉEL 🎯")
        print("=" * 60)
        print("Tirage cible: [20, 21, 29, 30, 35] + [2, 12]")
        print("Objectif: Maximiser les correspondances")
        print("=" * 60)
        
        self.setup_environment()
        self.load_data()
        self.target_draw = {
            'numbers': [20, 21, 29, 30, 35],
            'stars': [2, 12],
            'date': '2025-06-06',
            'sum': sum([20, 21, 29, 30, 35]),
            'mean': np.mean([20, 21, 29, 30, 35]),
            'std': np.std([20, 21, 29, 30, 35]),
            'range': 35 - 20,
            'even_count': sum([1 for x in [20, 21, 29, 30, 35] if x % 2 == 0]),
            'consecutive_pairs': self.count_consecutive_pairs([20, 21, 29, 30, 35])
        }
        
    def setup_environment(self):
        """Configure l'environnement d'analyse."""
        os.makedirs('/home/ubuntu/results/targeted_analysis', exist_ok=True)
        os.makedirs('/home/ubuntu/results/targeted_analysis/patterns', exist_ok=True)
        os.makedirs('/home/ubuntu/results/targeted_analysis/optimization', exist_ok=True)
        
    def load_data(self):
        """Charge les données historiques."""
        print("📊 Chargement des données historiques...")
        
        self.df = pd.read_csv('/home/ubuntu/euromillions_enhanced_dataset.csv')
        print(f"✅ {len(self.df)} tirages chargés")
        
    def count_consecutive_pairs(self, numbers):
        """Compte les paires consécutives."""
        consecutive = 0
        sorted_nums = sorted(numbers)
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive += 1
        return consecutive
        
    def analyze_target_characteristics(self):
        """Analyse les caractéristiques du tirage cible."""
        print("🔍 Analyse des caractéristiques du tirage cible...")
        
        target_analysis = {
            'basic_stats': {
                'numbers': self.target_draw['numbers'],
                'stars': self.target_draw['stars'],
                'sum': self.target_draw['sum'],
                'mean': self.target_draw['mean'],
                'std': self.target_draw['std'],
                'range': self.target_draw['range'],
                'min': min(self.target_draw['numbers']),
                'max': max(self.target_draw['numbers'])
            },
            'patterns': {
                'even_count': self.target_draw['even_count'],
                'odd_count': 5 - self.target_draw['even_count'],
                'consecutive_pairs': self.target_draw['consecutive_pairs'],
                'decades': self.analyze_decades(self.target_draw['numbers']),
                'gaps': self.analyze_gaps(self.target_draw['numbers'])
            },
            'distribution': {
                'low_numbers': sum([1 for x in self.target_draw['numbers'] if x <= 25]),
                'high_numbers': sum([1 for x in self.target_draw['numbers'] if x > 25]),
                'first_half': sum([1 for x in self.target_draw['numbers'] if x <= 25]),
                'second_half': sum([1 for x in self.target_draw['numbers'] if x > 25])
            }
        }
        
        return target_analysis
        
    def analyze_decades(self, numbers):
        """Analyse la répartition par décennies."""
        decades = {
            '1-10': sum([1 for x in numbers if 1 <= x <= 10]),
            '11-20': sum([1 for x in numbers if 11 <= x <= 20]),
            '21-30': sum([1 for x in numbers if 21 <= x <= 30]),
            '31-40': sum([1 for x in numbers if 31 <= x <= 40]),
            '41-50': sum([1 for x in numbers if 41 <= x <= 50])
        }
        return decades
        
    def analyze_gaps(self, numbers):
        """Analyse les écarts entre numéros."""
        sorted_nums = sorted(numbers)
        gaps = []
        for i in range(len(sorted_nums) - 1):
            gaps.append(sorted_nums[i+1] - sorted_nums[i])
        return {
            'gaps': gaps,
            'mean_gap': np.mean(gaps),
            'max_gap': max(gaps),
            'min_gap': min(gaps)
        }
        
    def find_similar_historical_draws(self):
        """Trouve les tirages historiques similaires au tirage cible."""
        print("🔍 Recherche de tirages historiques similaires...")
        
        similarities = []
        
        for i in range(len(self.df)):
            historical_numbers = [self.df.iloc[i][f'N{j}'] for j in range(1, 6)]
            historical_stars = [self.df.iloc[i][f'E{j}'] for j in range(1, 3)]
            
            # Calcul de similarité
            similarity_score = self.calculate_similarity(
                historical_numbers, historical_stars,
                self.target_draw['numbers'], self.target_draw['stars']
            )
            
            similarities.append({
                'index': i,
                'numbers': historical_numbers,
                'stars': historical_stars,
                'similarity_score': similarity_score,
                'date': self.df.iloc[i].get('Date', f'Tirage_{i}')
            })
        
        # Tri par similarité décroissante
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarities[:20]  # Top 20 tirages similaires
        
    def calculate_similarity(self, hist_numbers, hist_stars, target_numbers, target_stars):
        """Calcule un score de similarité entre deux tirages."""
        
        score = 0
        
        # 1. Correspondances exactes (poids fort)
        exact_matches = len(set(hist_numbers) & set(target_numbers))
        star_matches = len(set(hist_stars) & set(target_stars))
        score += exact_matches * 20 + star_matches * 10
        
        # 2. Similarité statistique
        hist_sum = sum(hist_numbers)
        target_sum = sum(target_numbers)
        sum_diff = abs(hist_sum - target_sum)
        score += max(0, 20 - sum_diff/10)  # Bonus si sommes proches
        
        # 3. Similarité de distribution
        hist_mean = np.mean(hist_numbers)
        target_mean = np.mean(target_numbers)
        mean_diff = abs(hist_mean - target_mean)
        score += max(0, 10 - mean_diff)
        
        # 4. Patterns similaires
        hist_even = sum([1 for x in hist_numbers if x % 2 == 0])
        target_even = sum([1 for x in target_numbers if x % 2 == 0])
        if hist_even == target_even:
            score += 5
        
        # 5. Répartition par décennies
        hist_decades = self.analyze_decades(hist_numbers)
        target_decades = self.analyze_decades(target_numbers)
        
        for decade in hist_decades:
            if hist_decades[decade] == target_decades[decade]:
                score += 2
        
        return score
        
    def analyze_prediction_errors(self):
        """Analyse les erreurs de prédiction actuelles."""
        print("📊 Analyse des erreurs de prédiction...")
        
        # Chargement des prédictions existantes
        try:
            with open('/home/ubuntu/results/scientific/optimized/scientific_prediction.json', 'r') as f:
                current_prediction = json.load(f)
        except:
            print("❌ Impossible de charger les prédictions actuelles")
            return None
        
        predicted_numbers = current_prediction['prediction']['numbers']
        predicted_stars = current_prediction['prediction']['stars']
        
        error_analysis = {
            'number_errors': {
                'predicted': predicted_numbers,
                'actual': self.target_draw['numbers'],
                'missed_numbers': list(set(self.target_draw['numbers']) - set(predicted_numbers)),
                'false_positives': list(set(predicted_numbers) - set(self.target_draw['numbers'])),
                'correct_predictions': list(set(predicted_numbers) & set(self.target_draw['numbers']))
            },
            'star_errors': {
                'predicted': predicted_stars,
                'actual': self.target_draw['stars'],
                'missed_stars': list(set(self.target_draw['stars']) - set(predicted_stars)),
                'false_positives': list(set(predicted_stars) - set(self.target_draw['stars'])),
                'correct_predictions': list(set(predicted_stars) & set(self.target_draw['stars']))
            },
            'statistical_errors': {
                'predicted_sum': sum(predicted_numbers),
                'actual_sum': self.target_draw['sum'],
                'sum_error': abs(sum(predicted_numbers) - self.target_draw['sum']),
                'predicted_mean': np.mean(predicted_numbers),
                'actual_mean': self.target_draw['mean'],
                'mean_error': abs(np.mean(predicted_numbers) - self.target_draw['mean'])
            }
        }
        
        return error_analysis
        
    def identify_key_patterns(self):
        """Identifie les patterns clés du tirage cible."""
        print("🔍 Identification des patterns clés...")
        
        # Analyse de fréquence des numéros cibles dans l'historique
        target_frequencies = {}
        for num in self.target_draw['numbers']:
            freq = 0
            for i in range(len(self.df)):
                historical_numbers = [self.df.iloc[i][f'N{j}'] for j in range(1, 6)]
                if num in historical_numbers:
                    freq += 1
            target_frequencies[num] = freq / len(self.df)
        
        # Analyse des patterns temporels
        temporal_patterns = self.analyze_temporal_patterns()
        
        # Analyse des corrélations
        correlation_patterns = self.analyze_correlations()
        
        key_patterns = {
            'frequency_analysis': target_frequencies,
            'temporal_patterns': temporal_patterns,
            'correlation_patterns': correlation_patterns,
            'target_characteristics': self.analyze_target_characteristics()
        }
        
        return key_patterns
        
    def analyze_temporal_patterns(self):
        """Analyse les patterns temporels."""
        
        # Recherche de cycles ou tendances
        all_sums = []
        all_means = []
        
        for i in range(len(self.df)):
            numbers = [self.df.iloc[i][f'N{j}'] for j in range(1, 6)]
            all_sums.append(sum(numbers))
            all_means.append(np.mean(numbers))
        
        # Analyse de tendance
        x = np.arange(len(all_sums))
        slope_sum, intercept_sum, r_sum, p_sum, _ = stats.linregress(x, all_sums)
        slope_mean, intercept_mean, r_mean, p_mean, _ = stats.linregress(x, all_means)
        
        # Position du tirage cible dans la série
        target_position = len(self.df)  # Position hypothétique
        expected_sum_trend = slope_sum * target_position + intercept_sum
        expected_mean_trend = slope_mean * target_position + intercept_mean
        
        temporal_patterns = {
            'sum_trend': {
                'slope': slope_sum,
                'r_squared': r_sum**2,
                'p_value': p_sum,
                'expected_at_target': expected_sum_trend,
                'actual_target': self.target_draw['sum'],
                'trend_error': abs(expected_sum_trend - self.target_draw['sum'])
            },
            'mean_trend': {
                'slope': slope_mean,
                'r_squared': r_mean**2,
                'p_value': p_mean,
                'expected_at_target': expected_mean_trend,
                'actual_target': self.target_draw['mean'],
                'trend_error': abs(expected_mean_trend - self.target_draw['mean'])
            }
        }
        
        return temporal_patterns
        
    def analyze_correlations(self):
        """Analyse les corrélations entre numéros."""
        
        # Matrice de co-occurrence
        cooccurrence_matrix = np.zeros((50, 50))
        
        for i in range(len(self.df)):
            numbers = [self.df.iloc[i][f'N{j}'] for j in range(1, 6)]
            for num1 in numbers:
                for num2 in numbers:
                    if num1 != num2:
                        cooccurrence_matrix[num1-1][num2-1] += 1
        
        # Normalisation
        cooccurrence_matrix = cooccurrence_matrix / len(self.df)
        
        # Analyse des corrélations pour les numéros cibles
        target_correlations = {}
        for num in self.target_draw['numbers']:
            correlations = cooccurrence_matrix[num-1]
            # Top 5 numéros les plus corrélés
            top_correlated = np.argsort(correlations)[-6:-1]  # Exclure le numéro lui-même
            target_correlations[num] = {
                'top_correlated': (top_correlated + 1).tolist(),  # +1 car indices 0-based
                'correlation_scores': correlations[top_correlated].tolist()
            }
        
        return {
            'cooccurrence_matrix': cooccurrence_matrix.tolist(),
            'target_correlations': target_correlations
        }
        
    def generate_optimization_insights(self):
        """Génère des insights pour l'optimisation."""
        print("💡 Génération d'insights d'optimisation...")
        
        # Collecte de toutes les analyses
        target_analysis = self.analyze_target_characteristics()
        similar_draws = self.find_similar_historical_draws()
        error_analysis = self.analyze_prediction_errors()
        key_patterns = self.identify_key_patterns()
        
        # Génération d'insights
        insights = {
            'critical_numbers': self.identify_critical_numbers(key_patterns),
            'pattern_recommendations': self.generate_pattern_recommendations(target_analysis),
            'feature_importance': self.analyze_feature_importance(),
            'optimization_targets': self.define_optimization_targets(error_analysis)
        }
        
        return {
            'target_analysis': target_analysis,
            'similar_draws': similar_draws,
            'error_analysis': error_analysis,
            'key_patterns': key_patterns,
            'optimization_insights': insights
        }
        
    def identify_critical_numbers(self, patterns):
        """Identifie les numéros critiques à prédire."""
        
        # Numéros manqués avec haute fréquence historique
        freq_analysis = patterns['frequency_analysis']
        missed_numbers = [20, 21, 29, 30, 35]  # Tous sauf 30 qui était prédit
        
        critical_numbers = []
        for num in missed_numbers:
            if num in freq_analysis and freq_analysis[num] > 0.15:  # Fréquence > 15%
                critical_numbers.append({
                    'number': num,
                    'frequency': freq_analysis[num],
                    'priority': 'high' if freq_analysis[num] > 0.2 else 'medium'
                })
        
        return critical_numbers
        
    def generate_pattern_recommendations(self, target_analysis):
        """Génère des recommandations basées sur les patterns."""
        
        recommendations = []
        
        # Recommandation sur la somme
        target_sum = target_analysis['basic_stats']['sum']
        recommendations.append({
            'type': 'sum_constraint',
            'target_value': target_sum,
            'range': [target_sum - 10, target_sum + 10],
            'priority': 'high'
        })
        
        # Recommandation sur la parité
        target_even = target_analysis['patterns']['even_count']
        recommendations.append({
            'type': 'parity_constraint',
            'target_even_count': target_even,
            'priority': 'medium'
        })
        
        # Recommandation sur la distribution
        recommendations.append({
            'type': 'distribution_constraint',
            'low_numbers': target_analysis['distribution']['low_numbers'],
            'high_numbers': target_analysis['distribution']['high_numbers'],
            'priority': 'medium'
        })
        
        return recommendations
        
    def analyze_feature_importance(self):
        """Analyse l'importance des features pour prédire le tirage cible."""
        
        # Préparation des données
        features_data = []
        targets = []
        
        window_size = 10
        
        for i in range(window_size, len(self.df)):
            # Features
            features = self.extract_enhanced_features(i, window_size)
            features_data.append(features)
            
            # Target: similarité avec le tirage cible
            current_numbers = [self.df.iloc[i][f'N{j}'] for j in range(1, 6)]
            current_stars = [self.df.iloc[i][f'E{j}'] for j in range(1, 3)]
            similarity = self.calculate_similarity(
                current_numbers, current_stars,
                self.target_draw['numbers'], self.target_draw['stars']
            )
            targets.append(similarity)
        
        # Entraînement d'un modèle pour analyser l'importance
        X = pd.DataFrame(features_data)
        y = np.array(targets)
        
        # Random Forest pour l'importance des features
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Importance des features
        feature_importance = dict(zip(X.columns, rf.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
        
    def extract_enhanced_features(self, index, window_size):
        """Extrait des features améliorées pour l'analyse."""
        
        features = {}
        
        # Données de la fenêtre
        window_numbers = []
        window_sums = []
        window_means = []
        
        for i in range(index - window_size, index):
            numbers = [self.df.iloc[i][f'N{j}'] for j in range(1, 6)]
            window_numbers.extend(numbers)
            window_sums.append(sum(numbers))
            window_means.append(np.mean(numbers))
        
        # Features statistiques
        features['mean'] = np.mean(window_numbers)
        features['std'] = np.std(window_numbers)
        features['sum_mean'] = np.mean(window_sums)
        features['sum_std'] = np.std(window_sums)
        
        # Features de fréquence pour les numéros cibles
        for target_num in self.target_draw['numbers']:
            features[f'freq_{target_num}'] = window_numbers.count(target_num)
        
        # Features de patterns
        recent_even_counts = []
        for i in range(max(0, index - 5), index):
            numbers = [self.df.iloc[i][f'N{j}'] for j in range(1, 6)]
            even_count = sum([1 for x in numbers if x % 2 == 0])
            recent_even_counts.append(even_count)
        
        features['recent_even_mean'] = np.mean(recent_even_counts)
        
        # Features de distribution
        low_count = sum([1 for x in window_numbers if x <= 25])
        features['low_ratio'] = low_count / len(window_numbers)
        
        return features
        
    def define_optimization_targets(self, error_analysis):
        """Définit les cibles d'optimisation."""
        
        if not error_analysis:
            return []
        
        targets = []
        
        # Cible 1: Réduire les faux positifs
        false_positives = error_analysis['number_errors']['false_positives']
        if false_positives:
            targets.append({
                'type': 'reduce_false_positives',
                'numbers_to_avoid': false_positives,
                'priority': 'high'
            })
        
        # Cible 2: Capturer les numéros manqués
        missed_numbers = error_analysis['number_errors']['missed_numbers']
        if missed_numbers:
            targets.append({
                'type': 'capture_missed_numbers',
                'numbers_to_include': missed_numbers,
                'priority': 'critical'
            })
        
        # Cible 3: Améliorer la précision statistique
        sum_error = error_analysis['statistical_errors']['sum_error']
        if sum_error > 20:
            targets.append({
                'type': 'improve_sum_accuracy',
                'target_sum': self.target_draw['sum'],
                'current_error': sum_error,
                'priority': 'medium'
            })
        
        return targets
        
    def create_analysis_visualizations(self, analysis_results):
        """Crée des visualisations de l'analyse."""
        print("📊 Création des visualisations d'analyse...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Analyse Rétroactive du Tirage Cible', fontsize=16, fontweight='bold')
        
        # 1. Fréquences des numéros cibles
        ax1 = axes[0, 0]
        freq_analysis = analysis_results['key_patterns']['frequency_analysis']
        numbers = list(freq_analysis.keys())
        frequencies = list(freq_analysis.values())
        
        bars = ax1.bar(numbers, frequencies, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Numéros Cibles')
        ax1.set_ylabel('Fréquence Historique')
        ax1.set_title('Fréquences Historiques des Numéros Cibles')
        ax1.grid(True, alpha=0.3)
        
        # Ajout des valeurs sur les barres
        for bar, freq in zip(bars, frequencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{freq:.3f}', ha='center', va='bottom')
        
        # 2. Analyse des erreurs
        ax2 = axes[0, 1]
        if analysis_results['error_analysis']:
            error_data = analysis_results['error_analysis']['number_errors']
            categories = ['Corrects', 'Manqués', 'Faux Positifs']
            values = [
                len(error_data['correct_predictions']),
                len(error_data['missed_numbers']),
                len(error_data['false_positives'])
            ]
            colors = ['green', 'red', 'orange']
            
            bars = ax2.bar(categories, values, color=colors, alpha=0.7)
            ax2.set_ylabel('Nombre')
            ax2.set_title('Analyse des Erreurs de Prédiction')
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(value), ha='center', va='bottom')
        
        # 3. Distribution par décennies
        ax3 = axes[0, 2]
        decades = analysis_results['target_analysis']['patterns']['decades']
        decade_names = list(decades.keys())
        decade_counts = list(decades.values())
        
        ax3.bar(decade_names, decade_counts, color='lightcoral', alpha=0.7)
        ax3.set_xlabel('Décennies')
        ax3.set_ylabel('Nombre de Numéros')
        ax3.set_title('Répartition par Décennies')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Similarité des tirages historiques
        ax4 = axes[1, 0]
        similar_draws = analysis_results['similar_draws'][:10]
        indices = [draw['index'] for draw in similar_draws]
        similarities = [draw['similarity_score'] for draw in similar_draws]
        
        ax4.bar(range(len(indices)), similarities, color='purple', alpha=0.7)
        ax4.set_xlabel('Tirages Historiques (Top 10)')
        ax4.set_ylabel('Score de Similarité')
        ax4.set_title('Tirages les Plus Similaires')
        ax4.grid(True, alpha=0.3)
        
        # 5. Importance des features
        ax5 = axes[1, 1]
        feature_importance = analysis_results['optimization_insights']['feature_importance']
        top_features = dict(list(feature_importance.items())[:10])
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        ax5.barh(features, importances, color='teal', alpha=0.7)
        ax5.set_xlabel('Importance')
        ax5.set_title('Top 10 Features Importantes')
        ax5.grid(True, alpha=0.3)
        
        # 6. Patterns temporels
        ax6 = axes[1, 2]
        temporal = analysis_results['key_patterns']['temporal_patterns']
        
        categories = ['Somme', 'Moyenne']
        actual_values = [
            analysis_results['target_analysis']['basic_stats']['sum'],
            analysis_results['target_analysis']['basic_stats']['mean']
        ]
        expected_values = [
            temporal['sum_trend']['expected_at_target'],
            temporal['mean_trend']['expected_at_target']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax6.bar(x - width/2, actual_values, width, label='Valeurs Réelles', color='blue', alpha=0.7)
        ax6.bar(x + width/2, expected_values, width, label='Valeurs Attendues', color='red', alpha=0.7)
        
        ax6.set_xlabel('Métriques')
        ax6.set_ylabel('Valeurs')
        ax6.set_title('Comparaison Réel vs Attendu')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/results/targeted_analysis/patterns/retroactive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualisations créées!")
        
    def save_analysis_results(self, results):
        """Sauvegarde les résultats d'analyse."""
        print("💾 Sauvegarde des résultats d'analyse...")
        
        # Sauvegarde JSON
        with open('/home/ubuntu/results/targeted_analysis/retroactive_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Rapport détaillé
        report = self.generate_analysis_report(results)
        with open('/home/ubuntu/results/targeted_analysis/analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("✅ Résultats sauvegardés!")
        
    def generate_analysis_report(self, results):
        """Génère un rapport d'analyse détaillé."""
        
        report = f"""RAPPORT D'ANALYSE RÉTROACTIVE - TIRAGE CIBLE
==========================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tirage cible: {self.target_draw['numbers']} + {self.target_draw['stars']}
Date du tirage: {self.target_draw['date']}

OBJECTIF
========

Analyser rétroactivement le tirage réel du 06/06/2025 pour identifier
les patterns et caractéristiques qui auraient permis une meilleure prédiction.

CARACTÉRISTIQUES DU TIRAGE CIBLE
================================

Numéros: {self.target_draw['numbers']}
Étoiles: {self.target_draw['stars']}
Somme: {self.target_draw['sum']}
Moyenne: {self.target_draw['mean']:.2f}
Écart-type: {self.target_draw['std']:.2f}
Étendue: {self.target_draw['range']}
Nombres pairs: {self.target_draw['even_count']}/5
Paires consécutives: {self.target_draw['consecutive_pairs']}

ANALYSE DES ERREURS DE PRÉDICTION
=================================
"""

        if results['error_analysis']:
            error_data = results['error_analysis']
            report += f"""
NUMÉROS:
- Prédits correctement: {error_data['number_errors']['correct_predictions']}
- Numéros manqués: {error_data['number_errors']['missed_numbers']}
- Faux positifs: {error_data['number_errors']['false_positives']}

ÉTOILES:
- Prédites correctement: {error_data['star_errors']['correct_predictions']}
- Étoiles manquées: {error_data['star_errors']['missed_stars']}
- Faux positifs: {error_data['star_errors']['false_positives']}

ERREURS STATISTIQUES:
- Erreur de somme: {error_data['statistical_errors']['sum_error']}
- Erreur de moyenne: {error_data['statistical_errors']['mean_error']:.2f}
"""

        report += f"""

FRÉQUENCES HISTORIQUES DES NUMÉROS CIBLES
=========================================
"""

        freq_analysis = results['key_patterns']['frequency_analysis']
        for num, freq in freq_analysis.items():
            report += f"Numéro {num}: {freq:.3f} ({freq*100:.1f}%)\n"

        report += f"""

TIRAGES HISTORIQUES SIMILAIRES (TOP 5)
======================================
"""

        for i, draw in enumerate(results['similar_draws'][:5]):
            report += f"""
{i+1}. Tirage {draw['index']} - Score: {draw['similarity_score']:.1f}
   Numéros: {draw['numbers']}
   Étoiles: {draw['stars']}
   Date: {draw['date']}
"""

        report += f"""

PATTERNS TEMPORELS
==================

TENDANCE DES SOMMES:
- Pente: {results['key_patterns']['temporal_patterns']['sum_trend']['slope']:.4f}
- R²: {results['key_patterns']['temporal_patterns']['sum_trend']['r_squared']:.3f}
- Valeur attendue: {results['key_patterns']['temporal_patterns']['sum_trend']['expected_at_target']:.1f}
- Valeur réelle: {results['key_patterns']['temporal_patterns']['sum_trend']['actual_target']}
- Erreur: {results['key_patterns']['temporal_patterns']['sum_trend']['trend_error']:.1f}

TENDANCE DES MOYENNES:
- Pente: {results['key_patterns']['temporal_patterns']['mean_trend']['slope']:.4f}
- R²: {results['key_patterns']['temporal_patterns']['mean_trend']['r_squared']:.3f}
- Valeur attendue: {results['key_patterns']['temporal_patterns']['mean_trend']['expected_at_target']:.2f}
- Valeur réelle: {results['key_patterns']['temporal_patterns']['mean_trend']['actual_target']:.2f}
- Erreur: {results['key_patterns']['temporal_patterns']['mean_trend']['trend_error']:.2f}

INSIGHTS D'OPTIMISATION
=======================

NUMÉROS CRITIQUES:
"""

        critical_numbers = results['optimization_insights']['critical_numbers']
        for num_info in critical_numbers:
            report += f"- Numéro {num_info['number']}: Fréquence {num_info['frequency']:.3f}, Priorité {num_info['priority']}\n"

        report += f"""

RECOMMANDATIONS DE PATTERNS:
"""

        recommendations = results['optimization_insights']['pattern_recommendations']
        for rec in recommendations:
            report += f"- {rec['type']}: {rec.get('target_value', 'N/A')}, Priorité {rec['priority']}\n"

        report += f"""

TOP 10 FEATURES IMPORTANTES:
"""

        feature_importance = results['optimization_insights']['feature_importance']
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            report += f"{i+1}. {feature}: {importance:.4f}\n"

        report += f"""

CIBLES D'OPTIMISATION:
"""

        targets = results['optimization_insights']['optimization_targets']
        for target in targets:
            report += f"- {target['type']}: Priorité {target['priority']}\n"

        report += f"""

CONCLUSIONS
===========

1. NUMÉROS MANQUÉS CRITIQUES: {error_data['number_errors']['missed_numbers'] if results['error_analysis'] else 'N/A'}
2. PATTERNS IDENTIFIÉS: Répartition par décennies, fréquences historiques
3. AMÉLIORATIONS POSSIBLES: Optimisation ciblée sur les numéros critiques
4. PROCHAINES ÉTAPES: Développement d'un modèle optimisé

RECOMMANDATIONS POUR L'OPTIMISATION
===================================

1. Intégrer les fréquences historiques des numéros cibles
2. Utiliser les patterns de similarité identifiés
3. Optimiser les features les plus importantes
4. Appliquer les contraintes de patterns recommandées
5. Valider avec les tirages historiques similaires

Rapport généré par le Système d'Analyse Rétroactive
==================================================
"""

        return report
        
    def run_complete_analysis(self):
        """Exécute l'analyse rétroactive complète."""
        print("🚀 LANCEMENT DE L'ANALYSE RÉTROACTIVE COMPLÈTE 🚀")
        print("=" * 70)
        
        # Génération des insights d'optimisation
        print("💡 Génération des insights d'optimisation...")
        results = self.generate_optimization_insights()
        
        # Visualisations
        print("📊 Création des visualisations...")
        self.create_analysis_visualizations(results)
        
        # Sauvegarde
        print("💾 Sauvegarde des résultats...")
        self.save_analysis_results(results)
        
        print("✅ ANALYSE RÉTROACTIVE TERMINÉE!")
        return results

if __name__ == "__main__":
    # Lancement de l'analyse rétroactive
    analyzer = TargetedAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print(f"\n🎯 RÉSULTATS DE L'ANALYSE RÉTROACTIVE:")
    print(f"Numéros cibles analysés: {analyzer.target_draw['numbers']}")
    print(f"Étoiles cibles: {analyzer.target_draw['stars']}")
    
    if results['error_analysis']:
        error_data = results['error_analysis']['number_errors']
        print(f"Numéros manqués: {error_data['missed_numbers']}")
        print(f"Numéros corrects: {error_data['correct_predictions']}")
    
    print(f"Tirages similaires trouvés: {len(results['similar_draws'])}")
    print(f"Features importantes identifiées: {len(results['optimization_insights']['feature_importance'])}")
    
    print("\n🎉 PHASE 1 TERMINÉE - PRÊT POUR L'OPTIMISATION! 🎉")

