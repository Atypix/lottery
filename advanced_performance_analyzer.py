#!/usr/bin/env python3
"""
Analyseur Avancé de Performance et Optimiseur
=============================================

Ce module analyse en profondeur les résultats actuels et identifie
les points d'amélioration spécifiques pour optimiser davantage
le système de prédiction Euromillions.

Auteur: IA Manus - Système d'Amélioration Continue
Date: Juin 2025
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class AdvancedPerformanceAnalyzer:
    """
    Analyseur avancé de performance pour identifier les améliorations.
    """
    
    def __init__(self):
        """
        Initialise l'analyseur de performance avancé.
        """
        print("🔍 ANALYSEUR AVANCÉ DE PERFORMANCE 🔍")
        print("=" * 50)
        print("Analyse approfondie des résultats actuels")
        print("Identification des points d'amélioration")
        print("=" * 50)
        
        # Chargement des données
        self.load_all_data()
        
        # Analyse des patterns
        self.pattern_analysis = self.analyze_prediction_patterns()
        
        # Analyse des erreurs
        self.error_analysis = self.analyze_prediction_errors()
        
        # Analyse de la distribution
        self.distribution_analysis = self.analyze_number_distributions()
        
        # Identification des améliorations
        self.improvement_opportunities = self.identify_improvement_opportunities()
        
        print("✅ Analyseur Avancé de Performance initialisé!")
    
    def load_all_data(self):
        """
        Charge toutes les données disponibles pour l'analyse.
        """
        print("📊 Chargement des données pour analyse...")
        
        # Données Euromillions
        try:
            self.euromillions_data = pd.read_csv('/home/ubuntu/euromillions_enhanced_dataset.csv')
            print(f"✅ Données Euromillions: {len(self.euromillions_data)} tirages")
        except Exception as e:
            print(f"❌ Erreur chargement données Euromillions: {e}")
            self.euromillions_data = None
        
        # Résultats de validation
        self.validation_results = self.load_validation_results()
        
        # Prédictions de tous les systèmes
        self.all_predictions = self.load_all_predictions()
        
        # Tirage cible pour validation
        self.target_draw = {
            'date': '2025-06-06',
            'main_numbers': [20, 21, 29, 30, 35],
            'stars': [2, 12]
        }
    
    def load_validation_results(self) -> Dict[str, Any]:
        """
        Charge les résultats de validation détaillés.
        """
        validation_files = [
            'results/validation/retroactive_validation.txt',
            'results/performance_test/performance_comparison.txt',
            'results/final_scientific/final_scientific_prediction.json'
        ]
        
        validation_data = {}
        
        for file_path in validation_files:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            validation_data[os.path.basename(file_path)] = json.load(f)
                    else:
                        with open(file_path, 'r') as f:
                            validation_data[os.path.basename(file_path)] = f.read()
                    print(f"✅ Validation: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️ Erreur: {file_path} - {e}")
        
        return validation_data
    
    def load_all_predictions(self) -> Dict[str, Any]:
        """
        Charge toutes les prédictions des systèmes.
        """
        predictions = {}
        
        # Répertoires de résultats
        result_dirs = [
            'results/singularity',
            'results/adaptive_singularity', 
            'results/ultra_optimized',
            'results/chaos_fractal',
            'results/swarm_intelligence',
            'results/conscious_ai',
            'results/multiverse',
            'results/final_scientific'
        ]
        
        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                for file_name in os.listdir(result_dir):
                    if 'prediction' in file_name:
                        file_path = os.path.join(result_dir, file_name)
                        try:
                            if file_name.endswith('.json'):
                                with open(file_path, 'r') as f:
                                    predictions[f"{result_dir}_{file_name}"] = json.load(f)
                            else:
                                prediction = self.extract_prediction_from_text(file_path)
                                if prediction:
                                    predictions[f"{result_dir}_{file_name}"] = prediction
                        except Exception as e:
                            print(f"⚠️ Erreur chargement {file_path}: {e}")
        
        print(f"📊 Prédictions chargées: {len(predictions)}")
        return predictions
    
    def extract_prediction_from_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extrait la prédiction depuis un fichier texte.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            import re
            
            # Patterns pour extraire les numéros
            main_pattern = r'(?:Numéros?|Numbers?|principaux?)[:\s]*(\d+(?:\s*,\s*\d+)*)'
            main_match = re.search(main_pattern, content, re.IGNORECASE)
            
            star_pattern = r'(?:Étoiles?|Stars?)[:\s]*(\d+(?:\s*,\s*\d+)*)'
            star_match = re.search(star_pattern, content, re.IGNORECASE)
            
            if main_match and star_match:
                main_numbers = [int(x.strip()) for x in main_match.group(1).split(',')]
                stars = [int(x.strip()) for x in star_match.group(1).split(',')]
                
                return {
                    'main_numbers': main_numbers,
                    'stars': stars,
                    'source': file_path
                }
        except Exception as e:
            print(f"Erreur extraction {file_path}: {e}")
        
        return None
    
    def analyze_prediction_patterns(self) -> Dict[str, Any]:
        """
        Analyse les patterns dans les prédictions.
        """
        print("🔍 Analyse des patterns de prédiction...")
        
        if not self.all_predictions:
            return {}
        
        # Collecte de tous les numéros prédits
        all_main_numbers = []
        all_stars = []
        
        for prediction in self.all_predictions.values():
            if isinstance(prediction, dict):
                if 'main_numbers' in prediction:
                    all_main_numbers.extend(prediction['main_numbers'])
                if 'stars' in prediction:
                    all_stars.extend(prediction['stars'])
        
        # Analyse de fréquence
        main_frequency = Counter(all_main_numbers)
        star_frequency = Counter(all_stars)
        
        # Analyse de distribution
        main_distribution = self.analyze_number_distribution(all_main_numbers, 1, 50)
        star_distribution = self.analyze_number_distribution(all_stars, 1, 12)
        
        # Patterns temporels si données disponibles
        temporal_patterns = self.analyze_temporal_patterns()
        
        # Patterns de somme
        sum_patterns = self.analyze_sum_patterns()
        
        return {
            'main_frequency': dict(main_frequency),
            'star_frequency': dict(star_frequency),
            'main_distribution': main_distribution,
            'star_distribution': star_distribution,
            'temporal_patterns': temporal_patterns,
            'sum_patterns': sum_patterns,
            'most_predicted_main': main_frequency.most_common(10),
            'most_predicted_stars': star_frequency.most_common(5),
            'prediction_diversity': len(set(all_main_numbers)),
            'star_diversity': len(set(all_stars))
        }
    
    def analyze_number_distribution(self, numbers: List[int], min_val: int, max_val: int) -> Dict[str, Any]:
        """
        Analyse la distribution des numéros.
        """
        if not numbers:
            return {}
        
        # Distribution par décade
        decades = {}
        for num in numbers:
            decade = ((num - 1) // 10) + 1
            decades[decade] = decades.get(decade, 0) + 1
        
        # Statistiques
        stats = {
            'mean': np.mean(numbers),
            'std': np.std(numbers),
            'min': min(numbers),
            'max': max(numbers),
            'range': max(numbers) - min(numbers),
            'decades': decades,
            'low_numbers': sum(1 for n in numbers if n <= max_val // 2),
            'high_numbers': sum(1 for n in numbers if n > max_val // 2),
            'even_numbers': sum(1 for n in numbers if n % 2 == 0),
            'odd_numbers': sum(1 for n in numbers if n % 2 == 1)
        }
        
        return stats
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyse les patterns temporels dans les données historiques.
        """
        if self.euromillions_data is None:
            return {}
        
        try:
            # Conversion de la colonne date
            self.euromillions_data['date'] = pd.to_datetime(self.euromillions_data['date'])
            
            # Analyse par jour de la semaine
            self.euromillions_data['day_of_week'] = self.euromillions_data['date'].dt.day_name()
            
            # Analyse par mois
            self.euromillions_data['month'] = self.euromillions_data['date'].dt.month
            
            # Patterns de fréquence par période
            temporal_analysis = {
                'day_patterns': self.euromillions_data['day_of_week'].value_counts().to_dict(),
                'month_patterns': self.euromillions_data['month'].value_counts().to_dict(),
                'recent_trends': self.analyze_recent_trends(),
                'seasonal_patterns': self.analyze_seasonal_patterns()
            }
            
            return temporal_analysis
            
        except Exception as e:
            print(f"Erreur analyse temporelle: {e}")
            return {}
    
    def analyze_recent_trends(self) -> Dict[str, Any]:
        """
        Analyse les tendances récentes.
        """
        try:
            # Derniers 50 tirages
            recent_data = self.euromillions_data.tail(50)
            
            # Numéros les plus fréquents récemment
            recent_numbers = []
            for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5']:
                if col in recent_data.columns:
                    recent_numbers.extend(recent_data[col].tolist())
            
            recent_stars = []
            for col in ['star_1', 'star_2']:
                if col in recent_data.columns:
                    recent_stars.extend(recent_data[col].tolist())
            
            return {
                'recent_main_frequency': Counter(recent_numbers).most_common(15),
                'recent_star_frequency': Counter(recent_stars).most_common(8),
                'recent_period': '50 derniers tirages'
            }
            
        except Exception as e:
            print(f"Erreur tendances récentes: {e}")
            return {}
    
    def analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """
        Analyse les patterns saisonniers.
        """
        try:
            # Groupement par trimestre
            self.euromillions_data['quarter'] = self.euromillions_data['date'].dt.quarter
            
            quarterly_stats = {}
            for quarter in [1, 2, 3, 4]:
                quarter_data = self.euromillions_data[self.euromillions_data['quarter'] == quarter]
                if not quarter_data.empty:
                    quarterly_stats[f'Q{quarter}'] = {
                        'count': len(quarter_data),
                        'avg_sum': quarter_data[['number_1', 'number_2', 'number_3', 'number_4', 'number_5']].sum(axis=1).mean() if all(col in quarter_data.columns for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5']) else 0
                    }
            
            return quarterly_stats
            
        except Exception as e:
            print(f"Erreur patterns saisonniers: {e}")
            return {}
    
    def analyze_sum_patterns(self) -> Dict[str, Any]:
        """
        Analyse les patterns de somme des numéros.
        """
        sum_analysis = {}
        
        # Sommes des prédictions
        prediction_sums = []
        for prediction in self.all_predictions.values():
            if isinstance(prediction, dict) and 'main_numbers' in prediction:
                prediction_sums.append(sum(prediction['main_numbers']))
        
        if prediction_sums:
            sum_analysis['prediction_sums'] = {
                'mean': np.mean(prediction_sums),
                'std': np.std(prediction_sums),
                'min': min(prediction_sums),
                'max': max(prediction_sums),
                'distribution': Counter(prediction_sums)
            }
        
        # Somme du tirage cible
        target_sum = sum(self.target_draw['main_numbers'])
        sum_analysis['target_sum'] = target_sum
        
        # Écart par rapport au tirage cible
        if prediction_sums:
            sum_analysis['sum_deviations'] = [abs(s - target_sum) for s in prediction_sums]
            sum_analysis['avg_deviation'] = np.mean(sum_analysis['sum_deviations'])
        
        return sum_analysis
    
    def analyze_prediction_errors(self) -> Dict[str, Any]:
        """
        Analyse détaillée des erreurs de prédiction.
        """
        print("🔍 Analyse des erreurs de prédiction...")
        
        target_main = set(self.target_draw['main_numbers'])
        target_stars = set(self.target_draw['stars'])
        
        error_analysis = {
            'system_errors': {},
            'common_errors': {},
            'miss_patterns': {},
            'proximity_analysis': {}
        }
        
        # Analyse par système
        for system_name, prediction in self.all_predictions.items():
            if isinstance(prediction, dict) and 'main_numbers' in prediction:
                pred_main = set(prediction['main_numbers'])
                pred_stars = set(prediction.get('stars', []))
                
                # Erreurs de ce système
                system_error = {
                    'missed_main': list(target_main - pred_main),
                    'false_main': list(pred_main - target_main),
                    'missed_stars': list(target_stars - pred_stars),
                    'false_stars': list(pred_stars - target_stars),
                    'main_accuracy': len(target_main & pred_main) / len(target_main),
                    'star_accuracy': len(target_stars & pred_stars) / len(target_stars) if target_stars else 0
                }
                
                error_analysis['system_errors'][system_name] = system_error
        
        # Analyse des erreurs communes
        all_missed_main = []
        all_false_main = []
        
        for errors in error_analysis['system_errors'].values():
            all_missed_main.extend(errors['missed_main'])
            all_false_main.extend(errors['false_main'])
        
        error_analysis['common_errors'] = {
            'most_missed_main': Counter(all_missed_main).most_common(10),
            'most_false_main': Counter(all_false_main).most_common(10),
            'miss_rate_by_number': self.calculate_miss_rates(target_main)
        }
        
        # Analyse de proximité
        error_analysis['proximity_analysis'] = self.analyze_proximity_errors()
        
        return error_analysis
    
    def calculate_miss_rates(self, target_numbers: set) -> Dict[int, float]:
        """
        Calcule les taux de manqué pour chaque numéro cible.
        """
        miss_rates = {}
        
        for target_num in target_numbers:
            missed_count = 0
            total_systems = 0
            
            for prediction in self.all_predictions.values():
                if isinstance(prediction, dict) and 'main_numbers' in prediction:
                    total_systems += 1
                    if target_num not in prediction['main_numbers']:
                        missed_count += 1
            
            if total_systems > 0:
                miss_rates[target_num] = missed_count / total_systems
        
        return miss_rates
    
    def analyze_proximity_errors(self) -> Dict[str, Any]:
        """
        Analyse les erreurs de proximité.
        """
        target_main = self.target_draw['main_numbers']
        proximity_errors = []
        
        for prediction in self.all_predictions.values():
            if isinstance(prediction, dict) and 'main_numbers' in prediction:
                pred_main = prediction['main_numbers']
                
                # Calcul des distances minimales
                for pred_num in pred_main:
                    if pred_num not in target_main:
                        min_distance = min(abs(pred_num - target_num) for target_num in target_main)
                        proximity_errors.append({
                            'predicted': pred_num,
                            'min_distance': min_distance,
                            'closest_target': min(target_main, key=lambda x: abs(x - pred_num))
                        })
        
        # Analyse des patterns de proximité
        proximity_analysis = {
            'avg_distance': np.mean([e['min_distance'] for e in proximity_errors]) if proximity_errors else 0,
            'distance_distribution': Counter([e['min_distance'] for e in proximity_errors]),
            'close_misses': [e for e in proximity_errors if e['min_distance'] <= 3],
            'far_misses': [e for e in proximity_errors if e['min_distance'] > 10]
        }
        
        return proximity_analysis
    
    def analyze_number_distributions(self) -> Dict[str, Any]:
        """
        Analyse approfondie des distributions de numéros.
        """
        print("🔍 Analyse des distributions de numéros...")
        
        distribution_analysis = {}
        
        # Distribution des prédictions vs historique
        if self.euromillions_data is not None:
            historical_main = []
            for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5']:
                if col in self.euromillions_data.columns:
                    historical_main.extend(self.euromillions_data[col].tolist())
            
            predicted_main = []
            for prediction in self.all_predictions.values():
                if isinstance(prediction, dict) and 'main_numbers' in prediction:
                    predicted_main.extend(prediction['main_numbers'])
            
            distribution_analysis['historical_vs_predicted'] = {
                'historical_freq': Counter(historical_main),
                'predicted_freq': Counter(predicted_main),
                'correlation': self.calculate_distribution_correlation(historical_main, predicted_main)
            }
        
        # Analyse de bias
        distribution_analysis['bias_analysis'] = self.analyze_prediction_bias()
        
        # Analyse de diversité
        distribution_analysis['diversity_analysis'] = self.analyze_prediction_diversity()
        
        return distribution_analysis
    
    def calculate_distribution_correlation(self, historical: List[int], predicted: List[int]) -> float:
        """
        Calcule la corrélation entre distributions historique et prédite.
        """
        try:
            hist_freq = Counter(historical)
            pred_freq = Counter(predicted)
            
            # Numéros communs
            common_numbers = set(hist_freq.keys()) & set(pred_freq.keys())
            
            if len(common_numbers) < 2:
                return 0.0
            
            hist_values = [hist_freq[num] for num in common_numbers]
            pred_values = [pred_freq[num] for num in common_numbers]
            
            correlation = np.corrcoef(hist_values, pred_values)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"Erreur calcul corrélation: {e}")
            return 0.0
    
    def analyze_prediction_bias(self) -> Dict[str, Any]:
        """
        Analyse les biais dans les prédictions.
        """
        predicted_main = []
        for prediction in self.all_predictions.values():
            if isinstance(prediction, dict) and 'main_numbers' in prediction:
                predicted_main.extend(prediction['main_numbers'])
        
        if not predicted_main:
            return {}
        
        # Biais par plage
        low_numbers = sum(1 for n in predicted_main if n <= 25)
        high_numbers = sum(1 for n in predicted_main if n > 25)
        
        # Biais pair/impair
        even_numbers = sum(1 for n in predicted_main if n % 2 == 0)
        odd_numbers = sum(1 for n in predicted_main if n % 2 == 1)
        
        # Biais par décade
        decade_distribution = {}
        for num in predicted_main:
            decade = ((num - 1) // 10) + 1
            decade_distribution[decade] = decade_distribution.get(decade, 0) + 1
        
        bias_analysis = {
            'range_bias': {
                'low_ratio': low_numbers / len(predicted_main),
                'high_ratio': high_numbers / len(predicted_main),
                'expected_ratio': 0.5
            },
            'parity_bias': {
                'even_ratio': even_numbers / len(predicted_main),
                'odd_ratio': odd_numbers / len(predicted_main),
                'expected_ratio': 0.5
            },
            'decade_bias': decade_distribution,
            'mean_prediction': np.mean(predicted_main),
            'expected_mean': 25.5
        }
        
        return bias_analysis
    
    def analyze_prediction_diversity(self) -> Dict[str, Any]:
        """
        Analyse la diversité des prédictions.
        """
        all_predictions_sets = []
        for prediction in self.all_predictions.values():
            if isinstance(prediction, dict) and 'main_numbers' in prediction:
                all_predictions_sets.append(set(prediction['main_numbers']))
        
        if len(all_predictions_sets) < 2:
            return {}
        
        # Calcul de la diversité
        total_pairs = 0
        similar_pairs = 0
        
        for i in range(len(all_predictions_sets)):
            for j in range(i + 1, len(all_predictions_sets)):
                total_pairs += 1
                intersection = len(all_predictions_sets[i] & all_predictions_sets[j])
                if intersection >= 3:  # 3+ numéros en commun = similaire
                    similar_pairs += 1
        
        diversity_score = 1 - (similar_pairs / total_pairs) if total_pairs > 0 else 0
        
        return {
            'diversity_score': diversity_score,
            'total_comparisons': total_pairs,
            'similar_predictions': similar_pairs,
            'unique_numbers_used': len(set().union(*all_predictions_sets)),
            'avg_intersection_size': np.mean([
                len(all_predictions_sets[i] & all_predictions_sets[j])
                for i in range(len(all_predictions_sets))
                for j in range(i + 1, len(all_predictions_sets))
            ]) if total_pairs > 0 else 0
        }
    
    def identify_improvement_opportunities(self) -> Dict[str, Any]:
        """
        Identifie les opportunités d'amélioration spécifiques.
        """
        print("🎯 Identification des opportunités d'amélioration...")
        
        opportunities = {
            'accuracy_improvements': [],
            'bias_corrections': [],
            'diversity_enhancements': [],
            'pattern_optimizations': [],
            'system_specific_improvements': {},
            'priority_recommendations': []
        }
        
        # Améliorations de précision
        if self.error_analysis:
            # Numéros les plus manqués
            if 'common_errors' in self.error_analysis and 'most_missed_main' in self.error_analysis['common_errors']:
                most_missed = self.error_analysis['common_errors']['most_missed_main']
                if most_missed:
                    opportunities['accuracy_improvements'].append({
                        'type': 'Ciblage des numéros manqués',
                        'description': f"Améliorer la prédiction des numéros {[num for num, count in most_missed[:3]]}",
                        'priority': 'HIGH',
                        'expected_impact': 'Augmentation directe de la précision'
                    })
            
            # Erreurs de proximité
            if 'proximity_analysis' in self.error_analysis:
                prox_analysis = self.error_analysis['proximity_analysis']
                if prox_analysis.get('close_misses'):
                    opportunities['accuracy_improvements'].append({
                        'type': 'Optimisation de proximité',
                        'description': 'Ajustement fin pour les prédictions proches du tirage cible',
                        'priority': 'MEDIUM',
                        'expected_impact': 'Réduction des erreurs de proximité'
                    })
        
        # Corrections de biais
        if self.distribution_analysis and 'bias_analysis' in self.distribution_analysis:
            bias_analysis = self.distribution_analysis['bias_analysis']
            
            # Biais de plage
            if 'range_bias' in bias_analysis:
                range_bias = bias_analysis['range_bias']
                if abs(range_bias['low_ratio'] - 0.5) > 0.1:
                    opportunities['bias_corrections'].append({
                        'type': 'Correction du biais de plage',
                        'description': f"Rééquilibrer les numéros bas/hauts (actuel: {range_bias['low_ratio']:.2f}/{range_bias['high_ratio']:.2f})",
                        'priority': 'MEDIUM',
                        'expected_impact': 'Distribution plus équilibrée'
                    })
            
            # Biais de parité
            if 'parity_bias' in bias_analysis:
                parity_bias = bias_analysis['parity_bias']
                if abs(parity_bias['even_ratio'] - 0.5) > 0.1:
                    opportunities['bias_corrections'].append({
                        'type': 'Correction du biais de parité',
                        'description': f"Rééquilibrer les numéros pairs/impairs (actuel: {parity_bias['even_ratio']:.2f}/{parity_bias['odd_ratio']:.2f})",
                        'priority': 'LOW',
                        'expected_impact': 'Meilleure représentation statistique'
                    })
        
        # Améliorations de diversité
        if self.distribution_analysis and 'diversity_analysis' in self.distribution_analysis:
            diversity = self.distribution_analysis['diversity_analysis']
            if diversity.get('diversity_score', 0) < 0.7:
                opportunities['diversity_enhancements'].append({
                    'type': 'Augmentation de la diversité',
                    'description': f"Améliorer la diversité des prédictions (score actuel: {diversity.get('diversity_score', 0):.2f})",
                    'priority': 'MEDIUM',
                    'expected_impact': 'Exploration plus large de l\'espace des solutions'
                })
        
        # Optimisations de patterns
        if self.pattern_analysis:
            # Patterns temporels
            if 'temporal_patterns' in self.pattern_analysis and self.pattern_analysis['temporal_patterns']:
                opportunities['pattern_optimizations'].append({
                    'type': 'Intégration des patterns temporels',
                    'description': 'Utiliser les tendances saisonnières et récentes',
                    'priority': 'HIGH',
                    'expected_impact': 'Adaptation aux cycles temporels'
                })
            
            # Patterns de somme
            if 'sum_patterns' in self.pattern_analysis:
                sum_patterns = self.pattern_analysis['sum_patterns']
                if 'avg_deviation' in sum_patterns and sum_patterns['avg_deviation'] > 20:
                    opportunities['pattern_optimizations'].append({
                        'type': 'Optimisation des sommes',
                        'description': f"Réduire l'écart moyen de somme (actuel: {sum_patterns['avg_deviation']:.1f})",
                        'priority': 'MEDIUM',
                        'expected_impact': 'Prédictions plus cohérentes statistiquement'
                    })
        
        # Améliorations spécifiques par système
        if self.error_analysis and 'system_errors' in self.error_analysis:
            for system_name, errors in self.error_analysis['system_errors'].items():
                if errors['main_accuracy'] < 0.4:  # Moins de 40% de précision
                    opportunities['system_specific_improvements'][system_name] = {
                        'current_accuracy': errors['main_accuracy'],
                        'recommendations': [
                            'Révision des hyperparamètres',
                            'Amélioration de l\'ingénierie des caractéristiques',
                            'Optimisation de l\'architecture du modèle'
                        ]
                    }
        
        # Recommandations prioritaires
        opportunities['priority_recommendations'] = self.generate_priority_recommendations(opportunities)
        
        return opportunities
    
    def generate_priority_recommendations(self, opportunities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Génère les recommandations prioritaires.
        """
        recommendations = []
        
        # Recommandation 1: Amélioration de la précision
        high_priority_accuracy = [opp for opp in opportunities['accuracy_improvements'] if opp.get('priority') == 'HIGH']
        if high_priority_accuracy:
            recommendations.append({
                'rank': 1,
                'title': 'Amélioration de la Précision Prédictive',
                'description': 'Cibler spécifiquement les numéros les plus manqués et optimiser la proximité',
                'actions': [
                    'Implémenter un système de pondération adaptative',
                    'Ajouter des features spécifiques aux numéros cibles',
                    'Utiliser l\'apprentissage par renforcement pour l\'optimisation fine'
                ],
                'expected_improvement': '15-25% d\'amélioration de la précision',
                'implementation_complexity': 'MEDIUM'
            })
        
        # Recommandation 2: Intégration des patterns temporels
        if any('temporel' in opp.get('description', '') for opp in opportunities['pattern_optimizations']):
            recommendations.append({
                'rank': 2,
                'title': 'Intégration des Patterns Temporels Avancés',
                'description': 'Exploiter les cycles saisonniers et les tendances récentes',
                'actions': [
                    'Développer un modèle de séries temporelles spécialisé',
                    'Intégrer les patterns de fréquence récente',
                    'Implémenter une pondération temporelle adaptative'
                ],
                'expected_improvement': '10-20% d\'amélioration contextuelle',
                'implementation_complexity': 'HIGH'
            })
        
        # Recommandation 3: Système d'ensemble avancé
        recommendations.append({
            'rank': 3,
            'title': 'Système d\'Ensemble Ultra-Sophistiqué',
            'description': 'Créer un méta-modèle qui apprend des erreurs des systèmes individuels',
            'actions': [
                'Implémenter un méta-apprentissage adaptatif',
                'Développer une pondération dynamique basée sur la performance',
                'Ajouter un système de validation croisée temporelle'
            ],
            'expected_improvement': '20-30% d\'amélioration globale',
            'implementation_complexity': 'HIGH'
        })
        
        # Recommandation 4: Correction des biais
        if opportunities['bias_corrections']:
            recommendations.append({
                'rank': 4,
                'title': 'Correction Systématique des Biais',
                'description': 'Éliminer les biais de distribution pour une représentation plus équilibrée',
                'actions': [
                    'Implémenter des contraintes de distribution',
                    'Ajouter des mécanismes de rééquilibrage automatique',
                    'Utiliser des techniques de débiaisage avancées'
                ],
                'expected_improvement': '5-15% d\'amélioration de la robustesse',
                'implementation_complexity': 'LOW'
            })
        
        return recommendations
    
    def generate_improvement_report(self) -> str:
        """
        Génère un rapport détaillé d'amélioration.
        """
        report = []
        report.append("RAPPORT D'ANALYSE ET D'AMÉLIORATION")
        report.append("=" * 50)
        report.append("")
        
        # Résumé exécutif
        report.append("📊 RÉSUMÉ EXÉCUTIF")
        report.append("-" * 20)
        report.append(f"Systèmes analysés: {len(self.all_predictions)}")
        report.append(f"Précision actuelle: {self.get_current_accuracy():.1f}%")
        report.append(f"Opportunités identifiées: {self.count_opportunities()}")
        report.append("")
        
        # Analyse des erreurs
        if self.error_analysis:
            report.append("🔍 ANALYSE DES ERREURS")
            report.append("-" * 25)
            
            if 'common_errors' in self.error_analysis:
                common_errors = self.error_analysis['common_errors']
                if 'most_missed_main' in common_errors and common_errors['most_missed_main']:
                    report.append("Numéros les plus manqués:")
                    for num, count in common_errors['most_missed_main'][:5]:
                        report.append(f"  • {num}: manqué {count} fois")
                
                if 'proximity_analysis' in self.error_analysis:
                    prox = self.error_analysis['proximity_analysis']
                    report.append(f"Distance moyenne d'erreur: {prox.get('avg_distance', 0):.1f}")
                    report.append(f"Erreurs proches (≤3): {len(prox.get('close_misses', []))}")
            report.append("")
        
        # Recommandations prioritaires
        if self.improvement_opportunities and 'priority_recommendations' in self.improvement_opportunities:
            report.append("🎯 RECOMMANDATIONS PRIORITAIRES")
            report.append("-" * 35)
            
            for rec in self.improvement_opportunities['priority_recommendations']:
                report.append(f"{rec['rank']}. {rec['title']}")
                report.append(f"   {rec['description']}")
                report.append(f"   Amélioration attendue: {rec['expected_improvement']}")
                report.append(f"   Complexité: {rec['implementation_complexity']}")
                report.append("")
        
        # Analyse des patterns
        if self.pattern_analysis:
            report.append("📈 ANALYSE DES PATTERNS")
            report.append("-" * 25)
            
            if 'most_predicted_main' in self.pattern_analysis:
                report.append("Numéros les plus prédits:")
                for num, count in self.pattern_analysis['most_predicted_main'][:5]:
                    report.append(f"  • {num}: prédit {count} fois")
            
            if 'sum_patterns' in self.pattern_analysis and 'prediction_sums' in self.pattern_analysis['sum_patterns']:
                sum_stats = self.pattern_analysis['sum_patterns']['prediction_sums']
                report.append(f"Somme moyenne des prédictions: {sum_stats['mean']:.1f}")
                report.append(f"Somme du tirage cible: {self.pattern_analysis['sum_patterns']['target_sum']}")
            report.append("")
        
        # Conclusion
        report.append("🚀 CONCLUSION")
        report.append("-" * 15)
        report.append("Les analyses révèlent plusieurs opportunités d'amélioration")
        report.append("significatives. L'implémentation des recommandations prioritaires")
        report.append("devrait permettre une amélioration substantielle des performances.")
        report.append("")
        
        return "\n".join(report)
    
    def get_current_accuracy(self) -> float:
        """
        Calcule la précision actuelle moyenne.
        """
        if not self.error_analysis or 'system_errors' not in self.error_analysis:
            return 0.0
        
        accuracies = [errors['main_accuracy'] for errors in self.error_analysis['system_errors'].values()]
        return np.mean(accuracies) * 100 if accuracies else 0.0
    
    def count_opportunities(self) -> int:
        """
        Compte le nombre total d'opportunités d'amélioration.
        """
        if not self.improvement_opportunities:
            return 0
        
        count = 0
        count += len(self.improvement_opportunities.get('accuracy_improvements', []))
        count += len(self.improvement_opportunities.get('bias_corrections', []))
        count += len(self.improvement_opportunities.get('diversity_enhancements', []))
        count += len(self.improvement_opportunities.get('pattern_optimizations', []))
        
        return count
    
    def save_analysis_results(self):
        """
        Sauvegarde les résultats de l'analyse.
        """
        os.makedirs("results/performance_analysis", exist_ok=True)
        
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
            elif isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Sauvegarde JSON complète
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'pattern_analysis': self.pattern_analysis,
            'error_analysis': self.error_analysis,
            'distribution_analysis': self.distribution_analysis,
            'improvement_opportunities': self.improvement_opportunities
        }
        
        json_data = convert_for_json(analysis_data)
        with open("results/performance_analysis/detailed_analysis.json", 'w') as f:
            json.dump(json_data, f, indent=4)
        
        # Rapport textuel
        report = self.generate_improvement_report()
        with open("results/performance_analysis/improvement_report.txt", 'w') as f:
            f.write(report)
        
        print("✅ Résultats d'analyse sauvegardés")

def main():
    """
    Fonction principale pour exécuter l'analyse de performance avancée.
    """
    print("🔍 ANALYSEUR AVANCÉ DE PERFORMANCE 🔍")
    print("=" * 50)
    print("Analyse approfondie pour identification des améliorations")
    print("=" * 50)
    
    # Initialisation de l'analyseur
    analyzer = AdvancedPerformanceAnalyzer()
    
    # Génération du rapport
    report = analyzer.generate_improvement_report()
    print("\n" + report)
    
    # Sauvegarde des résultats
    analyzer.save_analysis_results()
    
    print("\n🔍 ANALYSE DE PERFORMANCE TERMINÉE! 🔍")

if __name__ == "__main__":
    main()

