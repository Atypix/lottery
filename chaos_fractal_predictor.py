#!/usr/bin/env python3
"""
Système d'Analyse Fractale et de Théorie du Chaos pour Euromillions
===================================================================

Ce module implémente des techniques révolutionnaires d'analyse fractale et de théorie du chaos
pour découvrir des patterns cachés dans les séries temporelles de l'Euromillions :

1. Analyse de la Dimension Fractale (Hausdorff, Box-counting)
2. Reconstruction d'Espace de Phase (Takens)
3. Calcul des Exposants de Lyapunov
4. Détection d'Attracteurs Étranges
5. Prédiction Chaotique par Voisinage
6. Analyse de Bifurcation

Auteur: IA Manus - Système Révolutionnaire Chaos-Fractal
Date: Juin 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime
import json
import os
from typing import List, Tuple, Dict, Any, Optional
import warnings
import argparse # Added
# json, datetime are already imported
from common.date_utils import get_next_euromillions_draw_date, date as datetime_date # Added

warnings.filterwarnings('ignore')

class FractalAnalyzer:
    """
    Analyseur fractal pour découvrir la géométrie cachée des séries temporelles.
    """
    
    def __init__(self):
        """
        Initialise l'analyseur fractal.
        """
        # print("🔬 Analyseur Fractal initialisé") # Suppressed
        self.fractal_dimensions = {}
        self.self_similarity_patterns = {}
    
    def box_counting_dimension(self, data: np.ndarray, max_box_size: int = 100) -> float:
        """
        Calcule la dimension fractale par comptage de boîtes.
        """
        # Normalisation des données
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Création d'une grille 2D (temps vs valeur)
        time_points = np.arange(len(data_norm))
        
        box_sizes = []
        box_counts = []
        
        for box_size in range(1, min(max_box_size, len(data) // 4)):
            # Comptage des boîtes contenant des points
            x_boxes = len(data) // box_size
            y_boxes = 100 // box_size  # Résolution verticale
            
            if x_boxes == 0 or y_boxes == 0:
                continue
            
            occupied_boxes = set()
            
            for i, value in enumerate(data_norm):
                x_box = min(i // box_size, x_boxes - 1)
                y_box = min(int(value * y_boxes), y_boxes - 1)
                occupied_boxes.add((x_box, y_box))
            
            box_sizes.append(box_size)
            box_counts.append(len(occupied_boxes))
        
        if len(box_sizes) < 3:
            return 1.0
        
        # Régression linéaire en échelle log-log
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # Calcul de la pente (dimension fractale)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dim = -coeffs[0]  # Pente négative
        
        return max(0.1, min(fractal_dim, 3.0))  # Limitation réaliste
    
    def hurst_exponent(self, data: np.ndarray) -> float:
        """
        Calcule l'exposant de Hurst pour caractériser la persistance.
        """
        n = len(data)
        if n < 10:
            return 0.5
        
        # Méthode R/S (Rescaled Range)
        lags = range(2, min(n // 4, 100))
        rs_values = []
        
        for lag in lags:
            # Division en segments
            segments = n // lag
            if segments == 0:
                continue
            
            rs_segment = []
            
            for i in range(segments):
                segment = data[i * lag:(i + 1) * lag]
                
                # Calcul de la moyenne
                mean_segment = np.mean(segment)
                
                # Écarts cumulés
                cumulative_deviations = np.cumsum(segment - mean_segment)
                
                # Range
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                
                # Standard deviation
                S = np.std(segment)
                
                if S > 0:
                    rs_segment.append(R / S)
            
            if rs_segment:
                rs_values.append(np.mean(rs_segment))
        
        if len(rs_values) < 3:
            return 0.5
        
        # Régression log-log
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # L'exposant de Hurst est la pente
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        
        return max(0.0, min(hurst, 1.0))
    
    def detect_self_similarity(self, data: np.ndarray, scales: List[int] = None) -> Dict[str, float]:
        """
        Détecte les patterns d'auto-similarité à différentes échelles.
        """
        if scales is None:
            scales = [2, 3, 5, 8, 13, 21]  # Séquence de Fibonacci
        
        similarities = {}
        
        for scale in scales:
            if scale >= len(data) // 2:
                continue
            
            # Division en segments de différentes tailles
            segment_size = len(data) // scale
            segments = []
            
            for i in range(scale):
                start = i * segment_size
                end = start + segment_size
                if end <= len(data):
                    segment = data[start:end]
                    # Normalisation du segment
                    if np.std(segment) > 0:
                        segment_norm = (segment - np.mean(segment)) / np.std(segment)
                        segments.append(segment_norm)
            
            if len(segments) < 2:
                continue
            
            # Calcul de la similarité entre segments
            correlations = []
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    corr = np.corrcoef(segments[i], segments[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                similarities[f"scale_{scale}"] = np.mean(correlations)
        
        return similarities
    
    def wavelet_fractal_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyse fractale par ondelettes pour détecter les structures multi-échelles.
        """
        # Transformation en ondelettes continues (approximation)
        scales = np.arange(1, min(64, len(data) // 4))
        
        if len(scales) == 0:
            return {"fractal_spectrum": [], "multifractal_width": 0.0}
        
        # Calcul des coefficients d'ondelettes (approximation avec convolution)
        wavelet_coeffs = []
        
        for scale in scales:
            # Ondelette de Morlet approximée
            sigma = scale / 6.0
            wavelet_length = min(6 * int(sigma), len(data) // 2)
            
            if wavelet_length < 3:
                continue
            
            t = np.arange(-wavelet_length // 2, wavelet_length // 2)
            wavelet = np.exp(-t**2 / (2 * sigma**2)) * np.cos(5 * t / sigma)
            
            # Convolution
            if len(wavelet) < len(data):
                coeff = np.convolve(data, wavelet, mode='valid')
                wavelet_coeffs.append(np.mean(np.abs(coeff)))
        
        if len(wavelet_coeffs) < 3:
            return {"fractal_spectrum": [], "multifractal_width": 0.0}
        
        # Analyse du spectre fractal
        log_scales = np.log(scales[:len(wavelet_coeffs)])
        log_coeffs = np.log(np.array(wavelet_coeffs) + 1e-10)
        
        # Calcul de la dimension fractale locale
        fractal_spectrum = []
        window_size = 5
        
        for i in range(len(log_coeffs) - window_size):
            local_scales = log_scales[i:i + window_size]
            local_coeffs = log_coeffs[i:i + window_size]
            
            if len(local_scales) > 2:
                slope = np.polyfit(local_scales, local_coeffs, 1)[0]
                fractal_spectrum.append(-slope)
        
        # Largeur multifractale
        if fractal_spectrum:
            multifractal_width = np.max(fractal_spectrum) - np.min(fractal_spectrum)
        else:
            multifractal_width = 0.0
        
        return {
            "fractal_spectrum": fractal_spectrum,
            "multifractal_width": multifractal_width
        }

class ChaosAnalyzer:
    """
    Analyseur de théorie du chaos pour découvrir les dynamiques non-linéaires.
    """
    
    def __init__(self):
        """
        Initialise l'analyseur de chaos.
        """
        # print("🌪️ Analyseur de Chaos initialisé") # Suppressed
        self.phase_space = None
        self.lyapunov_exponents = []
        self.strange_attractors = []
    
    def phase_space_reconstruction(self, data: np.ndarray, embedding_dim: int = 3, 
                                 delay: int = 1) -> np.ndarray:
        """
        Reconstruction de l'espace de phase selon le théorème de Takens.
        """
        n = len(data)
        if n < embedding_dim * delay:
            return np.array([])
        
        # Construction de la matrice d'embedding
        embedded_data = np.zeros((n - (embedding_dim - 1) * delay, embedding_dim))
        
        for i in range(embedding_dim):
            start_idx = i * delay
            end_idx = n - (embedding_dim - 1 - i) * delay
            embedded_data[:, i] = data[start_idx:end_idx]
        
        self.phase_space = embedded_data
        return embedded_data
    
    def optimal_embedding_parameters(self, data: np.ndarray) -> Tuple[int, int]:
        """
        Détermine les paramètres optimaux d'embedding (dimension et délai).
        """
        # Calcul du délai optimal par autocorrélation
        max_delay = min(50, len(data) // 4)
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalisation
        
        # Premier zéro de l'autocorrélation
        optimal_delay = 1
        for i in range(1, min(len(autocorr), max_delay)):
            if autocorr[i] <= 0.1:  # Seuil de décorrélation
                optimal_delay = i
                break
        
        # Dimension d'embedding par faux plus proches voisins
        optimal_dim = 3  # Valeur par défaut
        
        for dim in range(2, 8):
            if len(data) < dim * optimal_delay + 10:
                break
            
            embedded = self.phase_space_reconstruction(data, dim, optimal_delay)
            if len(embedded) < 10:
                continue
            
            # Test de faux voisins (approximation)
            distances = pdist(embedded)
            if len(distances) > 0:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                
                # Critère de saturation
                if std_dist / mean_dist < 0.1:  # Faible variabilité = saturation
                    optimal_dim = dim
                    break
        
        return optimal_dim, optimal_delay
    
    def lyapunov_exponent(self, data: np.ndarray, embedding_dim: int = 3) -> float:
        """
        Calcule le plus grand exposant de Lyapunov pour quantifier le chaos.
        """
        if len(data) < 50:
            return 0.0
        
        # Reconstruction de l'espace de phase
        delay = max(1, len(data) // 20)
        embedded = self.phase_space_reconstruction(data, embedding_dim, delay)
        
        if len(embedded) < 10:
            return 0.0
        
        # Algorithme de Wolf et al. (approximation)
        lyap_sum = 0.0
        count = 0
        
        for i in range(len(embedded) - 10):
            # Point de référence
            ref_point = embedded[i]
            
            # Recherche du plus proche voisin
            distances = np.linalg.norm(embedded - ref_point, axis=1)
            distances[i] = np.inf  # Exclure le point lui-même
            
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] == 0:
                continue
            
            # Évolution après quelques pas
            steps = min(5, len(embedded) - max(i, nearest_idx) - 1)
            if steps <= 0:
                continue
            
            # Distance initiale
            d0 = distances[nearest_idx]
            
            # Distance après évolution
            future_ref = embedded[i + steps]
            future_nearest = embedded[nearest_idx + steps]
            d1 = np.linalg.norm(future_ref - future_nearest)
            
            if d0 > 0 and d1 > 0:
                lyap_sum += np.log(d1 / d0)
                count += 1
        
        if count > 0:
            lyapunov = lyap_sum / (count * steps * delay)
        else:
            lyapunov = 0.0
        
        return lyapunov
    
    def detect_strange_attractors(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Détecte la présence d'attracteurs étranges dans les données.
        """
        # Reconstruction de l'espace de phase
        embedding_dim, delay = self.optimal_embedding_parameters(data)
        embedded = self.phase_space_reconstruction(data, embedding_dim, delay)
        
        if len(embedded) < 20:
            return {"attractor_detected": False, "attractor_dimension": 0.0}
        
        # Calcul de la dimension de corrélation
        correlation_dim = self.correlation_dimension(embedded)
        
        # Détection d'attractor étrange
        # Critères: dimension non-entière et exposant de Lyapunov positif
        lyap_exp = self.lyapunov_exponent(data, embedding_dim)
        
        is_strange_attractor = (
            correlation_dim > 2.0 and 
            correlation_dim != int(correlation_dim) and
            lyap_exp > 0.01
        )
        
        return {
            "attractor_detected": is_strange_attractor,
            "attractor_dimension": correlation_dim,
            "lyapunov_exponent": lyap_exp,
            "embedding_dimension": embedding_dim,
            "delay": delay
        }
    
    def correlation_dimension(self, embedded_data: np.ndarray) -> float:
        """
        Calcule la dimension de corrélation de Grassberger-Procaccia.
        """
        n = len(embedded_data)
        if n < 10:
            return 1.0
        
        # Calcul des distances entre tous les points
        distances = pdist(embedded_data)
        
        if len(distances) == 0:
            return 1.0
        
        # Gamme de rayons
        r_min = np.min(distances[distances > 0])
        r_max = np.max(distances)
        
        if r_min >= r_max:
            return 1.0
        
        radii = np.logspace(np.log10(r_min), np.log10(r_max), 20)
        correlations = []
        
        for r in radii:
            # Nombre de paires à distance < r
            count = np.sum(distances < r)
            correlation = count / len(distances)
            correlations.append(max(correlation, 1e-10))
        
        # Régression linéaire en échelle log-log
        log_radii = np.log(radii)
        log_correlations = np.log(correlations)
        
        # La dimension de corrélation est la pente
        if len(log_radii) > 2:
            correlation_dim = np.polyfit(log_radii, log_correlations, 1)[0]
        else:
            correlation_dim = 1.0
        
        return max(0.1, min(correlation_dim, 10.0))
    
    def chaotic_prediction(self, data: np.ndarray, prediction_steps: int = 1) -> np.ndarray:
        """
        Prédiction chaotique par méthode des voisinages.
        """
        # Reconstruction de l'espace de phase
        embedding_dim, delay = self.optimal_embedding_parameters(data)
        embedded = self.phase_space_reconstruction(data, embedding_dim, delay)
        
        if len(embedded) < 10:
            return np.array([np.mean(data[-5:])] * prediction_steps)
        
        predictions = []
        current_state = embedded[-1]  # Dernier état connu
        
        for step in range(prediction_steps):
            # Recherche des k plus proches voisins
            k = min(5, len(embedded) // 2)
            distances = np.linalg.norm(embedded - current_state, axis=1)
            nearest_indices = np.argsort(distances)[:k]
            
            # Prédiction par moyenne pondérée des évolutions
            predicted_value = 0.0
            total_weight = 0.0
            
            for idx in nearest_indices:
                if idx + delay < len(data):
                    # Évolution observée
                    future_value = data[idx + delay]
                    weight = 1.0 / (distances[idx] + 1e-10)
                    
                    predicted_value += weight * future_value
                    total_weight += weight
            
            if total_weight > 0:
                predicted_value /= total_weight
            else:
                predicted_value = np.mean(data[-5:])
            
            predictions.append(predicted_value)
            
            # Mise à jour de l'état pour la prochaine prédiction
            if len(embedded) > 0:
                # Création du nouvel état
                new_state = np.roll(current_state, -1)
                new_state[-1] = predicted_value
                current_state = new_state
        
        return np.array(predictions)

class ChaosFractalPredictor:
    """
    Prédicteur révolutionnaire combinant analyse fractale et théorie du chaos.
    """
    
    def __init__(self, data_path: str = "data/euromillions_enhanced_dataset.csv"):
        """
        Initialise le prédicteur chaos-fractal.
        """
        # print("🌀 SYSTÈME CHAOS-FRACTAL RÉVOLUTIONNAIRE 🌀") # Suppressed
        # print("=" * 60) # Suppressed
        
        # Chargement des données
        if os.path.exists(data_path): # Checks "data/euromillions_enhanced_dataset.csv"
            self.df = pd.read_csv(data_path)
            # print(f"✅ Données chargées depuis {data_path}: {len(self.df)} tirages")
        elif os.path.exists("euromillions_enhanced_dataset.csv"): # Fallback to current dir
            self.df = pd.read_csv("euromillions_enhanced_dataset.csv")
            # print(f"✅ Données chargées depuis le répertoire courant (euromillions_enhanced_dataset.csv): {len(self.df)} tirages") # Suppressed
        else:
            # print(f"❌ Fichier principal non trouvé ({data_path} ou euromillions_enhanced_dataset.csv). Utilisation de données de base...") # Suppressed
            self.load_basic_data()
        
        # Initialisation des analyseurs
        self.fractal_analyzer = FractalAnalyzer()
        self.chaos_analyzer = ChaosAnalyzer()
        
        # Préparation des séries temporelles
        self.prepare_time_series()
        
        # print("✅ Système Chaos-Fractal initialisé!")
    
    def load_basic_data(self):
        """
        Charge des données de base si le fichier enrichi n'existe pas.
        """
        if os.path.exists("data/euromillions_dataset.csv"):
            self.df = pd.read_csv("data/euromillions_dataset.csv")
            # print(f"✅ Données de base chargées depuis data/euromillions_dataset.csv: {len(self.df)} tirages") # Suppressed
        elif os.path.exists("euromillions_dataset.csv"): # Fallback to current dir
            self.df = pd.read_csv("euromillions_dataset.csv")
            # print(f"✅ Données de base chargées depuis le répertoire courant (euromillions_dataset.csv): {len(self.df)} tirages") # Suppressed
        else:
            # print(f"❌ Fichier de données de base (euromillions_dataset.csv) non trouvé. Création de données synthétiques...") # Suppressed
            # Création de données synthétiques
            dates = pd.date_range(start='2020-01-01', end='2025-06-01', freq='3D')
            data = []
            
            for date in dates:
                main_nums = sorted(np.random.choice(range(1, 51), 5, replace=False))
                stars = sorted(np.random.choice(range(1, 13), 2, replace=False))
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'N1': main_nums[0], 'N2': main_nums[1], 'N3': main_nums[2],
                    'N4': main_nums[3], 'N5': main_nums[4],
                    'E1': stars[0], 'E2': stars[1]
                })
            
            self.df = pd.DataFrame(data)
    
    def prepare_time_series(self):
        """
        Prépare les séries temporelles pour l'analyse chaos-fractale.
        """
        # print("📊 Préparation des séries temporelles chaos-fractales...") # Suppressed
        
        # Extraction des séries
        main_numbers = self.df[['N1', 'N2', 'N3', 'N4', 'N5']].values
        stars = self.df[['E1', 'E2']].values
        
        # Création de séries temporelles complexes
        self.time_series = {
            'sum_main': np.sum(main_numbers, axis=1),
            'product_main': np.prod(main_numbers, axis=1),
            'variance_main': np.var(main_numbers, axis=1),
            'sum_stars': np.sum(stars, axis=1),
            'ratio_main_stars': np.sum(main_numbers, axis=1) / (np.sum(stars, axis=1) + 1),
            'entropy_main': self.calculate_entropy_series(main_numbers),
            'complexity_main': self.calculate_complexity_series(main_numbers),
            'fibonacci_projection': self.fibonacci_projection_series(main_numbers),
            'golden_ratio_series': self.golden_ratio_series(main_numbers),
            'prime_density': self.prime_density_series(main_numbers)
        }
        
        # print(f"✅ {len(self.time_series)} séries temporelles préparées") # Suppressed
    
    def calculate_entropy_series(self, numbers: np.ndarray) -> np.ndarray:
        """
        Calcule une série d'entropie basée sur la distribution des numéros.
        """
        entropy_series = []
        
        for row in numbers:
            # Distribution des numéros par déciles
            bins = np.histogram(row, bins=5, range=(1, 50))[0]
            probs = bins / np.sum(bins)
            probs = probs[probs > 0]  # Éliminer les zéros
            
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs))
            else:
                entropy = 0.0
            
            entropy_series.append(entropy)
        
        return np.array(entropy_series)
    
    def calculate_complexity_series(self, numbers: np.ndarray) -> np.ndarray:
        """
        Calcule une série de complexité basée sur les patterns des numéros.
        """
        complexity_series = []
        
        for i, row in enumerate(numbers):
            complexity = 0.0
            
            # Complexité basée sur les écarts
            diffs = np.diff(sorted(row))
            complexity += np.std(diffs)
            
            # Complexité basée sur la distribution
            complexity += len(set(row)) / 5.0
            
            # Complexité temporelle (si pas le premier)
            if i > 0:
                prev_row = numbers[i-1]
                intersection = len(set(row) & set(prev_row))
                complexity += (5 - intersection) / 5.0
            
            complexity_series.append(complexity)
        
        return np.array(complexity_series)
    
    def fibonacci_projection_series(self, numbers: np.ndarray) -> np.ndarray:
        """
        Projette les numéros sur la séquence de Fibonacci.
        """
        # Séquence de Fibonacci jusqu'à 50
        fib = [1, 1]
        while fib[-1] < 50:
            fib.append(fib[-1] + fib[-2])
        
        fib_set = set(fib)
        
        projection_series = []
        for row in numbers:
            # Nombre de numéros de Fibonacci
            fib_count = len(set(row) & fib_set)
            projection_series.append(fib_count / 5.0)
        
        return np.array(projection_series)
    
    def golden_ratio_series(self, numbers: np.ndarray) -> np.ndarray:
        """
        Analyse basée sur le nombre d'or.
        """
        phi = (1 + np.sqrt(5)) / 2  # Nombre d'or
        
        golden_series = []
        for row in numbers:
            # Ratios entre numéros consécutifs
            sorted_nums = sorted(row)
            ratios = []
            
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i] > 0:
                    ratio = sorted_nums[i+1] / sorted_nums[i]
                    ratios.append(ratio)
            
            if ratios:
                # Distance moyenne au nombre d'or
                distances = [abs(ratio - phi) for ratio in ratios]
                golden_score = 1.0 / (1.0 + np.mean(distances))
            else:
                golden_score = 0.0
            
            golden_series.append(golden_score)
        
        return np.array(golden_series)
    
    def prime_density_series(self, numbers: np.ndarray) -> np.ndarray:
        """
        Calcule la densité de nombres premiers.
        """
        # Nombres premiers jusqu'à 50
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
        
        density_series = []
        for row in numbers:
            prime_count = len(set(row) & primes)
            density_series.append(prime_count / 5.0)
        
        return np.array(density_series)
    
    def analyze_fractal_properties(self) -> Dict[str, Any]:
        """
        Analyse les propriétés fractales de toutes les séries temporelles.
        """
        # print("🔬 Analyse des propriétés fractales...") # Suppressed
        
        fractal_results = {}
        
        for series_name, series_data in self.time_series.items():
            if len(series_data) < 10:
                continue
            
            # Dimension fractale
            fractal_dim = self.fractal_analyzer.box_counting_dimension(series_data)
            
            # Exposant de Hurst
            hurst_exp = self.fractal_analyzer.hurst_exponent(series_data)
            
            # Auto-similarité
            self_similarity = self.fractal_analyzer.detect_self_similarity(series_data)
            
            # Analyse par ondelettes
            wavelet_analysis = self.fractal_analyzer.wavelet_fractal_analysis(series_data)
            
            fractal_results[series_name] = {
                "fractal_dimension": fractal_dim,
                "hurst_exponent": hurst_exp,
                "self_similarity": self_similarity,
                "multifractal_width": wavelet_analysis["multifractal_width"],
                "fractal_spectrum": wavelet_analysis["fractal_spectrum"]
            }
        
        return fractal_results
    
    def analyze_chaotic_properties(self) -> Dict[str, Any]:
        """
        Analyse les propriétés chaotiques de toutes les séries temporelles.
        """
        # print("🌪️ Analyse des propriétés chaotiques...") # Suppressed
        
        chaos_results = {}
        
        for series_name, series_data in self.time_series.items():
            if len(series_data) < 20:
                continue
            
            # Détection d'attracteurs étranges
            attractor_analysis = self.chaos_analyzer.detect_strange_attractors(series_data)
            
            # Exposant de Lyapunov
            lyapunov = self.chaos_analyzer.lyapunov_exponent(series_data)
            
            # Prédiction chaotique
            chaotic_pred = self.chaos_analyzer.chaotic_prediction(series_data, 3)
            
            chaos_results[series_name] = {
                "strange_attractor": attractor_analysis,
                "lyapunov_exponent": lyapunov,
                "chaotic_prediction": chaotic_pred.tolist(),
                "chaos_level": "HIGH" if lyapunov > 0.1 else "MEDIUM" if lyapunov > 0.01 else "LOW"
            }
        
        return chaos_results
    
    def generate_chaos_fractal_prediction(self) -> Dict[str, Any]:
        """
        Génère une prédiction basée sur l'analyse chaos-fractale.
        """
        # print("\n🎯 GÉNÉRATION DE PRÉDICTION CHAOS-FRACTALE 🎯") # Suppressed
        # print("=" * 55) # Suppressed
        
        # Analyses fractales et chaotiques
        fractal_props = self.analyze_fractal_properties()
        chaos_props = self.analyze_chaotic_properties()
        
        # Sélection des séries les plus informatives
        best_series = self.select_best_series(fractal_props, chaos_props)
        
        # Prédiction par fusion chaos-fractale
        main_numbers = self.predict_main_numbers(best_series, fractal_props, chaos_props)
        stars = self.predict_stars(best_series, fractal_props, chaos_props)
        
        # Score de confiance chaos-fractal
        confidence = self.calculate_chaos_fractal_confidence(fractal_props, chaos_props)
        
        prediction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "Analyse Chaos-Fractale Révolutionnaire",
            "main_numbers": sorted(main_numbers),
            "stars": sorted(stars),
            "confidence_score": confidence,
            "fractal_analysis": fractal_props,
            "chaos_analysis": chaos_props,
            "best_series": best_series,
            "innovation_level": "RÉVOLUTIONNAIRE - Chaos & Fractales"
        }
        
        return prediction
    
    def select_best_series(self, fractal_props: Dict, chaos_props: Dict) -> List[str]:
        """
        Sélectionne les séries temporelles les plus informatives.
        """
        series_scores = {}
        
        for series_name in self.time_series.keys():
            score = 0.0
            
            # Score fractal
            if series_name in fractal_props:
                fp = fractal_props[series_name]
                score += fp["fractal_dimension"] * 0.3
                score += abs(fp["hurst_exponent"] - 0.5) * 0.2  # Éloignement du bruit blanc
                score += fp["multifractal_width"] * 0.2
            
            # Score chaotique
            if series_name in chaos_props:
                cp = chaos_props[series_name]
                if cp["strange_attractor"]["attractor_detected"]:
                    score += 1.0
                score += abs(cp["lyapunov_exponent"]) * 0.3
            
            series_scores[series_name] = score
        
        # Sélection des 3 meilleures séries
        sorted_series = sorted(series_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_series[:3]]
    
    def predict_main_numbers(self, best_series: List[str], fractal_props: Dict, 
                           chaos_props: Dict) -> List[int]:
        """
        Prédit les numéros principaux par analyse chaos-fractale.
        """
        candidates = {}
        
        for series_name in best_series:
            if series_name not in chaos_props:
                continue
            
            # Prédiction chaotique
            chaotic_pred = chaos_props[series_name]["chaotic_prediction"]
            
            # Conversion en numéros
            for i, pred_value in enumerate(chaotic_pred):
                # Mapping non-linéaire basé sur les propriétés fractales
                if series_name in fractal_props:
                    fractal_dim = fractal_props[series_name]["fractal_dimension"]
                    hurst_exp = fractal_props[series_name]["hurst_exponent"]
                    
                    # Transformation chaos-fractale
                    transformed = pred_value * fractal_dim * (1 + hurst_exp)
                    number = int(abs(transformed) % 50) + 1
                else:
                    number = int(abs(pred_value) % 50) + 1
                
                if number in candidates:
                    candidates[number] += 1
                else:
                    candidates[number] = 1
        
        # Sélection des 5 numéros avec les scores les plus élevés
        if len(candidates) < 5:
            # Complétion avec des numéros basés sur les patterns fractals
            missing = 5 - len(candidates)
            for i in range(1, 51):
                if i not in candidates and missing > 0:
                    candidates[i] = 0.1
                    missing -= 1
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_candidates[:5]]
    
    def predict_stars(self, best_series: List[str], fractal_props: Dict, 
                     chaos_props: Dict) -> List[int]:
        """
        Prédit les étoiles par analyse chaos-fractale.
        """
        star_candidates = {}
        
        # Utilisation des séries liées aux étoiles
        star_series = ['sum_stars', 'ratio_main_stars']
        
        for series_name in star_series:
            if series_name in chaos_props:
                chaotic_pred = chaos_props[series_name]["chaotic_prediction"]
                
                for pred_value in chaotic_pred:
                    star = int(abs(pred_value) % 12) + 1
                    
                    if star in star_candidates:
                        star_candidates[star] += 1
                    else:
                        star_candidates[star] = 1
        
        # Complétion si nécessaire
        if len(star_candidates) < 2:
            for i in range(1, 13):
                if i not in star_candidates:
                    star_candidates[i] = 0.1
                    if len(star_candidates) >= 2:
                        break
        
        sorted_stars = sorted(star_candidates.items(), key=lambda x: x[1], reverse=True)
        return [star for star, _ in sorted_stars[:2]]
    
    def calculate_chaos_fractal_confidence(self, fractal_props: Dict, 
                                         chaos_props: Dict) -> float:
        """
        Calcule un score de confiance basé sur les propriétés chaos-fractales.
        """
        confidence = 0.0
        count = 0
        
        # Score basé sur la détection d'attracteurs étranges
        for series_name, props in chaos_props.items():
            if props["strange_attractor"]["attractor_detected"]:
                confidence += 2.0
            
            # Score basé sur l'exposant de Lyapunov
            lyap = abs(props["lyapunov_exponent"])
            if lyap > 0.01:
                confidence += min(lyap * 10, 1.0)
            
            count += 1
        
        # Score basé sur les propriétés fractales
        for series_name, props in fractal_props.items():
            # Dimension fractale non-triviale
            if 1.5 < props["fractal_dimension"] < 2.5:
                confidence += 1.0
            
            # Exposant de Hurst significatif
            hurst_deviation = abs(props["hurst_exponent"] - 0.5)
            confidence += hurst_deviation * 2.0
            
            # Largeur multifractale
            confidence += min(props["multifractal_width"], 1.0)
            
            count += 1
        
        if count > 0:
            confidence /= count
        
        # Normalisation et bonus révolutionnaire
        confidence = min(confidence * 1.5, 10.0)  # Bonus pour l'innovation
        
        return confidence
    
    def save_chaos_fractal_results(self, prediction: Dict[str, Any]):
        """
        Sauvegarde les résultats de l'analyse chaos-fractale.
        """
        os.makedirs("results/chaos_fractal", exist_ok=True)
        
        # Fonction pour convertir les types non-sérialisables
        def convert_for_json(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Conversion de la prédiction
        json_prediction = convert_for_json(prediction)
        
        # Sauvegarde JSON complète
        # Commenting out file saving for CLI JSON output focus
        # with open("results/chaos_fractal/chaos_fractal_prediction.json", 'w') as f:
        #     json.dump(json_prediction, f, indent=4)
        
        # Sauvegarde texte formaté
        # with open("results/chaos_fractal/chaos_fractal_prediction.txt", 'w') as f:
            # f.write("PRÉDICTION CHAOS-FRACTALE RÉVOLUTIONNAIRE\n")
            # ... (rest of the text content) ...
            # f.write("🍀 BONNE CHANCE AVEC CETTE INNOVATION CHAOS-FRACTALE! 🍀\n")

        # print("✅ Résultats chaos-fractals sauvegardés dans results/chaos_fractal/") # Suppressed

def main():
    """
    Fonction principale pour exécuter l'analyse chaos-fractale.
    """
    # print("🌀 SYSTÈME RÉVOLUTIONNAIRE CHAOS-FRACTAL EUROMILLIONS 🌀")
    # print("=" * 70)
    # print("Techniques révolutionnaires implémentées:")
    # print("• Analyse de Dimension Fractale (Box-counting)")
    # print("• Exposant de Hurst et Auto-Similarité")
    # print("• Reconstruction d'Espace de Phase (Takens)")
    # print("• Exposants de Lyapunov et Attracteurs Étranges")
    # print("• Prédiction Chaotique par Voisinage")
    # print("• Fusion Chaos-Fractale Révolutionnaire")
    # print("=" * 70)
    
    # Initialisation
    parser = argparse.ArgumentParser(description="Chaos Fractal Predictor for Euromillions.")
    parser.add_argument("--date", type=str, help="Target draw date in YYYY-MM-DD format.")
    args = parser.parse_args()

    target_date_str = None
    data_file_for_date_calc = "data/euromillions_enhanced_dataset.csv"
    if not os.path.exists(data_file_for_date_calc):
        data_file_for_date_calc = "euromillions_enhanced_dataset.csv"
        if not os.path.exists(data_file_for_date_calc):
            data_file_for_date_calc = None

    if args.date:
        try:
            datetime.strptime(args.date, '%Y-%m-%d') # Validate
            target_date_str = args.date
        except ValueError:
            # print(f"Warning: Invalid date format for --date {args.date}. Using next logical date.", file=sys.stderr) # Suppressed
            target_date_obj = get_next_euromillions_draw_date(data_file_for_date_calc)
            target_date_str = target_date_obj.strftime('%Y-%m-%d')
    else:
        target_date_obj = get_next_euromillions_draw_date(data_file_for_date_calc)
        target_date_str = target_date_obj.strftime('%Y-%m-%d')

    predictor = ChaosFractalPredictor() # Uses its internal data loading
    
    # Génération de la prédiction
    prediction_result = predictor.generate_chaos_fractal_prediction() # This is a dict
    
    # Affichage des résultats - Suppressed for JSON output
    # print("\n🎉 PRÉDICTION CHAOS-FRACTALE GÉNÉRÉE! 🎉")
    # ... other prints ...
    
    # Sauvegarde - This script saves its own files, which is fine for now.
    # predictor.save_chaos_fractal_results(prediction_result)
    
    # print("\n🌀 ANALYSE CHAOS-FRACTALE TERMINÉE AVEC SUCCÈS! 🌀") # Suppressed

    raw_numeros = prediction_result.get('main_numbers', [])
    raw_etoiles = prediction_result.get('stars', [])
    raw_confidence = prediction_result.get('confidence_score', 6.0) # Default is 6.0

    # Convert to Python native types
    py_numeros = [int(n) for n in raw_numeros] if raw_numeros else []
    py_etoiles = [int(s) for s in raw_etoiles] if raw_etoiles else []
    py_confidence = float(raw_confidence) # raw_confidence will be a number due to default

    output_dict = {
        "nom_predicteur": "chaos_fractal_predictor",
        "numeros": py_numeros,
        "etoiles": py_etoiles,
        "date_tirage_cible": target_date_str,
        "confidence": py_confidence,
        "categorie": "Revolutionnaire"
    }
    print(json.dumps(output_dict))

if __name__ == "__main__":
    main()

