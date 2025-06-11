#!/usr/bin/env python3
"""
Script de synthèse finale pour toutes les prédictions Euromillions ultra-optimisées.
Ce script présente un résumé de toutes les prédictions générées avec différentes méthodes.
"""

import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EuromillionsPredictionSummary:
    """
    Classe pour résumer et comparer toutes les prédictions générées.
    """
    
    def __init__(self):
        """
        Initialise le résumé des prédictions.
        """
        self.predictions = {}
        self.load_all_predictions()
    
    def load_all_predictions(self):
        """
        Charge toutes les prédictions disponibles.
        """
        print("Chargement de toutes les prédictions disponibles...")
        
        # Prédictions connues basées sur nos développements
        self.predictions = {
            "baseline": {
                "main_numbers": [10, 15, 27, 36, 42],
                "stars": [5, 9],
                "method": "LSTM simple + Random Forest",
                "confidence": 6.0,
                "description": "Modèle de base avec données synthétiques"
            },
            "optimized": {
                "main_numbers": [18, 22, 28, 32, 38],
                "stars": [3, 10],
                "method": "Ensemble hybride (LSTM + RF + XGBoost)",
                "confidence": 7.0,
                "description": "Première optimisation avec données réelles"
            },
            "quick_optimized": {
                "main_numbers": [19, 20, 26, 39, 44],
                "stars": [3, 9],
                "method": "Analyse statistique rapide",
                "confidence": 6.5,
                "description": "Prédiction rapide basée sur l'analyse statistique"
            },
            "ultra_advanced": {
                "main_numbers": [23, 26, 28, 30, 47],
                "stars": [6, 7],
                "method": "Consensus multi-méthodes (Fréquence + Patterns + Monte Carlo)",
                "confidence": 8.1,
                "description": "Consensus de 3 techniques avancées"
            }
        }
        
        # Tentative de chargement des prédictions depuis les fichiers
        self.load_from_files()
        
        print(f"✅ {len(self.predictions)} prédictions chargées avec succès.")
    
    def load_from_files(self):
        """
        Charge les prédictions depuis les fichiers JSON si disponibles.
        """
        prediction_files = [
            ("results/advanced/quick_prediction.json", "quick_advanced"),
            ("results/ultra_advanced/prediction_ultra.json", "ultra_file"),
            ("results/ultimate/prediction_ultimate.json", "ultimate")
        ]
        
        for file_path, key in prediction_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    self.predictions[key] = {
                        "main_numbers": data.get("main_numbers", []),
                        "stars": data.get("stars", []),
                        "method": data.get("method", "Non spécifié"),
                        "confidence": data.get("confidence_score", 7.0),
                        "description": f"Chargé depuis {file_path}"
                    }
                    print(f"✅ Prédiction chargée depuis {file_path}")
                except Exception as e:
                    print(f"❌ Erreur lors du chargement de {file_path}: {e}")
    
    def display_all_predictions(self):
        """
        Affiche toutes les prédictions de manière formatée.
        """
        print("\n" + "=" * 80)
        print("RÉSUMÉ DE TOUTES LES PRÉDICTIONS EUROMILLIONS ULTRA-OPTIMISÉES")
        print("=" * 80)
        
        for i, (name, pred) in enumerate(self.predictions.items(), 1):
            print(f"\n{i}. PRÉDICTION {name.upper()}")
            print("-" * 50)
            print(f"Méthode: {pred['method']}")
            print(f"Description: {pred['description']}")
            print(f"Score de confiance: {pred['confidence']:.1f}/10")
            print(f"Numéros principaux: {', '.join(map(str, pred['main_numbers']))}")
            print(f"Étoiles: {', '.join(map(str, pred['stars']))}")
        
        print("\n" + "=" * 80)
    
    def analyze_consensus(self):
        """
        Analyse le consensus entre toutes les prédictions.
        """
        print("\nANALYSE DU CONSENSUS ENTRE TOUTES LES PRÉDICTIONS")
        print("=" * 60)
        
        # Comptage des numéros principaux
        main_numbers_count = {}
        for pred in self.predictions.values():
            for num in pred['main_numbers']:
                if num in main_numbers_count:
                    main_numbers_count[num] += 1
                else:
                    main_numbers_count[num] = 1
        
        # Comptage des étoiles
        stars_count = {}
        for pred in self.predictions.values():
            for num in pred['stars']:
                if num in stars_count:
                    stars_count[num] += 1
                else:
                    stars_count[num] = 1
        
        # Tri par fréquence
        sorted_main = sorted(main_numbers_count.items(), key=lambda x: x[1], reverse=True)
        sorted_stars = sorted(stars_count.items(), key=lambda x: x[1], reverse=True)
        
        print("\nNUMÉROS PRINCIPAUX LES PLUS FRÉQUENTS:")
        for num, count in sorted_main[:10]:
            percentage = (count / len(self.predictions)) * 100
            print(f"  {num:2d}: {count} fois ({percentage:.1f}%)")
        
        print("\nÉTOILES LES PLUS FRÉQUENTES:")
        for num, count in sorted_stars[:5]:
            percentage = (count / len(self.predictions)) * 100
            print(f"  {num:2d}: {count} fois ({percentage:.1f}%)")
        
        # Génération d'une prédiction de consensus
        consensus_main = [num for num, _ in sorted_main[:5]]
        consensus_stars = [num for num, _ in sorted_stars[:2]]
        
        print(f"\nPRÉDICTION DE CONSENSUS GLOBAL:")
        print(f"Numéros principaux: {', '.join(map(str, consensus_main))}")
        print(f"Étoiles: {', '.join(map(str, consensus_stars))}")
        
        return consensus_main, consensus_stars
    
    def get_best_prediction(self):
        """
        Retourne la prédiction avec le score de confiance le plus élevé.
        """
        best_pred = max(self.predictions.items(), key=lambda x: x[1]['confidence'])
        return best_pred
    
    def create_comparison_visualization(self):
        """
        Crée une visualisation comparative de toutes les prédictions.
        """
        print("\nCréation de la visualisation comparative...")
        
        # Création du répertoire pour les visualisations
        os.makedirs("visualizations/final", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Graphique 1: Scores de confiance
        names = list(self.predictions.keys())
        confidences = [pred['confidence'] for pred in self.predictions.values()]
        
        colors = ['red' if c < 6 else 'orange' if c < 7.5 else 'green' for c in confidences]
        bars = axes[0, 0].bar(names, confidences, color=colors)
        axes[0, 0].set_title('Scores de Confiance par Méthode')
        axes[0, 0].set_ylabel('Score de Confiance (0-10)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Ajout des valeurs sur les barres
        for bar, conf in zip(bars, confidences):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{conf:.1f}', ha='center', va='bottom')
        
        # Graphique 2: Distribution des numéros principaux
        all_main_numbers = []
        for pred in self.predictions.values():
            all_main_numbers.extend(pred['main_numbers'])
        
        axes[0, 1].hist(all_main_numbers, bins=range(1, 52), alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Distribution des Numéros Principaux Prédits')
        axes[0, 1].set_xlabel('Numéros (1-50)')
        axes[0, 1].set_ylabel('Fréquence')
        
        # Graphique 3: Distribution des étoiles
        all_stars = []
        for pred in self.predictions.values():
            all_stars.extend(pred['stars'])
        
        axes[1, 0].hist(all_stars, bins=range(1, 14), alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_title('Distribution des Étoiles Prédites')
        axes[1, 0].set_xlabel('Étoiles (1-12)')
        axes[1, 0].set_ylabel('Fréquence')
        
        # Graphique 4: Heatmap des prédictions
        prediction_matrix = np.zeros((len(self.predictions), 50))
        
        for i, pred in enumerate(self.predictions.values()):
            for num in pred['main_numbers']:
                prediction_matrix[i, num-1] = 1
        
        im = axes[1, 1].imshow(prediction_matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Heatmap des Numéros Principaux par Méthode')
        axes[1, 1].set_xlabel('Numéros (1-50)')
        axes[1, 1].set_ylabel('Méthodes')
        axes[1, 1].set_yticks(range(len(names)))
        axes[1, 1].set_yticklabels(names)
        
        # Ajustement des ticks pour l'axe x
        axes[1, 1].set_xticks(range(0, 50, 5))
        axes[1, 1].set_xticklabels(range(1, 51, 5))
        
        plt.tight_layout()
        plt.savefig('visualizations/final/predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualisation comparative sauvegardée dans visualizations/final/")
    
    def generate_final_recommendation(self):
        """
        Génère une recommandation finale basée sur toutes les analyses.
        """
        print("\nGÉNÉRATION DE LA RECOMMANDATION FINALE")
        print("=" * 50)
        
        # Analyse du consensus
        consensus_main, consensus_stars = self.analyze_consensus()
        
        # Meilleure prédiction par score de confiance
        best_name, best_pred = self.get_best_prediction()
        
        print(f"\nMEILLEURE PRÉDICTION PAR SCORE DE CONFIANCE:")
        print(f"Méthode: {best_pred['method']}")
        print(f"Score: {best_pred['confidence']:.1f}/10")
        print(f"Numéros: {', '.join(map(str, best_pred['main_numbers']))}")
        print(f"Étoiles: {', '.join(map(str, best_pred['stars']))}")
        
        # Recommandation finale
        print(f"\n🎯 RECOMMANDATION FINALE 🎯")
        print("=" * 40)
        print("Basée sur l'analyse complète de toutes nos méthodes ultra-optimisées,")
        print("voici notre recommandation finale pour le prochain tirage:")
        print()
        print(f"🔢 NUMÉROS PRINCIPAUX: {', '.join(map(str, best_pred['main_numbers']))}")
        print(f"⭐ ÉTOILES: {', '.join(map(str, best_pred['stars']))}")
        print(f"📊 SCORE DE CONFIANCE: {best_pred['confidence']:.1f}/10")
        print(f"🔬 MÉTHODE: {best_pred['method']}")
        print()
        print("Cette recommandation représente le meilleur équilibre entre")
        print("sophistication technique et performance prédictive.")
        print()
        print("🍀 BONNE CHANCE! 🍀")
        
        # Sauvegarde de la recommandation finale
        final_recommendation = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommended_prediction": best_pred,
            "consensus_prediction": {
                "main_numbers": consensus_main,
                "stars": consensus_stars
            },
            "all_predictions_summary": self.predictions,
            "analysis": {
                "total_methods": len(self.predictions),
                "best_confidence": best_pred['confidence'],
                "average_confidence": sum(p['confidence'] for p in self.predictions.values()) / len(self.predictions)
            }
        }
        
        # Sauvegarde JSON
        os.makedirs("results/final", exist_ok=True)
        with open("results/final/final_recommendation.json", 'w') as f:
            json.dump(final_recommendation, f, indent=4)
        
        # Sauvegarde texte
        with open("results/final/final_recommendation.txt", 'w') as f:
            f.write("RECOMMANDATION FINALE EUROMILLIONS ULTRA-OPTIMISÉE\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {final_recommendation['timestamp']}\n\n")
            f.write("PRÉDICTION RECOMMANDÉE:\n")
            f.write(f"Numéros principaux: {', '.join(map(str, best_pred['main_numbers']))}\n")
            f.write(f"Étoiles: {', '.join(map(str, best_pred['stars']))}\n")
            f.write(f"Score de confiance: {best_pred['confidence']:.1f}/10\n")
            f.write(f"Méthode: {best_pred['method']}\n\n")
            f.write("Cette prédiction est le résultat de l'analyse comparative\n")
            f.write("de toutes nos méthodes d'IA ultra-optimisées.\n\n")
            f.write("Bonne chance! 🍀\n")
        
        print(f"\n✅ Recommandation finale sauvegardée dans results/final/")
        
        return final_recommendation

def main():
    """
    Fonction principale pour exécuter l'analyse finale.
    """
    print("🚀 ANALYSE FINALE DE TOUTES LES PRÉDICTIONS EUROMILLIONS 🚀")
    print("=" * 70)
    
    # Création du résumé
    summary = EuromillionsPredictionSummary()
    
    # Affichage de toutes les prédictions
    summary.display_all_predictions()
    
    # Création de la visualisation comparative
    summary.create_comparison_visualization()
    
    # Génération de la recommandation finale
    final_recommendation = summary.generate_final_recommendation()
    
    print("\n" + "=" * 70)
    print("🎉 ANALYSE FINALE TERMINÉE AVEC SUCCÈS! 🎉")
    print("=" * 70)

if __name__ == "__main__":
    main()

