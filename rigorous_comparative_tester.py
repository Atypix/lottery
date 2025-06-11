#!/usr/bin/env python3
"""
Testeur Comparatif Rigoureux - Tous les Systèmes
===============================================

Tests comparatifs rigoureux de tous les systèmes développés pour identifier
les meilleures approches et performances réelles.

Objectif: Tester et comparer objectivement tous les systèmes pour déterminer
les plus performants et tirer les enseignements.

Auteur: IA Manus - Tests Comparatifs
Date: Juin 2025
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import importlib.util
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class RigorousComparativeTester:
    """
    Testeur comparatif rigoureux pour tous les systèmes.
    """
    
    def __init__(self):
        print("🧪 TESTS COMPARATIFS RIGOUREUX DE TOUS LES SYSTÈMES 🧪")
        print("=" * 70)
        print("Objectif: Comparer objectivement toutes les approches développées")
        print("Méthode: Tests standardisés avec tirage de référence")
        print("=" * 70)
        
        self.setup_testing_environment()
        self.reference_draw = {
            'numbers': [20, 21, 29, 30, 35],
            'stars': [2, 12],
            'date': '2025-06-06'
        }
        
        self.test_results = {}
        self.performance_metrics = {}
        
    def setup_testing_environment(self):
        """Configure l'environnement de test."""
        self.test_dir = '/home/ubuntu/results/comparative_testing'
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(f'{self.test_dir}/individual_results', exist_ok=True)
        os.makedirs(f'{self.test_dir}/performance_analysis', exist_ok=True)
        os.makedirs(f'{self.test_dir}/visualizations', exist_ok=True)
        
        # Chargement des données de référence
        self.df = pd.read_csv('/home/ubuntu/euromillions_enhanced_dataset.csv')
        
        print("✅ Environnement de test configuré")
        print(f"✅ {len(self.df)} tirages de référence chargés")
        
    def identify_testable_systems(self):
        """Identifie les systèmes testables (avec fonction de prédiction)."""
        print("🔍 Identification des systèmes testables...")
        
        # Chargement de l'inventaire
        with open('/home/ubuntu/results/comprehensive_analysis/systems_inventory/complete_inventory.json', 'r') as f:
            inventory = json.load(f)
        
        testable_systems = []
        
        for system in inventory['detailed_systems']:
            file_path = f"/home/ubuntu/{system['name']}"
            
            # Vérification que le fichier existe et est un prédicteur
            if (os.path.exists(file_path) and 
                system['type'] in ['Predictor', 'Model', 'Unknown'] and
                'predictor' in system['name'].lower() or 'model' in system['name'].lower()):
                
                testable_systems.append({
                    'name': system['name'],
                    'file_path': file_path,
                    'technologies': system['technologies'],
                    'description': system['description']
                })
        
        print(f"✅ {len(testable_systems)} systèmes testables identifiés")
        return testable_systems
        
    def load_and_test_system(self, system_info):
        """Charge et teste un système spécifique."""
        system_name = system_info['name']
        file_path = system_info['file_path']
        
        print(f"🧪 Test de {system_name}...")
        
        test_result = {
            'system_name': system_name,
            'technologies': system_info['technologies'],
            'test_status': 'FAILED',
            'error_message': None,
            'prediction': None,
            'matches': 0,
            'accuracy_percentage': 0.0,
            'execution_time': 0.0,
            'confidence_score': 0.0
        }
        
        try:
            start_time = datetime.now()
            
            # Tentative d'exécution du système
            if self.execute_system_safely(file_path, system_name):
                test_result['test_status'] = 'SUCCESS'
                
                # Recherche de la prédiction générée
                prediction = self.extract_prediction_from_results(system_name)
                if prediction:
                    test_result['prediction'] = prediction
                    
                    # Calcul des correspondances
                    matches = self.calculate_matches(prediction, self.reference_draw)
                    test_result['matches'] = matches['total']
                    test_result['accuracy_percentage'] = (matches['total'] / 7) * 100
                    
                    # Extraction du score de confiance si disponible
                    confidence = self.extract_confidence_score(system_name)
                    if confidence:
                        test_result['confidence_score'] = confidence
            
            end_time = datetime.now()
            test_result['execution_time'] = (end_time - start_time).total_seconds()
            
        except Exception as e:
            test_result['error_message'] = str(e)
            test_result['test_status'] = 'ERROR'
        
        return test_result
        
    def execute_system_safely(self, file_path, system_name):
        """Exécute un système de manière sécurisée."""
        try:
            # Lecture du contenu du fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Vérification que c'est un script exécutable
            if '__main__' not in content:
                return False
            
            # Exécution via subprocess pour isolation
            import subprocess
            result = subprocess.run([
                'python3', file_path
            ], capture_output=True, text=True, timeout=60, cwd='/home/ubuntu')
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout pour {system_name}")
            return False
        except Exception as e:
            print(f"❌ Erreur d'exécution pour {system_name}: {e}")
            return False
    
    def extract_prediction_from_results(self, system_name):
        """Extrait la prédiction depuis les fichiers de résultats."""
        
        # Recherche dans différents emplacements possibles
        search_paths = [
            f'/home/ubuntu/results/**/*{system_name.replace(".py", "")}*.json',
            f'/home/ubuntu/results/**/*{system_name.replace(".py", "")}*.txt',
            f'/home/ubuntu/*{system_name.replace(".py", "")}*.json',
            f'/home/ubuntu/*{system_name.replace(".py", "")}*.txt',
            '/home/ubuntu/prediction*.json',
            '/home/ubuntu/ticket*.txt'
        ]
        
        import glob
        
        for pattern in search_paths:
            files = glob.glob(pattern, recursive=True)
            for file_path in files:
                try:
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Recherche de prédiction dans le JSON
                        if isinstance(data, dict):
                            if 'numbers' in data and 'stars' in data:
                                return {
                                    'numbers': data['numbers'],
                                    'stars': data['stars']
                                }
                            elif 'prediction' in data:
                                pred = data['prediction']
                                if isinstance(pred, dict) and 'numbers' in pred:
                                    return pred
                    
                    elif file_path.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extraction par regex
                        import re
                        
                        # Pattern pour numéros
                        number_patterns = [
                            r'numéros?[:\s]*([0-9, ]+)',
                            r'numbers?[:\s]*([0-9, ]+)',
                            r'([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)'
                        ]
                        
                        # Pattern pour étoiles
                        star_patterns = [
                            r'étoiles?[:\s]*([0-9, ]+)',
                            r'stars?[:\s]*([0-9, ]+)'
                        ]
                        
                        numbers = None
                        stars = None
                        
                        for pattern in number_patterns:
                            match = re.search(pattern, content, re.IGNORECASE)
                            if match:
                                if len(match.groups()) == 5:
                                    numbers = [int(match.group(i)) for i in range(1, 6)]
                                else:
                                    numbers_str = match.group(1)
                                    numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
                                    if len(numbers) == 5:
                                        break
                        
                        for pattern in star_patterns:
                            match = re.search(pattern, content, re.IGNORECASE)
                            if match:
                                stars_str = match.group(1)
                                stars = [int(x.strip()) for x in stars_str.split(',') if x.strip().isdigit()]
                                if len(stars) == 2:
                                    break
                        
                        if numbers and len(numbers) == 5 and stars and len(stars) == 2:
                            return {
                                'numbers': sorted(numbers),
                                'stars': sorted(stars)
                            }
                
                except Exception:
                    continue
        
        return None
    
    def extract_confidence_score(self, system_name):
        """Extrait le score de confiance depuis les résultats."""
        
        # Recherche dans les fichiers de performance
        performance_file = f'/home/ubuntu/results/comprehensive_analysis/systems_inventory/complete_inventory.json'
        
        try:
            with open(performance_file, 'r') as f:
                inventory = json.load(f)
            
            # Recherche dans les données de performance
            if 'performance_summary' in inventory:
                for name, metrics in inventory['performance_summary'].items():
                    if system_name.replace('.py', '') in name:
                        if 'confiance' in metrics:
                            return float(metrics['confiance'])
                        elif 'confidence' in metrics:
                            return float(metrics['confidence'])
                        elif 'score' in metrics:
                            return float(metrics['score'])
        
        except Exception:
            pass
        
        return 0.0
    
    def calculate_matches(self, prediction, reference):
        """Calcule les correspondances avec le tirage de référence."""
        
        if not prediction or 'numbers' not in prediction or 'stars' not in prediction:
            return {'numbers': 0, 'stars': 0, 'total': 0}
        
        pred_numbers = set(prediction['numbers'])
        pred_stars = set(prediction['stars'])
        
        ref_numbers = set(reference['numbers'])
        ref_stars = set(reference['stars'])
        
        number_matches = len(pred_numbers.intersection(ref_numbers))
        star_matches = len(pred_stars.intersection(ref_stars))
        
        return {
            'numbers': number_matches,
            'stars': star_matches,
            'total': number_matches + star_matches
        }
    
    def run_comprehensive_testing(self):
        """Exécute les tests complets sur tous les systèmes."""
        print("🚀 LANCEMENT DES TESTS COMPARATIFS COMPLETS 🚀")
        print("=" * 70)
        
        # 1. Identification des systèmes testables
        testable_systems = self.identify_testable_systems()
        
        # 2. Tests individuels
        print("🧪 Phase de tests individuels...")
        all_results = []
        
        for i, system_info in enumerate(testable_systems):
            print(f"📊 Test {i+1}/{len(testable_systems)}: {system_info['name']}")
            
            result = self.load_and_test_system(system_info)
            all_results.append(result)
            
            # Sauvegarde du résultat individuel
            with open(f"{self.test_dir}/individual_results/{result['system_name'].replace('.py', '')}_result.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # 3. Analyse comparative
        print("📊 Phase d'analyse comparative...")
        comparative_analysis = self.analyze_comparative_results(all_results)
        
        # 4. Génération des visualisations
        print("📈 Phase de visualisation...")
        self.generate_comparative_visualizations(all_results)
        
        # 5. Sauvegarde des résultats globaux
        print("💾 Sauvegarde des résultats...")
        self.save_comprehensive_results(all_results, comparative_analysis)
        
        print("✅ TESTS COMPARATIFS TERMINÉS!")
        
        return {
            'individual_results': all_results,
            'comparative_analysis': comparative_analysis
        }
    
    def analyze_comparative_results(self, results):
        """Analyse comparative des résultats de tous les systèmes."""
        
        # Filtrage des systèmes qui ont réussi
        successful_systems = [r for r in results if r['test_status'] == 'SUCCESS' and r['prediction']]
        
        if not successful_systems:
            return {'error': 'Aucun système n\'a réussi les tests'}
        
        # Classement par performance
        ranked_by_accuracy = sorted(successful_systems, key=lambda x: x['accuracy_percentage'], reverse=True)
        ranked_by_confidence = sorted(successful_systems, key=lambda x: x['confidence_score'], reverse=True)
        
        # Statistiques globales
        accuracies = [r['accuracy_percentage'] for r in successful_systems]
        confidences = [r['confidence_score'] for r in successful_systems if r['confidence_score'] > 0]
        execution_times = [r['execution_time'] for r in successful_systems]
        
        analysis = {
            'total_systems_tested': len(results),
            'successful_systems': len(successful_systems),
            'success_rate': (len(successful_systems) / len(results)) * 100,
            
            'performance_statistics': {
                'accuracy': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies),
                    'median': np.median(accuracies)
                },
                'confidence': {
                    'mean': np.mean(confidences) if confidences else 0,
                    'std': np.std(confidences) if confidences else 0,
                    'min': np.min(confidences) if confidences else 0,
                    'max': np.max(confidences) if confidences else 0,
                    'median': np.median(confidences) if confidences else 0
                },
                'execution_time': {
                    'mean': np.mean(execution_times),
                    'std': np.std(execution_times),
                    'min': np.min(execution_times),
                    'max': np.max(execution_times)
                }
            },
            
            'top_performers': {
                'by_accuracy': ranked_by_accuracy[:5],
                'by_confidence': ranked_by_confidence[:5]
            },
            
            'technology_analysis': self.analyze_technology_performance(successful_systems),
            
            'reference_draw_analysis': {
                'reference': self.reference_draw,
                'perfect_matches': [r for r in successful_systems if r['accuracy_percentage'] == 100.0],
                'high_performers': [r for r in successful_systems if r['accuracy_percentage'] >= 50.0]
            }
        }
        
        return analysis
    
    def analyze_technology_performance(self, successful_systems):
        """Analyse la performance par technologie."""
        
        tech_performance = {}
        
        for system in successful_systems:
            for tech in system['technologies']:
                if tech not in tech_performance:
                    tech_performance[tech] = {
                        'systems': [],
                        'accuracies': [],
                        'confidences': []
                    }
                
                tech_performance[tech]['systems'].append(system['system_name'])
                tech_performance[tech]['accuracies'].append(system['accuracy_percentage'])
                if system['confidence_score'] > 0:
                    tech_performance[tech]['confidences'].append(system['confidence_score'])
        
        # Calcul des moyennes par technologie
        tech_summary = {}
        for tech, data in tech_performance.items():
            tech_summary[tech] = {
                'system_count': len(data['systems']),
                'avg_accuracy': np.mean(data['accuracies']),
                'avg_confidence': np.mean(data['confidences']) if data['confidences'] else 0,
                'best_system': max(data['systems'], key=lambda s: next(sys['accuracy_percentage'] for sys in successful_systems if sys['system_name'] == s))
            }
        
        return tech_summary
    
    def generate_comparative_visualizations(self, results):
        """Génère les visualisations comparatives."""
        
        successful_systems = [r for r in results if r['test_status'] == 'SUCCESS' and r['prediction']]
        
        if not successful_systems:
            return
        
        # Configuration matplotlib
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analyse Comparative des Systèmes de Prédiction', fontsize=16, fontweight='bold')
        
        # 1. Graphique de précision
        names = [r['system_name'][:15] + '...' if len(r['system_name']) > 15 else r['system_name'] for r in successful_systems]
        accuracies = [r['accuracy_percentage'] for r in successful_systems]
        
        axes[0,0].bar(range(len(names)), accuracies, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Précision par Système (%)')
        axes[0,0].set_ylabel('Précision (%)')
        axes[0,0].set_xticks(range(len(names)))
        axes[0,0].set_xticklabels(names, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Distribution des correspondances
        matches = [r['matches'] for r in successful_systems]
        axes[0,1].hist(matches, bins=range(0, 8), alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Distribution des Correspondances')
        axes[0,1].set_xlabel('Nombre de Correspondances')
        axes[0,1].set_ylabel('Nombre de Systèmes')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Temps d'exécution
        exec_times = [r['execution_time'] for r in successful_systems]
        axes[1,0].scatter(accuracies, exec_times, alpha=0.6, color='orange')
        axes[1,0].set_title('Précision vs Temps d\'Exécution')
        axes[1,0].set_xlabel('Précision (%)')
        axes[1,0].set_ylabel('Temps d\'Exécution (s)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Scores de confiance
        confidences = [r['confidence_score'] for r in successful_systems if r['confidence_score'] > 0]
        conf_accuracies = [r['accuracy_percentage'] for r in successful_systems if r['confidence_score'] > 0]
        
        if confidences:
            axes[1,1].scatter(confidences, conf_accuracies, alpha=0.6, color='red')
            axes[1,1].set_title('Confiance vs Précision')
            axes[1,1].set_xlabel('Score de Confiance')
            axes[1,1].set_ylabel('Précision (%)')
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Pas de données\nde confiance', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Confiance vs Précision')
        
        plt.tight_layout()
        plt.savefig(f'{self.test_dir}/visualizations/comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualisations générées")
    
    def save_comprehensive_results(self, results, analysis):
        """Sauvegarde tous les résultats des tests."""
        
        # Résultats complets
        comprehensive_results = {
            'test_date': datetime.now().isoformat(),
            'reference_draw': self.reference_draw,
            'individual_results': results,
            'comparative_analysis': analysis
        }
        
        with open(f'{self.test_dir}/comprehensive_test_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Rapport de synthèse
        synthesis_report = self.generate_synthesis_report(analysis)
        
        with open(f'{self.test_dir}/test_synthesis_report.txt', 'w') as f:
            f.write(synthesis_report)
        
        print("✅ Résultats sauvegardés")
    
    def generate_synthesis_report(self, analysis):
        """Génère un rapport de synthèse des tests."""
        
        if 'error' in analysis:
            return f"ERREUR: {analysis['error']}"
        
        report = f"""
# RAPPORT DE SYNTHÈSE - TESTS COMPARATIFS RIGOUREUX

## RÉSUMÉ EXÉCUTIF
Date des tests: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Tirage de référence: {self.reference_draw['numbers']} + {self.reference_draw['stars']} ({self.reference_draw['date']})

## RÉSULTATS GLOBAUX
- Systèmes testés: {analysis['total_systems_tested']}
- Systèmes fonctionnels: {analysis['successful_systems']}
- Taux de succès: {analysis['success_rate']:.1f}%

## STATISTIQUES DE PERFORMANCE

### Précision (Correspondances avec tirage de référence)
- Moyenne: {analysis['performance_statistics']['accuracy']['mean']:.1f}%
- Médiane: {analysis['performance_statistics']['accuracy']['median']:.1f}%
- Minimum: {analysis['performance_statistics']['accuracy']['min']:.1f}%
- Maximum: {analysis['performance_statistics']['accuracy']['max']:.1f}%
- Écart-type: {analysis['performance_statistics']['accuracy']['std']:.1f}%

### Scores de Confiance
- Moyenne: {analysis['performance_statistics']['confidence']['mean']:.2f}
- Médiane: {analysis['performance_statistics']['confidence']['median']:.2f}
- Maximum: {analysis['performance_statistics']['confidence']['max']:.2f}

### Temps d'Exécution
- Moyenne: {analysis['performance_statistics']['execution_time']['mean']:.2f}s
- Minimum: {analysis['performance_statistics']['execution_time']['min']:.2f}s
- Maximum: {analysis['performance_statistics']['execution_time']['max']:.2f}s

## TOP PERFORMERS

### Par Précision (Top 5)
"""
        
        for i, system in enumerate(analysis['top_performers']['by_accuracy'][:5]):
            report += f"{i+1}. {system['system_name']} - {system['accuracy_percentage']:.1f}% ({system['matches']}/7 correspondances)\n"
        
        report += f"""
### Par Confiance (Top 5)
"""
        
        for i, system in enumerate(analysis['top_performers']['by_confidence'][:5]):
            report += f"{i+1}. {system['system_name']} - Score: {system['confidence_score']:.2f}\n"
        
        report += f"""
## ANALYSE PAR TECHNOLOGIE
"""
        
        for tech, data in analysis['technology_analysis'].items():
            report += f"- {tech}: {data['system_count']} systèmes, précision moyenne: {data['avg_accuracy']:.1f}%\n"
        
        report += f"""
## CORRESPONDANCES PARFAITES
Systèmes avec 100% de correspondances: {len(analysis['reference_draw_analysis']['perfect_matches'])}
"""
        
        for system in analysis['reference_draw_analysis']['perfect_matches']:
            report += f"- {system['system_name']}: {system['prediction']}\n"
        
        report += f"""
## HAUTES PERFORMANCES (≥50% correspondances)
Systèmes performants: {len(analysis['reference_draw_analysis']['high_performers'])}
"""
        
        for system in analysis['reference_draw_analysis']['high_performers']:
            report += f"- {system['system_name']}: {system['accuracy_percentage']:.1f}% ({system['matches']}/7)\n"
        
        report += f"""
---
Rapport généré automatiquement par le Testeur Comparatif Rigoureux
"""
        
        return report

if __name__ == "__main__":
    # Lancement des tests comparatifs rigoureux
    tester = RigorousComparativeTester()
    results = tester.run_comprehensive_testing()
    
    print(f"\n🧪 RÉSULTATS DES TESTS COMPARATIFS:")
    if 'error' not in results['comparative_analysis']:
        print(f"Systèmes testés: {results['comparative_analysis']['total_systems_tested']}")
        print(f"Systèmes fonctionnels: {results['comparative_analysis']['successful_systems']}")
        print(f"Taux de succès: {results['comparative_analysis']['success_rate']:.1f}%")
        
        if results['comparative_analysis']['reference_draw_analysis']['perfect_matches']:
            print(f"🏆 Correspondances parfaites: {len(results['comparative_analysis']['reference_draw_analysis']['perfect_matches'])}")
    
    print("\n🎉 TESTS COMPARATIFS TERMINÉS! 🎉")

