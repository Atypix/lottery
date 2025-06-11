#!/usr/bin/env python3
"""
Analyseur Rétrospectif Complet - Tous les Systèmes Développés
============================================================

Analyse exhaustive de tous les systèmes d'IA développés depuis la demande initiale
de création d'une IA TensorFlow pour prédire l'Euromillions.

Objectif: Inventorier, analyser et évaluer tous les systèmes pour tirer les enseignements
et identifier les meilleures approches.

Auteur: IA Manus - Analyse Rétrospective
Date: Juin 2025
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import re

class ComprehensiveSystemAnalyzer:
    """
    Analyseur complet de tous les systèmes développés.
    """
    
    def __init__(self):
        print("🔍 ANALYSE RÉTROSPECTIVE COMPLÈTE DE TOUS LES SYSTÈMES 🔍")
        print("=" * 70)
        print("Objectif: Inventorier et analyser tous les systèmes développés")
        print("Période: Depuis la demande initiale d'IA TensorFlow")
        print("=" * 70)
        
        self.setup_analysis_environment()
        self.systems_inventory = {}
        self.performance_data = {}
        self.evolution_timeline = []
        
    def setup_analysis_environment(self):
        """Configure l'environnement d'analyse."""
        self.analysis_dir = '/home/ubuntu/results/comprehensive_analysis'
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(f'{self.analysis_dir}/systems_inventory', exist_ok=True)
        os.makedirs(f'{self.analysis_dir}/performance_comparison', exist_ok=True)
        os.makedirs(f'{self.analysis_dir}/evolution_analysis', exist_ok=True)
        
        print("✅ Environnement d'analyse configuré")
        
    def scan_all_systems(self):
        """Scanne tous les fichiers pour identifier les systèmes développés."""
        print("🔍 Scan complet de tous les systèmes développés...")
        
        # Patterns de fichiers à analyser
        system_patterns = [
            '*predictor*.py',
            '*euromillions*.py', 
            '*model*.py',
            '*prediction*.py',
            '*optimizer*.py',
            '*validator*.py',
            '*analyzer*.py'
        ]
        
        systems_found = []
        
        # Scan du répertoire principal
        for pattern in system_patterns:
            files = glob.glob(f'/home/ubuntu/{pattern}')
            for file_path in files:
                if os.path.isfile(file_path):
                    systems_found.append(file_path)
        
        # Scan des sous-répertoires results
        results_files = glob.glob('/home/ubuntu/results/**/*.py', recursive=True)
        systems_found.extend(results_files)
        
        # Suppression des doublons
        systems_found = list(set(systems_found))
        
        print(f"✅ {len(systems_found)} fichiers système trouvés")
        
        return systems_found
        
    def analyze_system_file(self, file_path):
        """Analyse un fichier système pour extraire ses caractéristiques."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return None
        
        # Extraction des métadonnées
        system_info = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'creation_time': os.path.getctime(file_path),
            'modification_time': os.path.getmtime(file_path)
        }
        
        # Analyse du contenu
        lines = content.split('\n')
        system_info['line_count'] = len(lines)
        
        # Extraction de la description/docstring
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            system_info['description'] = docstring_match.group(1).strip()
        else:
            system_info['description'] = "Pas de description trouvée"
        
        # Identification des technologies utilisées
        technologies = []
        if 'tensorflow' in content.lower() or 'tf.' in content:
            technologies.append('TensorFlow')
        if 'sklearn' in content or 'scikit-learn' in content:
            technologies.append('Scikit-Learn')
        if 'optuna' in content.lower():
            technologies.append('Optuna')
        if 'bayesian' in content.lower():
            technologies.append('Bayesian')
        if 'neural' in content.lower() or 'mlp' in content.lower():
            technologies.append('Neural Networks')
        if 'random forest' in content.lower() or 'randomforest' in content:
            technologies.append('Random Forest')
        if 'gradient boost' in content.lower() or 'gradientboost' in content:
            technologies.append('Gradient Boosting')
        if 'ensemble' in content.lower():
            technologies.append('Ensemble')
        if 'quantum' in content.lower():
            technologies.append('Quantum')
        if 'genetic' in content.lower() or 'evolution' in content.lower():
            technologies.append('Genetic/Evolution')
        
        system_info['technologies'] = technologies
        
        # Identification du type de système
        system_type = "Unknown"
        if 'predictor' in file_path.lower():
            system_type = "Predictor"
        elif 'model' in file_path.lower():
            system_type = "Model"
        elif 'optimizer' in file_path.lower():
            system_type = "Optimizer"
        elif 'validator' in file_path.lower():
            system_type = "Validator"
        elif 'analyzer' in file_path.lower():
            system_type = "Analyzer"
        
        system_info['system_type'] = system_type
        
        # Recherche de métriques de performance
        performance_patterns = [
            r'score[:\s]*([0-9.]+)',
            r'accuracy[:\s]*([0-9.]+)',
            r'correspondance[s]?[:\s]*([0-9]+)/([0-9]+)',
            r'([0-9]+)/([0-9]+)\s*correspondance',
            r'confiance[:\s]*([0-9.]+)',
            r'r2[:\s]*([0-9.-]+)',
            r'mae[:\s]*([0-9.]+)'
        ]
        
        performance_metrics = {}
        for pattern in performance_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                performance_metrics[pattern] = matches
        
        system_info['performance_indicators'] = performance_metrics
        
        # Identification des prédictions générées
        prediction_patterns = [
            r'numéros?[:\s]*\[?([0-9, ]+)\]?',
            r'étoiles?[:\s]*\[?([0-9, ]+)\]?',
            r'numbers?[:\s]*\[?([0-9, ]+)\]?',
            r'stars?[:\s]*\[?([0-9, ]+)\]?'
        ]
        
        predictions_found = {}
        for pattern in prediction_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                predictions_found[pattern] = matches[:3]  # Limiter à 3 exemples
        
        system_info['predictions_found'] = predictions_found
        
        return system_info
        
    def categorize_systems(self, systems_data):
        """Catégorise les systèmes par approche et évolution."""
        print("📊 Catégorisation des systèmes...")
        
        categories = {
            'tensorflow_based': [],
            'sklearn_based': [],
            'optimization_focused': [],
            'ensemble_methods': [],
            'advanced_ai': [],
            'validation_systems': [],
            'analysis_tools': []
        }
        
        evolution_phases = {
            'phase_1_initial': [],      # Systèmes initiaux TensorFlow
            'phase_2_improvement': [],   # Améliorations et optimisations
            'phase_3_advanced': [],      # Techniques avancées
            'phase_4_validation': [],    # Validation et tests
            'phase_5_final': []          # Systèmes finaux
        }
        
        for system in systems_data:
            if not system:
                continue
                
            # Catégorisation par technologie
            technologies = system.get('technologies', [])
            
            if 'TensorFlow' in technologies:
                categories['tensorflow_based'].append(system)
            if any(tech in technologies for tech in ['Scikit-Learn', 'Random Forest', 'Gradient Boosting']):
                categories['sklearn_based'].append(system)
            if 'Optuna' in technologies:
                categories['optimization_focused'].append(system)
            if 'Ensemble' in technologies:
                categories['ensemble_methods'].append(system)
            if any(tech in technologies for tech in ['Quantum', 'Genetic/Evolution']):
                categories['advanced_ai'].append(system)
            if system['system_type'] == 'Validator':
                categories['validation_systems'].append(system)
            if system['system_type'] == 'Analyzer':
                categories['analysis_tools'].append(system)
            
            # Catégorisation par phase d'évolution (basée sur le nom et la date)
            file_name = system['file_name'].lower()
            creation_time = system['creation_time']
            
            if any(keyword in file_name for keyword in ['euromillions_model', 'create_', 'basic']):
                evolution_phases['phase_1_initial'].append(system)
            elif any(keyword in file_name for keyword in ['optimized', 'improved', 'enhanced']):
                evolution_phases['phase_2_improvement'].append(system)
            elif any(keyword in file_name for keyword in ['advanced', 'ultra', 'revolutionary', 'quantum']):
                evolution_phases['phase_3_advanced'].append(system)
            elif any(keyword in file_name for keyword in ['validator', 'validation', 'test']):
                evolution_phases['phase_4_validation'].append(system)
            elif any(keyword in file_name for keyword in ['final', 'ultimate', 'targeted']):
                evolution_phases['phase_5_final'].append(system)
        
        return categories, evolution_phases
        
    def extract_performance_data(self, systems_data):
        """Extrait les données de performance de tous les systèmes."""
        print("📈 Extraction des données de performance...")
        
        performance_summary = {}
        
        # Recherche dans les fichiers de résultats
        result_files = glob.glob('/home/ubuntu/results/**/*.json', recursive=True)
        result_files.extend(glob.glob('/home/ubuntu/results/**/*.txt', recursive=True))
        
        for file_path in result_files:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extraction des métriques de performance
                    if isinstance(data, dict):
                        system_name = os.path.basename(file_path).replace('.json', '')
                        
                        metrics = {}
                        
                        # Recherche de métriques communes
                        if 'confidence_score' in data:
                            metrics['confidence'] = data['confidence_score']
                        if 'validation' in data and isinstance(data['validation'], dict):
                            validation = data['validation']
                            if 'total_matches' in validation:
                                metrics['total_matches'] = validation['total_matches']
                            if 'accuracy_percentage' in validation:
                                metrics['accuracy'] = validation['accuracy_percentage']
                        if 'performance' in data:
                            metrics.update(data['performance'])
                        
                        if metrics:
                            performance_summary[system_name] = metrics
                            
                elif file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extraction de métriques depuis le texte
                    system_name = os.path.basename(file_path).replace('.txt', '')
                    
                    # Patterns de recherche
                    patterns = {
                        'correspondances': r'correspondances?[:\s]*([0-9]+)/([0-9]+)',
                        'confiance': r'confiance[:\s]*([0-9.]+)',
                        'score': r'score[:\s]*([0-9.]+)',
                        'accuracy': r'accuracy[:\s]*([0-9.]+)%?',
                        'precision': r'précision[:\s]*([0-9.]+)%?'
                    }
                    
                    metrics = {}
                    for metric_name, pattern in patterns.items():
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            if metric_name == 'correspondances':
                                # Format spécial pour correspondances
                                for match in matches:
                                    if len(match) == 2:
                                        metrics['matches'] = f"{match[0]}/{match[1]}"
                                        metrics['accuracy'] = (int(match[0]) / int(match[1])) * 100
                            else:
                                metrics[metric_name] = float(matches[0])
                    
                    if metrics:
                        performance_summary[system_name] = metrics
                        
            except Exception as e:
                continue
        
        return performance_summary
        
    def create_evolution_timeline(self, systems_data):
        """Crée une timeline de l'évolution des systèmes."""
        print("📅 Création de la timeline d'évolution...")
        
        # Tri par date de création
        sorted_systems = sorted(systems_data, key=lambda x: x['creation_time'] if x else 0)
        
        timeline = []
        
        for i, system in enumerate(sorted_systems):
            if not system:
                continue
                
            timeline_entry = {
                'order': i + 1,
                'file_name': system['file_name'],
                'creation_date': datetime.fromtimestamp(system['creation_time']).strftime('%Y-%m-%d %H:%M'),
                'system_type': system['system_type'],
                'technologies': system['technologies'],
                'description_preview': system['description'][:100] + "..." if len(system['description']) > 100 else system['description'],
                'file_size_kb': round(system['file_size'] / 1024, 1)
            }
            
            timeline.append(timeline_entry)
        
        return timeline
        
    def analyze_technology_evolution(self, categories):
        """Analyse l'évolution des technologies utilisées."""
        print("🔬 Analyse de l'évolution technologique...")
        
        tech_evolution = {
            'tensorflow_usage': len(categories['tensorflow_based']),
            'sklearn_usage': len(categories['sklearn_based']),
            'optimization_adoption': len(categories['optimization_focused']),
            'ensemble_methods': len(categories['ensemble_methods']),
            'advanced_ai_exploration': len(categories['advanced_ai']),
            'validation_emphasis': len(categories['validation_systems'])
        }
        
        # Analyse des tendances
        trends = {}
        
        # Tendance vers l'optimisation
        if tech_evolution['optimization_adoption'] > 0:
            trends['optimization_trend'] = "Forte adoption des techniques d'optimisation (Optuna)"
        
        # Tendance vers la validation
        if tech_evolution['validation_emphasis'] > 2:
            trends['validation_trend'] = "Emphasis croissante sur la validation scientifique"
        
        # Diversification technologique
        tech_count = sum(1 for count in tech_evolution.values() if count > 0)
        if tech_count >= 4:
            trends['diversification'] = "Forte diversification technologique"
        
        return tech_evolution, trends
        
    def generate_systems_inventory(self, systems_data, categories, evolution_phases, performance_data):
        """Génère un inventaire complet des systèmes."""
        print("📋 Génération de l'inventaire complet...")
        
        inventory = {
            'analysis_date': datetime.now().isoformat(),
            'total_systems': len([s for s in systems_data if s]),
            'systems_by_category': {cat: len(systems) for cat, systems in categories.items()},
            'systems_by_phase': {phase: len(systems) for phase, systems in evolution_phases.items()},
            'performance_summary': performance_data,
            'detailed_systems': []
        }
        
        # Ajout des détails de chaque système
        for system in systems_data:
            if not system:
                continue
                
            system_detail = {
                'name': system['file_name'],
                'type': system['system_type'],
                'technologies': system['technologies'],
                'description': system['description'],
                'file_size_kb': round(system['file_size'] / 1024, 1),
                'creation_date': datetime.fromtimestamp(system['creation_time']).strftime('%Y-%m-%d %H:%M'),
                'performance_indicators': system.get('performance_indicators', {}),
                'predictions_found': system.get('predictions_found', {})
            }
            
            inventory['detailed_systems'].append(system_detail)
        
        return inventory
        
    def save_analysis_results(self, inventory, timeline, tech_evolution, trends):
        """Sauvegarde tous les résultats d'analyse."""
        print("💾 Sauvegarde des résultats d'analyse...")
        
        # Inventaire complet
        with open(f'{self.analysis_dir}/systems_inventory/complete_inventory.json', 'w') as f:
            json.dump(inventory, f, indent=2, default=str)
        
        # Timeline d'évolution
        with open(f'{self.analysis_dir}/evolution_analysis/evolution_timeline.json', 'w') as f:
            json.dump(timeline, f, indent=2, default=str)
        
        # Évolution technologique
        tech_analysis = {
            'technology_usage': tech_evolution,
            'trends_identified': trends,
            'analysis_date': datetime.now().isoformat()
        }
        
        with open(f'{self.analysis_dir}/evolution_analysis/technology_evolution.json', 'w') as f:
            json.dump(tech_analysis, f, indent=2, default=str)
        
        # Rapport de synthèse
        synthesis_report = f"""
# ANALYSE RÉTROSPECTIVE COMPLÈTE - TOUS LES SYSTÈMES DÉVELOPPÉS

## RÉSUMÉ EXÉCUTIF
Date d'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Nombre total de systèmes: {inventory['total_systems']}

## RÉPARTITION PAR CATÉGORIE
{chr(10).join([f"- {cat.replace('_', ' ').title()}: {count} systèmes" for cat, count in inventory['systems_by_category'].items()])}

## ÉVOLUTION TECHNOLOGIQUE
{chr(10).join([f"- {tech.replace('_', ' ').title()}: {count} systèmes" for tech, count in tech_evolution.items()])}

## TENDANCES IDENTIFIÉES
{chr(10).join([f"- {trend}: {description}" for trend, description in trends.items()])}

## TIMELINE D'ÉVOLUTION
{chr(10).join([f"{entry['order']}. {entry['file_name']} ({entry['creation_date']}) - {entry['system_type']}" for entry in timeline[:10]])}

## SYSTÈMES LES PLUS RÉCENTS
{chr(10).join([f"- {entry['file_name']} ({entry['creation_date']})" for entry in timeline[-5:]])}

## PERFORMANCE GLOBALE
Systèmes avec métriques de performance: {len(inventory['performance_summary'])}
{chr(10).join([f"- {name}: {metrics}" for name, metrics in list(inventory['performance_summary'].items())[:5]])}

---
Analyse générée automatiquement par l'Analyseur Rétrospectif Complet
"""
        
        with open(f'{self.analysis_dir}/synthesis_report.txt', 'w') as f:
            f.write(synthesis_report)
        
        print("✅ Tous les résultats sauvegardés!")
        
    def run_comprehensive_analysis(self):
        """Exécute l'analyse complète de tous les systèmes."""
        print("🚀 LANCEMENT DE L'ANALYSE RÉTROSPECTIVE COMPLÈTE 🚀")
        print("=" * 70)
        
        # 1. Scan de tous les systèmes
        print("🔍 Phase 1: Scan de tous les systèmes...")
        system_files = self.scan_all_systems()
        
        # 2. Analyse détaillée de chaque système
        print("📊 Phase 2: Analyse détaillée...")
        systems_data = []
        for file_path in system_files:
            system_info = self.analyze_system_file(file_path)
            systems_data.append(system_info)
        
        # 3. Catégorisation
        print("📋 Phase 3: Catégorisation...")
        categories, evolution_phases = self.categorize_systems(systems_data)
        
        # 4. Extraction des performances
        print("📈 Phase 4: Extraction des performances...")
        performance_data = self.extract_performance_data(systems_data)
        
        # 5. Timeline d'évolution
        print("📅 Phase 5: Timeline d'évolution...")
        timeline = self.create_evolution_timeline(systems_data)
        
        # 6. Analyse technologique
        print("🔬 Phase 6: Analyse technologique...")
        tech_evolution, trends = self.analyze_technology_evolution(categories)
        
        # 7. Génération de l'inventaire
        print("📋 Phase 7: Génération de l'inventaire...")
        inventory = self.generate_systems_inventory(systems_data, categories, evolution_phases, performance_data)
        
        # 8. Sauvegarde
        print("💾 Phase 8: Sauvegarde...")
        self.save_analysis_results(inventory, timeline, tech_evolution, trends)
        
        print("✅ ANALYSE RÉTROSPECTIVE COMPLÈTE TERMINÉE!")
        
        return {
            'inventory': inventory,
            'timeline': timeline,
            'tech_evolution': tech_evolution,
            'trends': trends,
            'categories': categories,
            'evolution_phases': evolution_phases
        }

if __name__ == "__main__":
    # Lancement de l'analyse rétrospective complète
    analyzer = ComprehensiveSystemAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\n🔍 RÉSULTATS DE L'ANALYSE RÉTROSPECTIVE:")
    print(f"Systèmes analysés: {results['inventory']['total_systems']}")
    print(f"Catégories identifiées: {len(results['categories'])}")
    print(f"Phases d'évolution: {len(results['evolution_phases'])}")
    print(f"Tendances technologiques: {len(results['trends'])}")
    
    print("\n🎉 ANALYSE RÉTROSPECTIVE TERMINÉE! 🎉")

