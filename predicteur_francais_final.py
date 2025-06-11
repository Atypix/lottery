#!/usr/bin/env python3
"""
Script de prédiction final avec données françaises actualisées
Utilisation simple pour l'utilisateur
"""

import json
import os
from datetime import datetime

def load_french_prediction():
    """Charge la prédiction française finale"""
    result_path = '/home/ubuntu/results/french_aggregation/final_french_prediction.json'
    
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            return json.load(f)
    else:
        return None

def display_prediction():
    """Affiche la prédiction de manière claire"""
    print("🇫🇷 PRÉDICTEUR EUROMILLIONS - DONNÉES FRANÇAISES ACTUALISÉES")
    print("=" * 65)
    
    prediction_data = load_french_prediction()
    
    if prediction_data is None:
        print("❌ Erreur : Données de prédiction non trouvées")
        return
    
    prediction = prediction_data['prediction']
    validation = prediction_data['validation']
    
    print(f"📅 Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
    print(f"🇫🇷 Basé sur : {prediction_data['total_draws_analyzed']} tirages français récents")
    print()
    
    print("🎯 PRÉDICTION FINALE :")
    print("=" * 25)
    numbers_str = " - ".join(map(str, prediction['numbers']))
    stars_str = " - ".join(map(str, prediction['stars']))
    
    print(f"🔢 NUMÉROS : {numbers_str}")
    print(f"⭐ ÉTOILES : {stars_str}")
    print(f"📊 CONFIANCE : {prediction['confidence']:.1%}")
    print()
    
    print("✅ VALIDATION :")
    print("=" * 15)
    print(f"🎯 Tirage de référence : {', '.join(map(str, validation['reference_draw']))}")
    print(f"🔮 Prédiction générée : {', '.join(map(str, validation['predicted_draw']))}")
    print(f"✅ Correspondances : {validation['total_matches']}/7")
    print(f"📈 Précision : {validation['accuracy']:.1f}%")
    print()
    
    print("🔬 MÉTHODOLOGIE :")
    print("=" * 17)
    print("• Fréquences globales (20%)")
    print("• Tendances récentes (30%)")
    print("• Équilibrage statistique (25%)")
    print("• Consensus pondéré (25%)")
    print()
    
    print("🎫 TICKET DE JEU :")
    print("=" * 17)
    print(f"Numéros : {numbers_str}")
    print(f"Étoiles : {stars_str}")
    print()
    print("🎲 Bonne chance !")
    print()
    
    # Sauvegarde du ticket simple
    ticket_simple = f"""
🎫 TICKET EUROMILLIONS FRANÇAIS
==============================

NUMÉROS : {numbers_str}
ÉTOILES : {stars_str}

Confiance : {prediction['confidence']:.1%}
Précision validée : {validation['accuracy']:.1f}%

Généré le {datetime.now().strftime('%d/%m/%Y')}
Basé sur {prediction_data['total_draws_analyzed']} tirages français récents
"""
    
    with open('/home/ubuntu/ticket_final_francais.txt', 'w') as f:
        f.write(ticket_simple)
    
    print("💾 Ticket sauvegardé : ticket_final_francais.txt")

if __name__ == "__main__":
    display_prediction()

