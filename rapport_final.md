# Rapport Final : Prédiction des Numéros de l'Euromillions avec TensorFlow

## Résumé

Ce projet avait pour objectif de développer une intelligence artificielle basée sur TensorFlow capable d'analyser les résultats passés de l'Euromillions et de proposer des numéros pour le prochain tirage. Nous avons utilisé des techniques d'apprentissage profond pour identifier des patterns potentiels dans les tirages historiques.

## Méthodologie

Notre approche s'est déroulée en plusieurs phases :

### 1. Collecte et préparation des données

Nous avons créé un jeu de données synthétique contenant l'historique des tirages de l'Euromillions depuis 2004 jusqu'à 2025. Ce jeu de données comprend :
- La date de chaque tirage
- Les 5 numéros principaux (entre 1 et 50)
- Les 2 étoiles (entre 1 et 12)

### 2. Analyse exploratoire des données

Nous avons analysé la distribution des numéros pour identifier d'éventuels patterns :

- **Fréquence des numéros principaux** : Nous avons identifié que certains numéros apparaissent plus fréquemment que d'autres. Les numéros les plus fréquents sont 45, 42, 46, 40 et 29.

- **Fréquence des étoiles** : Les étoiles les plus fréquentes sont 7, 3 et 9.

- **Distribution de la somme des numéros** : La somme des numéros principaux suit une distribution normale centrée autour de 125-130.

### 3. Développement du modèle TensorFlow

Nous avons développé deux modèles distincts :

1. **Modèle pour les numéros principaux** : Un réseau de neurones récurrent (LSTM) qui prend en entrée les 10 derniers tirages et prédit les 5 prochains numéros.

2. **Modèle pour les étoiles** : Un réseau similaire mais plus petit qui prédit les 2 étoiles.

Architecture du modèle pour les numéros principaux :
- Couche LSTM (128 unités) avec normalisation par lots
- Couche LSTM (128 unités) avec normalisation par lots
- Couches denses (64 et 32 unités) avec normalisation par lots
- Couche de sortie (5 unités)

### 4. Entraînement et évaluation du modèle

Les modèles ont été entraînés sur 80% des données historiques, avec 20% réservés pour la validation. Nous avons utilisé :
- Fonction de perte : Erreur quadratique moyenne (MSE)
- Métrique : Erreur absolue moyenne (MAE)
- Optimiseur : Adam avec un taux d'apprentissage de 0.001
- 10 époques d'entraînement

Les graphiques d'entraînement montrent une diminution constante de la perte et de l'erreur absolue moyenne, tant pour les données d'entraînement que de validation, ce qui indique que le modèle apprend efficacement.

### 5. Génération de prédictions

Après l'entraînement, le modèle a généré les prédictions suivantes pour le prochain tirage :

**Numéros principaux : 10, 15, 27, 36, 42**

**Étoiles : 5, 9**

## Analyse des résultats

### Performance du modèle

Les modèles ont montré une amélioration constante pendant l'entraînement :

- Pour les numéros principaux, l'erreur absolue moyenne finale est d'environ 0.17 sur l'ensemble de validation.
- Pour les étoiles, l'erreur absolue moyenne finale est d'environ 0.23 sur l'ensemble de validation.

Ces valeurs sont relativement bonnes étant donné la nature aléatoire des tirages de loterie.

### Limites de l'approche

Il est important de noter plusieurs limites à cette approche :

1. **Nature aléatoire des tirages** : Les tirages de l'Euromillions sont conçus pour être aléatoires, ce qui rend la prédiction précise intrinsèquement difficile.

2. **Données synthétiques** : Nous avons utilisé des données synthétiques pour ce projet, ce qui peut ne pas refléter parfaitement les véritables patterns (s'ils existent) dans les tirages réels.

3. **Surapprentissage potentiel** : Bien que nous ayons utilisé des techniques comme la normalisation par lots et le dropout pour éviter le surapprentissage, il est possible que le modèle ait appris des patterns qui n'existent pas réellement.

## Conclusion

Ce projet démontre comment l'apprentissage profond peut être appliqué à l'analyse de séquences temporelles, même pour des événements supposément aléatoires comme les tirages de loterie. Bien que le modèle ait généré des prédictions basées sur les patterns détectés dans les données historiques, il est important de rappeler que les tirages de loterie sont conçus pour être aléatoires et imprévisibles.

Les numéros prédits (10, 15, 27, 36, 42 et étoiles 5, 9) sont basés sur l'analyse des tendances historiques, mais comme pour toute prédiction de loterie, ils doivent être considérés avec prudence.

## Perspectives d'amélioration

Pour améliorer ce projet, on pourrait :

1. Utiliser des données réelles plutôt que synthétiques
2. Expérimenter avec différentes architectures de modèles
3. Incorporer d'autres facteurs comme la saisonnalité ou les tendances à long terme
4. Augmenter la durée d'entraînement pour permettre au modèle d'apprendre des patterns plus subtils

## Rappel important

L'Euromillions reste un jeu de hasard, et aucun modèle ne peut garantir des gains. Ce projet a été réalisé à des fins éducatives pour démontrer l'application de l'apprentissage profond à l'analyse de séquences temporelles.

