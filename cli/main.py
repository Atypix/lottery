import argparse
import sys
import os
from collections import Counter # Added
import random # Added
from common.date_utils import get_next_euromillions_draw_date # Added

# Add the parent directory to sys.path to find fetch_real_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_real_data import update_euromillions_data
from predicteur_final_valide import FinalValidatedPredictor
from revolutionary_predictor import RevolutionaryPredictor # Renamed import
from aggregated_final_predictor import AggregatedFinalPredictor
from euromillions_model import predict_with_tensorflow_model, train_all_models_and_predict # Updated import

import subprocess # Added
import json # Added
# sys, os, Counter, random, get_next_euromillions_draw_date are already imported

PREDICTOR_CONFIGS = [
    # Scientifique
    {'name': 'advanced_ml_predictor', 'path': 'advanced_ml_predictor.py', 'category': 'Scientifique'},
    {'name': 'optimized_scientific_predictor', 'path': 'optimized_scientific_predictor.py', 'category': 'Scientifique'},
    {'name': 'predicteur_final_valide', 'path': 'predicteur_final_valide.py', 'category': 'Scientifique'},
    {'name': 'euromillions_predictor', 'path': 'euromillions_predictor.py', 'category': 'Scientifique'},
    {'name': 'predict_euromillions', 'path': 'predict_euromillions.py', 'category': 'Scientifique'},
    {'name': 'quick_optimized_prediction', 'path': 'quick_optimized_prediction.py', 'category': 'Scientifique'},
    # Révolutionnaire
    {'name': 'adaptive_singularity', 'path': 'adaptive_singularity.py', 'category': 'Revolutionnaire'},
    {'name': 'chaos_fractal_predictor', 'path': 'chaos_fractal_predictor.py', 'category': 'Revolutionnaire'},
    {'name': 'conscious_ai_predictor', 'path': 'conscious_ai_predictor.py', 'category': 'Revolutionnaire'},
    {'name': 'futuristic_phase1_optimized', 'path': 'futuristic_phase1_optimized.py', 'category': 'Revolutionnaire'},
    {'name': 'multiverse_predictor', 'path': 'multiverse_predictor.py', 'category': 'Revolutionnaire'},
    # Méta-Prédicteurs
    {'name': 'aggregated_final_predictor', 'path': 'aggregated_final_predictor.py', 'category': 'Meta-Predicteurs'},
    {'name': 'singularity_predictor', 'path': 'singularity_predictor.py', 'category': 'Meta-Predicteurs'},
    {'name': 'meta_revolutionary_predictor', 'path': 'meta_revolutionary_predictor.py', 'category': 'Meta-Predicteurs'},
]

# --- Model Invocation Functions ---
def run_final_valide():
    predictor = FinalValidatedPredictor()
    # Original dict from predictor.run_final_prediction() has 'confidence_score'
    # and 'model_name': 'predicteur_final_valide'
    original_pred = predictor.run_final_prediction()
    return {
        'numbers': original_pred.get('numbers'),
        'stars': original_pred.get('stars'),
        'confidence': original_pred.get('confidence_score'), # Map to 'confidence'
        'model_name': original_pred.get('model_name'),       # Pass original model_name
        'status': 'success' # Assuming success if predictor ran
    }

def run_revolutionnaire():
    predictor = RevolutionaryPredictor()
    return predictor.generate_revolutionary_prediction()

def run_agrege():
    print("Note: The 'agrege' model requires prior setup of synthesis and test result files.")
    print("Attempting to run. Ensure dummy files or actual results exist in 'results/' subdirectories.")
    try:
        predictor = AggregatedFinalPredictor()
        results = predictor.run_final_aggregation()
        # Extract standardized parts from the comprehensive results
        return {
            'numbers': results.get('prediction', {}).get('numbers', []),
            'stars': results.get('prediction', {}).get('stars', []),
            'confidence': results.get('metrics', {}).get('confidence_percentage', 0.0),
            'model_name': 'agrege', # Align with CLI key
            'status': 'success',
            'message': 'Aggregated prediction generated.',
            # aggregated_final_predictor.py now adds target_draw_date to its prediction dict
            'target_draw_date': results.get('prediction', {}).get('target_draw_date')
        }
    except FileNotFoundError as e:
        return {
            'numbers': [], 'stars': [], 'confidence': None,
            'model_name': 'aggregated_final_predictor', 'status': 'failure',
            'message': f"Failed to run aggregated model due to missing file: {e}. Ensure dummy files or actual results exist."
        }
    except Exception as e:
        return {
            'numbers': [], 'stars': [], 'confidence': None,
            'model_name': 'aggregated_final_predictor', 'status': 'failure',
            'message': f"An error occurred with aggregated model: {e}"
        }

import subprocess # Added
import json # Added

# --- Model Invocation Functions --- (Keep existing ones for 'predict' command)
# ... run_final_valide, run_revolutionnaire, run_agrege, run_tf_lstm_std, run_tf_lstm_enhanced ...
# These are kept for the 'predict' command functionality.

PREDICTOR_CONFIGS = [
    # Scientifique
    {'name': 'advanced_ml_predictor', 'path': 'advanced_ml_predictor.py', 'category': 'Scientifique'},
    {'name': 'optimized_scientific_predictor', 'path': 'optimized_scientific_predictor.py', 'category': 'Scientifique'},
    {'name': 'predicteur_final_valide', 'path': 'predicteur_final_valide.py', 'category': 'Scientifique'},
    {'name': 'euromillions_predictor', 'path': 'euromillions_predictor.py', 'category': 'Scientifique'},
    {'name': 'predict_euromillions', 'path': 'predict_euromillions.py', 'category': 'Scientifique'},
    {'name': 'quick_optimized_prediction', 'path': 'quick_optimized_prediction.py', 'category': 'Scientifique'},
    # Révolutionnaire
    {'name': 'adaptive_singularity', 'path': 'adaptive_singularity.py', 'category': 'Revolutionnaire'},
    {'name': 'chaos_fractal_predictor', 'path': 'chaos_fractal_predictor.py', 'category': 'Revolutionnaire'},
    {'name': 'conscious_ai_predictor', 'path': 'conscious_ai_predictor.py', 'category': 'Revolutionnaire'},
    {'name': 'futuristic_phase1_optimized', 'path': 'futuristic_phase1_optimized.py', 'category': 'Revolutionnaire'},
    {'name': 'multiverse_predictor', 'path': 'multiverse_predictor.py', 'category': 'Revolutionnaire'},
    # Méta-Prédicteurs
    {'name': 'aggregated_final_predictor', 'path': 'aggregated_final_predictor.py', 'category': 'Meta-Predicteurs'},
    {'name': 'singularity_predictor', 'path': 'singularity_predictor.py', 'category': 'Meta-Predicteurs'},
    {'name': 'meta_revolutionary_predictor', 'path': 'meta_revolutionary_predictor.py', 'category': 'Meta-Predicteurs'},
]

def _validate_prediction_data(data, script_name):
    """Helper to validate the structure and values of prediction data from scripts."""
    required_keys = {"nom_predicteur": str, "numeros": list, "etoiles": list}
    for key, expected_type in required_keys.items():
        if key not in data:
            print(f"Erreur Validation: Clé manquante '{key}' dans la sortie de {script_name}.", file=sys.stderr)
            return False
        if not isinstance(data[key], expected_type):
            print(f"Erreur Validation: Clé '{key}' devrait être de type {expected_type} mais est {type(data[key])} dans {script_name}.", file=sys.stderr)
            return False

    # Validate 'numeros'
    numeros = data['numeros']
    if len(numeros) != 5:
        print(f"Erreur Validation: 'numeros' doit contenir 5 éléments, {len(numeros)} trouvés dans {script_name}.", file=sys.stderr)
        return False
    if len(set(numeros)) != 5:
        print(f"Erreur Validation: 'numeros' doit contenir des valeurs uniques, {numeros} trouvés dans {script_name}.", file=sys.stderr)
        return False
    for num in numeros:
        if not (isinstance(num, int) and 1 <= num <= 50):
            print(f"Erreur Validation: Chaque numéro dans 'numeros' doit être un int entre 1-50, '{num}' trouvé dans {script_name}.", file=sys.stderr)
            return False

    # Validate 'etoiles'
    etoiles = data['etoiles']
    if len(etoiles) != 2:
        print(f"Erreur Validation: 'etoiles' doit contenir 2 éléments, {len(etoiles)} trouvés dans {script_name}.", file=sys.stderr)
        return False
    if len(set(etoiles)) != 2:
        print(f"Erreur Validation: 'etoiles' doit contenir des valeurs uniques, {etoiles} trouvés dans {script_name}.", file=sys.stderr)
        return False
    for star in etoiles:
        if not (isinstance(star, int) and 1 <= star <= 12):
            print(f"Erreur Validation: Chaque étoile dans 'etoiles' doit être un int entre 1-12, '{star}' trouvé dans {script_name}.", file=sys.stderr)
            return False

    # Optional keys validation (type check if present)
    if 'date_tirage_cible' in data and not isinstance(data['date_tirage_cible'], str):
        print(f"Erreur Validation: 'date_tirage_cible' devrait être str, {type(data['date_tirage_cible'])} trouvé dans {script_name}.", file=sys.stderr)
        return False
    if 'confidence' in data and not (data['confidence'] is None or isinstance(data['confidence'], float) or isinstance(data['confidence'], int)): # Allow None
        print(f"Erreur Validation: 'confidence' devrait être float/int, {type(data['confidence'])} trouvé dans {script_name}.", file=sys.stderr)
        return False
    if 'categorie' in data and not isinstance(data['categorie'], str):
        print(f"Erreur Validation: 'categorie' devrait être str, {type(data['categorie'])} trouvé dans {script_name}.", file=sys.stderr)
        return False

    return True

def run_consensus_by_frequency_prediction(selected_model_names: list = None): # Signature kept for now, but selected_model_names might be re-purposed or ignored
    print("🤝 Lancement du mode consensus des prédicteurs externes...")

    target_date_obj = get_next_euromillions_draw_date("data/euromillions_enhanced_dataset.csv") # Prefer data/
    if not target_date_obj: # Fallback if data/ not found
         target_date_obj = get_next_euromillions_draw_date("euromillions_enhanced_dataset.csv")

    target_date_str = None
    if target_date_obj:
        target_date_str = target_date_obj.strftime('%Y-%m-%d')
        print(f"🗓️  Date de tirage cible pour les prédicteurs: {target_date_str}")
    else:
        print("⚠️ Impossible de déterminer la prochaine date de tirage. Les prédicteurs utiliseront leur propre logique de date.", file=sys.stderr)

    successful_predictions = []
    failed_predictors = []

    # Filter PREDICTOR_CONFIGS if selected_model_names are provided
    predictors_to_run = PREDICTOR_CONFIGS
    if selected_model_names and len(selected_model_names) > 0:
        print(f"Consensus pour les modèles sélectionnés: {', '.join(selected_model_names)}")
        predictors_to_run = [p for p in PREDICTOR_CONFIGS if p['name'] in selected_model_names]
        if not predictors_to_run:
            print("Aucun des modèles sélectionnés n'est valide ou configuré.", file=sys.stderr)
            # Return an empty structure or specific error message
            return {
                'numbers': [], 'stars': [], 'confidence': None,
                'model_name': 'consensus_external', 'status': 'failure',
                'message': 'No valid models selected for consensus.',
                'target_draw_date': target_date_str
            }
    else:
        print(f"Consensus pour tous les {len(PREDICTOR_CONFIGS)} modèles configurés.")


    for script_config in predictors_to_run:
        script_name = script_config['name']
        script_path = script_config['path']
        print(f"\n▶️  Exécution de {script_name} ({script_config['category']})...")

        command = ['python3', script_path]
        if target_date_str:
            command.extend(['--date', target_date_str])

        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)

            if process.returncode != 0:
                error_message = f"Le script {script_name} a terminé avec le code {process.returncode}."
                stderr_output = process.stderr.strip()
                if stderr_output:
                    error_message += f"\nSortie d'erreur:\n{stderr_output}"
                print(error_message, file=sys.stderr)
                failed_predictors.append({'name': script_name, 'reason': f"Code de sortie {process.returncode}", 'details': stderr_output})
                continue

            try:
                prediction_data = json.loads(process.stdout.strip())
                if _validate_prediction_data(prediction_data, script_name):
                    # Augment with category from config to ensure it's there
                    prediction_data['category_from_config'] = script_config['category']
                    successful_predictions.append(prediction_data)
                    print(f"  ✅ {script_name}: Prédiction valide collectée.")
                else:
                    failed_predictors.append({'name': script_name, 'reason': 'Format de données de prédiction invalide.'})
            except json.JSONDecodeError as e:
                print(f"Erreur: Impossible de décoder la sortie JSON de {script_name}: {e}", file=sys.stderr)
                print(f"Sortie brute de {script_name}:\n{process.stdout.strip()}", file=sys.stderr)
                failed_predictors.append({'name': script_name, 'reason': 'Sortie JSON invalide.'})

        except subprocess.TimeoutExpired:
            print(f"Erreur: Le script {script_name} a dépassé le délai de 300 secondes.", file=sys.stderr)
            failed_predictors.append({'name': script_name, 'reason': 'Timeout'})
        except FileNotFoundError:
            print(f"Erreur: Script {script_path} non trouvé.", file=sys.stderr)
            failed_predictors.append({'name': script_name, 'reason': 'Script non trouvé.'})
        except Exception as e:
            print(f"Erreur inattendue lors de l'exécution de {script_name}: {e}", file=sys.stderr)
            failed_predictors.append({'name': script_name, 'reason': f"Erreur inattendue: {str(e)}"})

    print("\n--- Récapitulatif des Prédictions du Consensus ---")
    if not successful_predictions:
        print("Aucune prédiction n'a pu être collectée avec succès.")
    else:
        # Group by category
        grouped_predictions = {}
        for pred in successful_predictions:
            category = pred.get('category_from_config', pred.get('categorie', 'Inconnue')) # Prioritize config category
            if category not in grouped_predictions:
                grouped_predictions[category] = []
            grouped_predictions[category].append(pred)

        for category, preds_in_category in grouped_predictions.items():
            print(f"\n--- Catégorie: {category} ({len(preds_in_category)} prédictions) ---")
            for pred in preds_in_category:
                nom = pred.get('nom_predicteur', 'N/A')
                nums = pred.get('numeros', [])
                etoiles = pred.get('etoiles', [])
                conf = pred.get('confidence')
                conf_str = f"{conf:.2f}/10" if isinstance(conf, (float, int)) else "N/A"
                print(f"  {nom}: Numéros={nums}, Étoiles={etoiles}, Confiance={conf_str}")

    if failed_predictors:
        print("\n--- Prédicteurs Échoués ---")
        for failed in failed_predictors:
            print(f"  - {failed['name']}: {failed['reason']}")
            if 'details' in failed and failed['details']:
                 print(f"    Détails: {failed['details'][:200]}{'...' if len(failed['details']) > 200 else ''}")


    # The original function returned a single combined prediction.
    # This new version prints a report. We need to decide what it should return for the CLI.
    # For now, let's make it return a summary or a status.
    # The `display_prediction` function will then need to be adapted or this function will print directly.
    # The prompt asks for this function to *display* results, so direct printing is fine.
    # The `display_prediction` function might become obsolete for this command.

    # This function is called by `main` and its result is passed to `display_prediction`.
    # To avoid breaking that, we'll return a placeholder dict, as the primary output is now print-based.
    return {
        'numbers': [], # No single consensus numbers/stars from this process
        'stars': [],
        'confidence': None,
        'model_name': 'consensus_report', # Indicates this is a report
        'status': 'success' if successful_predictions else 'failure',
        'message': f'{len(successful_predictions)} prédictions collectées, {len(failed_predictors)} échecs.',
        'target_draw_date': target_date_str
    }


AVAILABLE_MODELS = {
    'final_valide': run_final_valide,
    'revolutionnaire': run_revolutionnaire,
    'agrege': run_agrege
    # 'tf_lstm_std': run_tf_lstm_std, # Removed
    # 'tf_lstm_enhanced': run_tf_lstm_enhanced # Removed
    # Note: AVAILABLE_MODELS is now only used by the 'predict' command, not 'predict-consensus'.
    # The 'predict-consensus' --models arg now filters PREDICTOR_CONFIGS.
}

def display_prediction(result):
    model_name_display = result.get('model_name', 'Unknown Model')
    target_date_display = result.get('target_draw_date') # Get target_draw_date

    print(f"--- Prediction Results: {model_name_display} ---")
    if target_date_display: # Display if available
        print(f"For Draw Date: {target_date_display}")

    if result.get('status') == 'failure':
        print(f"Error: {result.get('message', 'Prediction failed.')}")
        return

    print(f"Numbers: {result.get('numbers', 'N/A')}")
    print(f"Stars: {result.get('stars', 'N/A')}")

    confidence = result.get('confidence')
    # model_name is already extracted as model_name_display, use it for logic
    # This ensures we use the name that was part of the result dict.

    if confidence is not None:
        if isinstance(confidence, float):
            if model_name_display == 'revolutionnaire':
                 print(f"Confidence: {confidence:.1%}")
            elif model_name_display == 'agrege':
                 print(f"Confidence: {confidence:.1f}%")
            elif model_name_display == 'final_valide':
                 print(f"Confidence: {confidence}/10")
            # tf_lstm has confidence: None, so it's handled by the else below
            else: # Other float confidences (if any new model returns float differently)
                 print(f"Confidence: {confidence}")
        else: # Non-float confidences (currently none of the models return non-float)
             print(f"Confidence: {confidence}")
    else:
        print("Confidence: Not Available") # Handles tf_lstm and any other case

def main():
    parser = argparse.ArgumentParser(description="Euromillions AI CLI", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # update-data command
    parser_update = subparsers.add_parser('update-data', help='Update lottery data from the source')

    # list-models command
    parser_list = subparsers.add_parser('list-models', help='List available prediction models')

    # predict command
    predict_help = "Predict lottery numbers.\nAvailable models:\n" + "\n".join([f"  - {name}" for name in AVAILABLE_MODELS.keys()])
    parser_predict = subparsers.add_parser('predict', help=predict_help, formatter_class=argparse.RawTextHelpFormatter)
    parser_predict.add_argument('model_name', choices=list(AVAILABLE_MODELS.keys()), metavar='model_name', help='Name of the prediction model to use')

    # predict-consensus command
    parser_consensus = subparsers.add_parser(
        'predict-consensus',
        help='Generate a prediction by aggregating results from specified models (or all if none specified) based on frequency.'
    )
    parser_consensus.add_argument(
        '--models',
        nargs='*', # Allows zero or more arguments
        choices=[p['name'] for p in PREDICTOR_CONFIGS], # Use names from PREDICTOR_CONFIGS
        metavar='MODEL_NAME',
        help=f'Optional: List of model names to include in the consensus. Available: {", ".join([p["name"] for p in PREDICTOR_CONFIGS])}. If not provided, all are used.'
    )

    # New train-tf-model command
    parser_train_tf = subparsers.add_parser(
        'train-tf-model',
        help='Train the TensorFlow LSTM models. Models are saved to predefined paths.'
    )
    parser_train_tf.add_argument(
        '--use-enhanced-data',
        action='store_true', # Makes it a flag, if present, value is True
        help='Use euromillions_enhanced_dataset.csv for training. Default is euromillions_dataset.csv.'
    )

    args = parser.parse_args()

    if args.command == 'update-data':
        print("Attempting to update data...")
        try:
            update_result = update_euromillions_data() # This now returns a dict
            print(update_result['status_message'])

            latest_draw = update_result.get('latest_draw_data')
            if latest_draw:
                print("\n--- Details of the Latest Fetched Draw ---")
                print(f"  Date: {latest_draw.get('Date')}")
                # Ensure keys exist before trying to format, provide default empty list if not
                numbers_list = [latest_draw.get(f'N{i}') for i in range(1, 6)]
                stars_list = [latest_draw.get(f'E{i}') for i in range(1, 3)]
                # Filter out None values if any key was missing, though latest_draw_data structure should be consistent
                numbers = [n for n in numbers_list if n is not None]
                stars = [s for s in stars_list if s is not None]
                print(f"  Numbers: {numbers}")
                print(f"  Stars: {stars}")

        except Exception as e:
            print(f"An error occurred during data update: {e}")
            # import traceback # For debugging
            # traceback.print_exc()

    elif args.command == 'list-models':
        print("Available prediction models:")
        for model_name in AVAILABLE_MODELS.keys():
            print(f"  - {model_name}")

    elif args.command == 'predict':
        model_func = AVAILABLE_MODELS.get(args.model_name)
        # model_func should always be found due to 'choices' in add_argument
        print(f"Running prediction model: {args.model_name}...")
        try:
            prediction_result = model_func()
            display_prediction(prediction_result)
        except Exception as e:
            print(f"An unexpected error occurred while running model {args.model_name}: {e}")
            # import traceback
            # traceback.print_exc() # For debugging

    elif args.command == 'predict-consensus':
        print("🤝 Running consensus prediction mode...")
        try:
            # Pass the list of model names from args.models
            prediction_result = run_consensus_by_frequency_prediction(selected_model_names=args.models)
            display_prediction(prediction_result) # Use the existing display function
        except Exception as e:
            print(f"An error occurred during consensus prediction: {e}")
            # import traceback # For debugging
            # traceback.print_exc()

    elif args.command == 'train-tf-model':
        print(f"🏋️ Starting TensorFlow model training...")
        print(f"Using enhanced data: {args.use_enhanced_data}")
        try:
            # train_all_models_and_predict now accepts use_enhanced_data
            train_all_models_and_predict(use_enhanced_data=args.use_enhanced_data)
            print("✅ TensorFlow model training process completed.")
            print(f"Models saved based on {'enhanced' if args.use_enhanced_data else 'standard'} dataset configuration.")
            # The predict command will now use tf_lstm_std or tf_lstm_enhanced
            print("You can now use 'predict tf_lstm_std' or 'predict tf_lstm_enhanced'.")
        except Exception as e:
            print(f"An error occurred during TensorFlow model training: {e}")
            import traceback # For debugging
            traceback.print_exc()

    else: # Should not be reached if subparsers are required
        parser.print_help()

if __name__ == '__main__':
    main()
