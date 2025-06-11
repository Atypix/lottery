import argparse
import sys
import os

# Add the parent directory to sys.path to find fetch_real_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_real_data import update_euromillions_data
from predicteur_final_valide import FinalValidatedPredictor
from revolutionary_predictor import RevolutionaryPredictor # Renamed import
from aggregated_final_predictor import AggregatedFinalPredictor
from euromillions_model import predict_with_tensorflow_model

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

AVAILABLE_MODELS = {
    'final_valide': run_final_valide,
    'revolutionnaire': run_revolutionnaire,
    'agrege': run_agrege,
    'tf_lstm': predict_with_tensorflow_model
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
    parser_predict.add_argument('model_name', choices=AVAILABLE_MODELS.keys(), metavar='model_name', help='Name of the prediction model to use')

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

    else: # Should not be reached if subparsers are required
        parser.print_help()

if __name__ == '__main__':
    main()
