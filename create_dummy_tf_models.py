import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
import os
import shutil

SEQUENCE_LENGTH = 10 # From euromillions_model.py

# Define model structures
def create_dummy_model(input_features, output_neurons):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, input_features)),
        LSTM(output_neurons * 10, activation='relu'), # Simplified LSTM layer
        Dense(output_neurons, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

# Standard models
main_model_std_path_dir = "models/tf_main_std"
stars_model_std_path_dir = "models/tf_stars_std"
main_model_std_path = os.path.join(main_model_std_path_dir, "final_model.h5")
stars_model_std_path = os.path.join(stars_model_std_path_dir, "final_model.h5")

# Enhanced models
main_model_enhanced_path_dir = "models/tf_main_enhanced"
stars_model_enhanced_path_dir = "models/tf_stars_enhanced"
main_model_enhanced_path = os.path.join(main_model_enhanced_path_dir, "final_model.h5")
stars_model_enhanced_path = os.path.join(stars_model_enhanced_path_dir, "final_model.h5")

# Clean up old euromillions_model_tf directories if they exist
old_main_path_dir = "models/euromillions_model_tf_main"
old_stars_path_dir = "models/euromillions_model_tf_stars"
if os.path.exists(old_main_path_dir):
    shutil.rmtree(old_main_path_dir)
    print(f"Removed old directory: {old_main_path_dir}")
if os.path.exists(old_stars_path_dir):
    shutil.rmtree(old_stars_path_dir)
    print(f"Removed old directory: {old_stars_path_dir}")

# Create and save standard models
os.makedirs(main_model_std_path_dir, exist_ok=True)
main_model_std = create_dummy_model(5, 5)
main_model_std.save(main_model_std_path)
print(f"Dummy standard main model saved to {main_model_std_path}")

os.makedirs(stars_model_std_path_dir, exist_ok=True)
stars_model_std = create_dummy_model(2, 2)
stars_model_std.save(stars_model_std_path)
print(f"Dummy standard stars model saved to {stars_model_std_path}")

# Create and save enhanced models (can be same structure for dummy purposes)
os.makedirs(main_model_enhanced_path_dir, exist_ok=True)
main_model_enhanced = create_dummy_model(5, 5) # Assuming N1-N5 still primary inputs for now
main_model_enhanced.save(main_model_enhanced_path)
print(f"Dummy enhanced main model saved to {main_model_enhanced_path}")

os.makedirs(stars_model_enhanced_path_dir, exist_ok=True)
stars_model_enhanced = create_dummy_model(2, 2)
stars_model_enhanced.save(stars_model_enhanced_path)
print(f"Dummy enhanced stars model saved to {stars_model_enhanced_path}")

print("All dummy TensorFlow models created and saved.")
