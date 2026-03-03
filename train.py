import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model

PROCESSED_PATH = os.path.join('data', 'processed')
MODELS_PATH = 'models'

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

actions = []
sequences = []
labels = []

if os.path.exists(PROCESSED_PATH):
    actions = sorted([f for f in os.listdir(PROCESSED_PATH) if os.path.isdir(os.path.join(PROCESSED_PATH, f))])
else:
    print("ERROR: Processed data not found.")
    exit()

label_map = {label: num for num, label in enumerate(actions)}
SEQUENCE_LENGTH = 80  

print(f"Loading and Augmenting data for {len(actions)} classes...")

def augment_sequence(seq, noise_level=0.015):
    noise = np.random.normal(0, noise_level, seq.shape)
    return seq + noise

def scale_sequence(seq, scale_factor):
    return seq * scale_factor

for action in actions:
    action_path = os.path.join(PROCESSED_PATH, action)
    file_list = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    
    for file in file_list:
        data = np.load(os.path.join(action_path, file))
        
        # --- UPGRADE: Uniform Frame Sampling ---
        # Replaces the old zero-padding. Stretches/compresses smoothly.
        if len(data) != SEQUENCE_LENGTH:
            indices = np.linspace(0, len(data) - 1, SEQUENCE_LENGTH).astype(int)
            data = data[indices]
        # ---------------------------------------
            
        sequences.append(data)
        labels.append(label_map[action])
        
        sequences.append(augment_sequence(data, noise_level=0.01))
        labels.append(label_map[action])

        sequences.append(scale_sequence(data, scale_factor=0.95))
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

np.save(os.path.join(MODELS_PATH, 'actions.npy'), actions)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

model = build_model(SEQUENCE_LENGTH, X.shape[2], len(actions))

checkpoint = ModelCheckpoint(os.path.join(MODELS_PATH, 'best_isl_model.keras'), 
                             monitor='val_categorical_accuracy', 
                             save_best_only=True, 
                             mode='max')

early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

print(f"Total training sequences after augmentation: {len(X)}")
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
print("✅ Training complete.")