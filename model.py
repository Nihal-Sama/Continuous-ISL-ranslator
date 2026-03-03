from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Conv1D, MaxPooling1D, BatchNormalization

def build_model(sequence_length, num_features, num_classes):
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        # Reduced from 256 to 128 to prevent overfitting
        GRU(128, return_sequences=True, activation='relu'),
        # Increased Dropout from 0.4 to 0.5 (forces AI to rely on multiple features, not just one)
        Dropout(0.5), 
        
        # Reduced from 128 to 64
        GRU(64, return_sequences=False, activation='relu'),
        
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model