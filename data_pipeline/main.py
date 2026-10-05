import config
import preprocessing_utils as utils

def run_pipeline():
    # 1. Charger et Nettoyer
    df = utils.load_and_clean(config.DATA_PATH, config.COLS_TO_DROP)
    
    # 2. Visualiser (Avant normalisation pour mieux voir les échelles)
    utils.visualize_data(df)
    
    # 3. Normaliser
    df_normalized = utils.normalize(df)
    
    # 4. Créer les séquences pour le modèle de Walid/Meryem/Noussair
    X_sequences = utils.create_lstm_sequences(df_normalized, config.WINDOW_SIZE)
    
    print("\n--- BILAN DU PIPELINE ---")
    print(f"Lignes traitées : {len(df_normalized)}")
    print(f"Séquences LSTM générées : {X_sequences.shape}")
    print("Prêt pour l'entraînement TensorFlow !")

if __name__ == "__main__":
    run_pipeline()