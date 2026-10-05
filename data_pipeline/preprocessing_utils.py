import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean(path, cols_to_remove):
    print("Chargement et nettoyage initial...")
    df = pd.read_csv(path)
    # Supprimer colonnes inutiles
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])
    # Gérer les Infinity et NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().drop_duplicates()
    # Encodage binaire du Label
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    return df

def normalize(df):
    print("Normalisation Min-Max...")
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def create_lstm_sequences(data, window_size):
    print(f"Création des séquences (Fenêtre : {window_size})...")
    sequences = []
    # On enlève le label pour les séquences X
    values = data.drop(columns=['Label']).values
    for i in range(len(values) - window_size):
        sequences.append(values[i:i+window_size])
    return np.array(sequences)

def visualize_data(df):
    print("Génération des visualisations...")
    plt.figure(figsize=(12, 5))
    
    # Graphique 1 : Répartition des classes
    plt.subplot(1, 2, 1)
    sns.countplot(x='Label', data=df)
    plt.title('Répartition : Normal (0) vs Attaque (1)')
    
    # Graphique 2 : Heatmap de corrélation (10 premières features)
    plt.subplot(1, 2, 2)
    corr = df.iloc[:, :10].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de Corrélation')
    
    plt.tight_layout()
    plt.savefig('data_visualizations.png')
    plt.close()