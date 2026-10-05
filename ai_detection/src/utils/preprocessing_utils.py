import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_and_clean_multiple(data_dir, csv_files, cols_to_remove, sample_size=None):
    """Load and clean multiple CSV files with sampling"""
    print(f"Chargement de {len(csv_files)} fichiers CSV...")
    if sample_size:
        print(f"  Mode échantillonnage: {sample_size:,} lignes par fichier")
    
    dfs = []
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        if os.path.exists(file_path):
            print(f"  - Chargement: {csv_file}")
            
            # Compter les lignes d'abord
            with open(file_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # -1 pour header
            
            if sample_size and total_rows > sample_size:
                # Échantillonner aléatoirement
                skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                                   total_rows - sample_size, 
                                                   replace=False))
                df = pd.read_csv(file_path, skiprows=skip_rows, low_memory=False)
                print(f"    Échantillonné: {len(df):,}/{total_rows:,} lignes")
            else:
                df = pd.read_csv(file_path, low_memory=False)
                print(f"    Chargé: {len(df):,} lignes")
            
            # Nettoyer immédiatement pour économiser la mémoire
            df = clean_dataframe(df, cols_to_remove)
            dfs.append(df)
        else:
            print(f"  ⚠ Fichier non trouvé: {csv_file}")
    
    # Combiner tous les DataFrames
    print("\nCombination des fichiers...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total lignes après nettoyage: {len(df):,}")
    
    return df

def clean_dataframe(df, cols_to_remove):
    """Clean a single dataframe"""
    initial_rows = len(df)
    
    # Supprimer colonnes inutiles
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')
    
    # Encodage binaire du Label AVANT nettoyage (pour préserver les infos)
    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip() == 'Benign' else 1)
    
    # Convertir toutes les colonnes numériques
    numeric_cols = df.columns[df.columns != 'Label']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Gérer les Infinity et NaN (plus efficace)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Supprimer les doublons
    df = df.drop_duplicates()
    
    # Supprimer les colonnes avec trop de valeurs manquantes ou constantes
    if len(df) > 0:
        # Supprimer colonnes avec variance nulle (constantes)
        numeric_only = df.select_dtypes(include=[np.number])
        constant_cols = numeric_only.columns[numeric_only.nunique() <= 1].tolist()
        if constant_cols and 'Label' not in constant_cols:
            df = df.drop(columns=constant_cols, errors='ignore')
    
    return df

def normalize(df, scaler=None, save_path=None):
    """Normalize features (excluding Label column)"""
    print("Normalisation Min-Max des features...")
    
    # Séparer features et label
    if 'Label' in df.columns:
        labels = df['Label'].copy()
        features = df.drop(columns=['Label'])
    else:
        labels = None
        features = df.copy()
    
    # Identifier et supprimer les colonnes avec seulement des NaN
    nan_cols = features.columns[features.isna().all()].tolist()
    if nan_cols:
        print(f"  ⚠ Suppression de {len(nan_cols)} colonnes vides (all NaN)")
        features = features.drop(columns=nan_cols)
    
    # Identifier les colonnes avec des NaN partiels
    partial_nan_cols = features.columns[features.isna().any()].tolist()
    if partial_nan_cols:
        print(f"  ⚠ {len(partial_nan_cols)} colonnes contiennent des NaN partiels - remplissage avec 0")
        features = features.fillna(0)
    
    # Créer ou utiliser le scaler existant
    if scaler is None:
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features)
        print("  - Nouveau scaler créé et fitted")
    else:
        features_normalized = scaler.transform(features)
        print("  - Scaler existant utilisé")
    
    # Reconstruire le DataFrame
    df_normalized = pd.DataFrame(features_normalized, columns=features.columns, index=features.index)
    
    # Remettre le label (NON normalisé)
    if labels is not None:
        df_normalized['Label'] = labels.values
    
    # Sauvegarder le scaler
    if save_path and scaler is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  - Scaler sauvegardé: {save_path}")
    
    print(f"  - Features finales: {features_normalized.shape[1]} colonnes")
    
    return df_normalized, scaler

def create_lstm_sequences(data, window_size):
    """Create sequences for LSTM with corresponding labels"""
    print(f"Création des séquences LSTM (fenêtre={window_size})...")
    
    # Séparer features et labels
    if 'Label' in data.columns:
        features = data.drop(columns=['Label']).values
        labels = data['Label'].values
    else:
        features = data.values
        labels = None
    
    X_sequences = []
    y_labels = []
    
    # Créer des séquences glissantes
    for i in range(len(features) - window_size):
        X_sequences.append(features[i:i+window_size])
        if labels is not None:
            # Le label de la dernière observation de la séquence
            y_labels.append(labels[i+window_size])
    
    X = np.array(X_sequences)
    y = np.array(y_labels) if labels is not None else None
    
    print(f"  - Séquences X créées: {X.shape}")
    if y is not None:
        print(f"  - Labels y créés: {y.shape}")
        print(f"  - Distribution des labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets"""
    print(f"\nSéparation des données...")
    
    # Split train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train / val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"  - Train: {X_train.shape} | Labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  - Val:   {X_val.shape} | Labels: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"  - Test:  {X_test.shape} | Labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def balance_classes(df):
    """Balance classes by undersampling the majority class"""
    print("\nÉquilibrage des classes...")
    
    if 'Label' not in df.columns:
        return df
    
    # Compter les classes
    class_counts = df['Label'].value_counts()
    print(f"  - Avant équilibrage: {class_counts.to_dict()}")
    
    # Trouver la classe minoritaire
    min_class_size = class_counts.min()
    
    # Sous-échantillonner chaque classe
    df_balanced = pd.concat([
        df[df['Label'] == label].sample(n=min_class_size, random_state=42)
        for label in df['Label'].unique()
    ])
    
    # Mélanger
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    class_counts_after = df_balanced['Label'].value_counts()
    print(f"  - Après équilibrage: {class_counts_after.to_dict()}")
    print(f"  - Total échantillons: {len(df_balanced)}")
    
    return df_balanced

def visualize_data(df, save_path='data_visualizations_before.png'):
    """Generate comprehensive visualizations BEFORE normalization"""
    print("Génération des visualisations (avant normalisation)...")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Répartition des classes
    plt.subplot(3, 4, 1)
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts().sort_index()
        colors = ['green', 'red']
        plt.bar(['Benign (0)', 'Attack (1)'], label_counts.values, color=colors, alpha=0.7)
        plt.title('Class Distribution', fontsize=12, fontweight='bold')
        plt.ylabel('Count')
        for i, v in enumerate(label_counts.values):
            plt.text(i, v, str(v), ha='center', va='bottom')
    
    # 2. Distribution des features (histogrammes)
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'Label'][:6]
    for idx, col in enumerate(numeric_cols, 2):
        plt.subplot(3, 4, idx)
        df[col].hist(bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.title(f'{col[:20]}...', fontsize=9)
        plt.ylabel('Frequency')
    
    # 8. Boxplots par classe
    plt.subplot(3, 4, 8)
    if 'Label' in df.columns and len(numeric_cols) > 0:
        data_to_plot = [df[df['Label']==0][numeric_cols[0]].dropna(), 
                        df[df['Label']==1][numeric_cols[0]].dropna()]
        plt.boxplot(data_to_plot, labels=['Benign', 'Attack'])
        plt.title(f'{numeric_cols[0][:20]} by Class', fontsize=10)
        plt.ylabel('Value')
    
    # 9. Correlation heatmap
    plt.subplot(3, 4, 9)
    corr_cols = numeric_cols[:8]
    if len(corr_cols) > 1:
        corr = df[corr_cols].corr()
        sns.heatmap(corr, annot=False, cmap='RdYlBu_r', center=0, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation', fontsize=10)
    
    # 10. Valeurs nulles
    plt.subplot(3, 4, 10)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        null_counts[null_counts > 0][:10].plot(kind='barh', color='orange')
        plt.title('Null Values', fontsize=10)
    else:
        plt.text(0.5, 0.5, 'No Null Values', ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # 11. Statistiques descriptives
    plt.subplot(3, 4, 11)
    stats_text = f"Dataset Statistics\n{'='*30}\n"
    stats_text += f"Total Rows: {len(df):,}\n"
    stats_text += f"Total Features: {len(numeric_cols)}\n"
    if 'Label' in df.columns:
        stats_text += f"Benign: {(df['Label']==0).sum():,}\n"
        stats_text += f"Attack: {(df['Label']==1).sum():,}\n"
        balance_ratio = (df['Label']==1).sum() / (df['Label']==0).sum()
        stats_text += f"Attack/Benign Ratio: {balance_ratio:.2f}"
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')
    plt.axis('off')
    
    # 12. Feature value ranges
    plt.subplot(3, 4, 12)
    if len(numeric_cols) > 0:
        ranges = df[numeric_cols[:8]].max() - df[numeric_cols[:8]].min()
        ranges.plot(kind='bar', color='teal', alpha=0.7)
        plt.title('Feature Value Ranges', fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.ylabel('Range')
    
    plt.suptitle('Data Analysis - BEFORE Preprocessing', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def visualize_normalized_data(df, save_path='data_visualizations_after.png'):
    """Generate comprehensive visualizations AFTER normalization"""
    print("Génération des visualisations (après normalisation)...")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Class distribution
    plt.subplot(3, 4, 1)
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts().sort_index()
        colors = ['green', 'red']
        plt.bar(['Benign (0)', 'Attack (1)'], label_counts.values, color=colors, alpha=0.7)
        plt.title('Class Distribution (After Balancing)', fontsize=12, fontweight='bold')
        plt.ylabel('Count')
        for i, v in enumerate(label_counts.values):
            plt.text(i, v, str(v), ha='center', va='bottom')
    
    # 2-7. Normalized feature distributions
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'Label'][:6]
    for idx, col in enumerate(numeric_cols, 2):
        plt.subplot(3, 4, idx)
        df[col].hist(bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.title(f'{col[:20]}... (Normalized)', fontsize=9)
        plt.ylabel('Frequency')
        plt.xlim([0, 1])
    
    # 8. Feature distributions by class
    plt.subplot(3, 4, 8)
    if 'Label' in df.columns and len(numeric_cols) > 0:
        df[df['Label']==0][numeric_cols[0]].hist(bins=30, alpha=0.5, label='Benign', color='green')
        df[df['Label']==1][numeric_cols[0]].hist(bins=30, alpha=0.5, label='Attack', color='red')
        plt.title(f'{numeric_cols[0][:20]} Distribution', fontsize=10)
        plt.legend()
        plt.ylabel('Frequency')
    
    # 9. Correlation after normalization
    plt.subplot(3, 4, 9)
    corr_cols = numeric_cols[:8]
    if len(corr_cols) > 1:
        corr = df[corr_cols].corr()
        sns.heatmap(corr, annot=False, cmap='RdYlBu_r', center=0, cbar_kws={'shrink': 0.8})
        plt.title('Normalized Correlation', fontsize=10)
    
    # 10. Value ranges (should be 0-1)
    plt.subplot(3, 4, 10)
    if len(numeric_cols) > 0:
        mins = df[numeric_cols[:8]].min()
        maxs = df[numeric_cols[:8]].max()
        x = np.arange(len(mins))
        plt.bar(x - 0.2, mins, 0.4, label='Min', color='blue', alpha=0.7)
        plt.bar(x + 0.2, maxs, 0.4, label='Max', color='red', alpha=0.7)
        plt.title('Min/Max Values (0-1 Range)', fontsize=10)
        plt.xticks(x, [col[:10] for col in mins.index], rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.ylim([0, 1.1])
    
    # 11. Statistics after preprocessing
    plt.subplot(3, 4, 11)
    stats_text = f"After Preprocessing\n{'='*30}\n"
    stats_text += f"Total Rows: {len(df):,}\n"
    stats_text += f"Features: {len(numeric_cols)}\n"
    if 'Label' in df.columns:
        stats_text += f"Benign: {(df['Label']==0).sum():,}\n"
        stats_text += f"Attack: {(df['Label']==1).sum():,}\n"
        balance_ratio = (df['Label']==1).sum() / (df['Label']==0).sum()
        stats_text += f"Balance Ratio: {balance_ratio:.2f}\n"
    stats_text += f"\nAll features normalized to [0,1]"
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')
    plt.axis('off')
    
    # 12. Label distribution pie chart
    plt.subplot(3, 4, 12)
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts().sort_index()
        wedges, texts, autotexts = plt.pie(label_counts.values, labels=['Benign', 'Attack'], 
                autopct='%1.1f%%', colors=['green', 'red'])
        for w in wedges:
            w.set_alpha(0.7)
        plt.title('Class Balance', fontsize=10)
    
    plt.suptitle('Data Analysis - AFTER Preprocessing & Balancing', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def visualize_sequences(X, y, save_path='sequence_visualizations.png'):
    """Visualize LSTM sequences"""
    print("Génération des visualisations de séquences...")
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Sequence shape info
    plt.subplot(2, 3, 1)
    info_text = f"LSTM Sequences\n{'='*40}\n\n"
    info_text += f"X shape: {X.shape}\n"
    info_text += f"  - Samples: {X.shape[0]:,}\n"
    info_text += f"  - Timesteps: {X.shape[1]}\n"
    info_text += f"  - Features: {X.shape[2]}\n\n"
    info_text += f"y shape: {y.shape}\n"
    info_text += f"  - Labels: {y.shape[0]:,}\n\n"
    info_text += f"Memory: ~{X.nbytes / 1024 / 1024:.1f} MB"
    plt.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center', family='monospace')
    plt.axis('off')
    plt.title('Sequence Information', fontsize=12, fontweight='bold')
    
    # 2. Label distribution in sequences
    plt.subplot(2, 3, 2)
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(['Benign (0)', 'Attack (1)'], counts, color=['green', 'red'], alpha=0.7)
    plt.title('Label Distribution in Sequences', fontsize=12, fontweight='bold')
    plt.ylabel('Count')
    for i, v in enumerate(counts):
        plt.text(i, v, f'{v:,}\n({v/len(y)*100:.1f}%)', ha='center', va='bottom')
    
    # 3. Sample benign sequence
    plt.subplot(2, 3, 3)
    benign_idx = np.where(y == 0)[0][0]
    plt.imshow(X[benign_idx].T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Normalized Value')
    plt.title(f'Sample Benign Sequence (idx={benign_idx})', fontsize=11)
    plt.xlabel('Timestep')
    plt.ylabel('Feature')
    
    # 4. Sample attack sequence
    plt.subplot(2, 3, 4)
    attack_idx = np.where(y == 1)[0][0]
    plt.imshow(X[attack_idx].T, aspect='auto', cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Normalized Value')
    plt.title(f'Sample Attack Sequence (idx={attack_idx})', fontsize=11)
    plt.xlabel('Timestep')
    plt.ylabel('Feature')
    
    # 5. Average benign vs attack patterns
    plt.subplot(2, 3, 5)
    avg_benign = X[y == 0].mean(axis=0).mean(axis=1)
    avg_attack = X[y == 1].mean(axis=0).mean(axis=1)
    feature_indices = np.arange(len(avg_benign))
    plt.plot(feature_indices, avg_benign, 'g-', label='Avg Benign', linewidth=2, alpha=0.7)
    plt.plot(feature_indices, avg_attack, 'r-', label='Avg Attack', linewidth=2, alpha=0.7)
    plt.title('Average Feature Values: Benign vs Attack', fontsize=11)
    plt.xlabel('Feature Index')
    plt.ylabel('Average Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Temporal patterns (first 5 features over time)
    plt.subplot(2, 3, 6)
    seq_idx = np.where(y == 1)[0][0]
    for feat_idx in range(min(5, X.shape[2])):
        plt.plot(X[seq_idx, :, feat_idx], label=f'Feature {feat_idx}', alpha=0.7)
    plt.title(f'Temporal Pattern - Attack Sequence {seq_idx}', fontsize=11)
    plt.xlabel('Timestep')
    plt.ylabel('Normalized Value')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('LSTM Sequence Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    """Save processed data as numpy arrays"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSauvegarde des données dans: {output_dir}")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print("  ✓ Tous les fichiers sauvegardés")