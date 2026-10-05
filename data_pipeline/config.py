import os

# Chemins surchargeables par variables d'environnement.
DATA_PATH = os.environ.get("IDS_DATA_PATH", "data/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
SAVE_PATH = os.environ.get("IDS_SAVE_PATH", "data/data_nettoyee.csv")

# Paramètres techniques
WINDOW_SIZE = 10
COLS_TO_DROP = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP']
