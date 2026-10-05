# Data Pipeline

**Author:** Safaa Khalladi

First-pass cleaning of a raw CICFlowMeter CSV: drops identifier columns, removes NaN/Inf rows and duplicates, applies Min-Max normalisation, and plots the class balance and a feature correlation heatmap (`data_visualizations.png`).

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python data_pipeline/main.py
```

Input and output paths are set in `config.py` and can be overridden with the `IDS_DATA_PATH` and `IDS_SAVE_PATH` environment variables.

The model in `ai_detection/` is trained from its own, more extensive preprocessing (`ai_detection/src/data_processing/preprocess.py`), which handles all ten capture days, feature selection and the train/val/test split.
