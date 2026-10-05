@echo off
chcp 65001 > nul
title IDS-Blockchain Dashboard

echo.
echo  ==========================================
echo    IDS-Blockchain Dashboard - Demarrage
echo  ==========================================
echo.

:: Vérifier Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python non trouve. Installe Python 3.10+.
    pause
    exit /b 1
)

:: Installer les dépendances
echo [1/3] Installation des dependances...
pip install -r dashboard\requirements.txt -q
if errorlevel 1 (
    echo [ERREUR] Echec installation des dependances.
    pause
    exit /b 1
)
echo       OK

:: Vérifier Docker (optionnel — Kafka sera démarré automatiquement si disponible)
docker info > nul 2>&1
if errorlevel 1 (
    echo [INFO]  Docker non disponible - le dashboard demarre en mode simulateur.
) else (
    echo [2/3] Docker detecte - Kafka sera demarre automatiquement par le dashboard.
)

echo [3/3] Lancement du dashboard Streamlit...
echo.
echo  Ouvre http://localhost:8501 dans ton navigateur.
echo  Appuie sur Ctrl+C pour arreter.
echo.

cd dashboard
streamlit run app.py
