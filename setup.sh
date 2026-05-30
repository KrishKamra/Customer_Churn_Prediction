#!/bin/bash
# Create virtual environment
python -m venv churn_env

# Activate
source churn_env/Scripts/activate || source churn_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete. Use 'source churn_env/Scripts/activate' to start."