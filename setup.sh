python -m venv venv_continuum_xgboost
source venv_continuum_xgboost/bin/activate
pip install uv
pip install poetry
uv pip install -r pyproject.toml