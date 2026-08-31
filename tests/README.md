# Tests

Pytest suites live next to the code they cover (CI paths):

- `ml/tests/` — model / preprocess (`cd ml && pytest tests/`)
- `app/backend/tests/` — FastAPI (`cd app/backend && pytest tests/`)
- `tools/selftest.py` — dataset toolkit

Do not move those directories; CI and Docker expect them.
