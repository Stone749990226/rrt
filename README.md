uvicorn app2:app --log-level info --port 8123 --host 127.0.0.1 > uvicorn.log 2>&1

pytest -s test_app.py