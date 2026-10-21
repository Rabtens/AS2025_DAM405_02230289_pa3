#!/usr/bin/env bash
# End-to-end run: train + track + register -> explain -> serve -> traffic -> tests.
set -e
cd "$(dirname "$0")"
export MLFLOW_DISABLE_AGENT_HINT=1

[ -f data/adult.csv ] || python prepare_data.py
python train.py                      # baselines + FLAML, MLflow tracking, registry
python explain.py                    # SHAP explanations of the champion

python serve.py &                    # start the monitored service
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
for i in $(seq 1 30); do curl -s localhost:8000/health >/dev/null && break; sleep 1; done
python simulate_traffic.py           # normal, invalid, unusual and drifted traffic

python -m pytest -v tests/
echo "Done. Browse runs with:  mlflow ui   (http://localhost:5000)"
