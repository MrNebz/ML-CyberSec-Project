# CICIoT2023 IoT Intrusion Detection System

This project is an end-to-end machine-learning intrusion detection system for
the CICIoT2023 dataset. It includes trained model artifacts, a FastAPI inference
service, a Streamlit command-center dashboard, prediction logging, and drift
monitoring for live/demo traffic.

The application serves locally available models from `data/artifacts/`, accepts
CICIoT2023 feature rows through an API, logs predictions to SQLite, and displays
model metrics, live predictions, and drift signals in Streamlit.

## Project Structure

```text
src/serve/       FastAPI app, schemas, preprocessing, model loading
src/dashboard/   Streamlit dashboard
src/monitor/     SQLite prediction store and drift detectors
src/client/      CSV replay client for demo traffic
data/artifacts/  Trained models, scalers, labels, metrics, confusion matrices
data/processed/  Processed train/test splits used by Streamlit and drift refs
data/drift/      Step 7 drift demo CSVs
notebooks/       Training and preprocessing notebooks
scripts/         Helper scripts
docker/          Container startup script
```

## Included Runtime Data

The Docker image includes the processed splits required by Streamlit:

```text
data/processed/X_train_raw.csv
data/processed/y_train_encoded.csv
data/processed/X_test_raw.csv
data/processed/y_test_encoded.csv
```

Other large processed variants, raw data, subset files, and the prediction
database are excluded from Docker.

The training subset was built by capping frequent classes at 35,000 rows per
class before creating the model splits.

Random Forest artifacts are extremely large, with the selected model around
19 GB, so the public Docker image excludes the Random Forest model file. The
API still lists it as registered but unavailable inside Docker. DNN, Logistic
Regression, and Perceptron are included and usable.

For the Docker demo, the required model files are already packaged inside the
published Docker image. A user who runs the Docker image does not need to
download model files separately from GitHub. GitHub can omit large model/data
files as long as the public Docker image is used for the runnable demo.

## Local Setup

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run Locally

Start the FastAPI inference service:

```bash
uvicorn src.serve.main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal, start the Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

Open:

```text
API docs:  http://127.0.0.1:8000/docs
Dashboard: http://127.0.0.1:8501
```

The dashboard expects the API endpoint to be:

```text
http://127.0.0.1:8000
```

## API Endpoints

Useful endpoints:

```text
GET  /health
GET  /features
GET  /classes
GET  /models
GET  /models/{key}/metrics
GET  /models/{key}/confusion_matrix
GET  /models/{key}/confusion_matrix.png
POST /predict?model={key}
GET  /drift/status?model={key}
GET  /drift/confidence_history?model={key}
GET  /drift/feature_analysis?model={key}
POST /drift/reset?model={key}
```

Example model keys:

```text
deep_neural_network
logistic_regression
perceptron
random_forest
```

## Demo Traffic

After the API is running, replay test traffic into the prediction endpoint:

```bash
python -m src.client.stream_test --model deep_neural_network --limit 1000 --api http://127.0.0.1:8000
```

Replay Step 7 drift traffic:

```bash
python -m src.client.stream_test --model deep_neural_network --x data/drift/step7/X_drift_test.csv --y data/drift/step7/y_drift_test.csv --api http://127.0.0.1:8000
```

You can also run traffic directly from the Streamlit dashboard in the SOC
Monitor tab.

## Docker Build

Build the Docker image:

```bash
docker build -t mathieumoussa/ciciot2023-ids:latest .
```

The Dockerfile starts both services:

```text
FastAPI:   port 8000
Streamlit: port 8501
```

## Docker Run

Run the container:

```bash
docker run --rm -p 8000:8000 -p 8501:8501 mathieumoussa/ciciot2023-ids:latest
```

Open:

```text
API docs:  http://localhost:8000/docs
Dashboard: http://localhost:8501
```

In the Streamlit sidebar inside Docker, use this API endpoint:

```text
http://127.0.0.1:8000
```

## Docker Hub Publish

Log in:

```bash
docker login
```

Push the image:

```bash
docker push mathieumoussa/ciciot2023-ids:latest
```

The public image is available on Docker Hub:

```text
https://hub.docker.com/repository/docker/mathieumoussa/ciciot2023-ids
```

Anyone can then run:

```bash
docker pull mathieumoussa/ciciot2023-ids:latest
docker run --rm -p 8000:8000 -p 8501:8501 mathieumoussa/ciciot2023-ids:latest
```

## Docker Smoke Test

With the container running:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
```

Expected health response:

```json
{"status":"ok"}
```

## Notes

The prediction log is stored in:

```text
data/monitor/predictions.db
```

This file is created automatically at runtime and is intentionally not included
in Docker or Git.

If you want to persist prediction logs outside the container, run Docker with a
volume:

```bash
docker run --rm -p 8000:8000 -p 8501:8501 -v "%cd%/data/monitor:/app/data/monitor" mathieumoussa/ciciot2023-ids:latest
```

On macOS/Linux:

```bash
docker run --rm -p 8000:8000 -p 8501:8501 -v "$PWD/data/monitor:/app/data/monitor" mathieumoussa/ciciot2023-ids:latest
```
