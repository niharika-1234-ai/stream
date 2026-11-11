# California Housing Prediction App

A beginner-friendly Streamlit interface for the existing California Housing
regression model. It does not train a model. The saved model is bundled at
`model/trained_model.skops`; its fitted `MinMaxScaler` and `LinearRegression`
are loaded together, so predictions use the preprocessing pipeline from
training.

The bundled model is a copy of the `champion` model (version 3) in the local
MLflow registry. Its original artifact is
`mlruns/1/models/m-674a1b5a9e37486dbaa44935666b6468/artifacts/model.skops`.
The app loads the standalone copy and does not need `mlflow.db` or `mlruns` at
runtime. The `.skops` format is retained instead of renaming the model to
`.pkl`; this is the format in which the existing trained model was saved.

## Input features

The form uses the feature names and order expected by the existing model:

| Feature | Meaning | Accepted range |
| --- | --- | ---: |
| `MedInc` | Median household income, in tens of thousands of dollars | 0.4999-15.0001 |
| `HouseAge` | Median house age, in years | 1-52 |
| `AveRooms` | Average number of rooms per household | 0.8889-141.9091 |
| `AveBedrms` | Average number of bedrooms per household | 0.3333-25.6364 |
| `Population` | Block-group population | 3-35682 |
| `AveOccup` | Average household occupancy | 0.6923-1243.3333 |
| `Latitude` | Block-group latitude | 32.55-41.95 |
| `Longitude` | Block-group longitude | -124.35--114.31 |

Ranges are taken from the fitted scaler's observed training data. The target,
`MedHouseVal`, is reported as an estimated US dollar amount; the underlying
model output is in hundreds of thousands of dollars.

## Run locally

Use Python 3.14 to match the Python version used to save the model. From this
project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Fill in
the eight inputs and choose **Predict median house value**. The pre-filled
example is based on the existing model's example input.

## Build and run with Docker

Install and start Docker Desktop, then open a terminal in this project folder.
Build the image:

```powershell
docker build -t ml-prediction-app .
```

Start the container:

```powershell
docker run --rm -p 8501:8501 ml-prediction-app
```

Visit [http://localhost:8501](http://localhost:8501). Stop the container with
**Ctrl+C** in the terminal.

## Deployment steps

1. Keep `app.py`, `requirements.txt`, `Dockerfile`, `.dockerignore`, and
   `model/trained_model.skops` together in this project.
2. Install Docker Desktop and ensure its engine is running.
3. Run `docker build -t ml-prediction-app .` from the project directory.
4. Run `docker run --rm -p 8501:8501 ml-prediction-app`.
5. Open `http://localhost:8501` and make a prediction.

To publish the app on a server, push the image to your container registry and
deploy it with port `8501` exposed. Configure the service to route traffic to
container port `8501`; Streamlit is bound to `0.0.0.0` inside the container.

## Continuous integration

The repository-root
[Docker Build and Validation workflow](../../.github/workflows/docker-build-validation.yml)
runs on pushes to `main` and pull requests targeting `main`. It installs the
project dependencies, runs the model tests, builds an image tagged with the
commit SHA, then starts the container and checks the Streamlit health endpoint
and a sample prediction.

## Project files

```text
app.py
model/
  trained_model.skops
requirements.txt
Dockerfile
.dockerignore
README.md
```
