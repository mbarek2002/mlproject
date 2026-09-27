<div align="center">

# 🎓 Student Performance Predictor

**End-to-end Machine Learning project: from exploratory analysis to a tested, containerised and continuously deployed web app.**

Predicts a student's **math score** from their profile and their reading & writing scores.

[![CI](https://github.com/mbarek2002/mlproject/actions/workflows/ci.yml/badge.svg)](https://github.com/mbarek2002/mlproject/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.17-0194E2?logo=mlflow&logoColor=white)
![Databricks](https://img.shields.io/badge/Registry-Databricks%20Unity%20Catalog-FF3621?logo=databricks&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)

🌐 **Live demo:** https://student-performance-cgcd.onrender.com/predictdata

<sub>Free Render instance: the first visit after 15 min of inactivity can take ~1 min (cold start).</sub>

</div>

---

## 📑 Table of contents

- [Overview](#-overview)
- [Results](#-results)
- [Architecture](#-architecture)
- [How a prediction works](#-how-a-prediction-works)
- [Model registry: champion / challenger](#-model-registry-champion--challenger)
- [Project structure](#-project-structure)
- [Getting started](#-getting-started)
- [Configuration: local or Databricks registry](#-configuration-local-or-databricks-registry)
- [Usage](#-usage)
- [Tests & CI/CD](#-tests--cicd)
- [Technical choices](#-technical-choices)

---

## 🔎 Overview

| | |
|---|---|
| **Goal** | Predict `math_score` (0–100), a **regression** problem |
| **Dataset** | [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (Kaggle), 1,000 students, 8 columns |
| **Numerical features** | `reading_score`, `writing_score` |
| **Categorical features** | `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course` |
| **Best model** | Linear Regression, **R² = 0.878** on the held-out test set |

The project covers the full ML lifecycle:

1. **EDA**: [`notebooks/1 . EDA STUDENT PERFORMANCE .ipynb`](notebooks/)
2. **Modular training pipeline**: ingestion → transformation → training of 7 models with hyperparameter search
3. **Experiment tracking & model registry** with MLflow, hosted on **Databricks (Unity Catalog)**
4. **Web app** (Flask) with input validation, serving the registry's `@champion` model
5. **Automated tests** (pytest)
6. **Docker image** (490 MB, non-root, health check)
7. **CI/CD**: GitHub Actions → Render

A new model goes live **without redeploying**: training promotes it to `@champion` in Databricks, then the app reloads it.

---

## 📊 Results

7 models, each tuned with `GridSearchCV` (3-fold cross-validation). **The model is selected on the cross-validation score only**; the test set is used once, for the final evaluation.

| Model | CV R² (selection) | Train R² | Test R² |
|---|:---:|:---:|:---:|
| 🥇 **Linear Regression** | **0.868** | 0.874 | **0.878** |
| Gradient Boosting | 0.852 | 0.895 | 0.877 |
| CatBoost | 0.849 | 0.907 | 0.861 |
| Random Forest | 0.832 | 0.977 | 0.855 |
| AdaBoost | 0.827 | 0.841 | 0.843 |
| XGBoost | 0.825 | 0.929 | 0.849 |
| Decision Tree | 0.708 | 1.000 | 0.757 |

**Takeaways**

- `math_score` is almost **linearly** related to reading and writing scores, so the simplest model wins.
- Tree-based models **overfit**: the Decision Tree reaches R² = 1.000 on train but only 0.757 on test.

---

## 🏗 Architecture

```mermaid
flowchart TB
    subgraph TRAIN["🧪 1 · Training pipeline (local)"]
        direction LR
        A[("stud.csv<br/>1,000 students")] --> B["Data ingestion<br/>80 / 20 split"]
        B --> C["Data transformation<br/>impute · scale · one-hot"]
        C --> D["Model trainer<br/>7 models × GridSearchCV"]
    end

    subgraph DBX["☁️ 2 · Databricks (managed MLflow)"]
        direction LR
        M[("Experiment tracking<br/>7 runs per training")]
        UC[("Unity Catalog registry<br/>@champion · @challenger")]
        M ~~~ UC
    end

    subgraph CICD["🚀 3 · CI/CD & serving"]
        direction LR
        G["git push"] --> CI["GitHub Actions<br/>pytest · docker build · smoke test"]
        CI -- "checks pass" --> R["Render<br/>Docker web service"]
    end

    TRAIN -- "logs runs, registers the best model" --> DBX
    DBX -- "@champion loaded at runtime" --> CICD
    TRAIN -. "artifacts/*.pkl in git (fallback)" .-> CICD
    CICD <-- "profile → predicted math score" --> U(["👤 User"])
```

- **Code changes** go live through `git push` → CI → Render.
- **Model changes** go live through the registry: no commit, no image rebuild.

---

## 🔮 How a prediction works

```mermaid
sequenceDiagram
    actor U as User
    participant F as Flask app
    participant P as PredictPipeline
    participant R as Databricks registry
    participant K as artifacts/*.pkl

    U->>F: POST /predictdata (form)
    F->>F: validate_form()
    alt invalid input
        F-->>U: 400 + error message
    else valid input
        F->>P: predict(features DataFrame)
        opt first request only (model is then cached)
            P->>R: load model @champion
            alt registry unavailable
                P->>K: load preprocessor + model
            end
        end
        P-->>F: predicted math_score
        F-->>U: 200 + prediction
    end
```

The model is loaded **once** per process and kept in memory: the first request downloads it from Databricks (~10 s), the next ones take ~0.01 s.

`GET /health` shows which model is served:

```json
{"status": "ok", "model": {"source": "mlflow-registry", "model": "workspace.default.student_math_model", "alias": "champion", "version": "1"}}
```

`"source": "pickle-fallback"` means the registry could not be reached and the app serves `artifacts/*.pkl`.

---

## 🏆 Model registry: champion / challenger

Each training run registers a new version of `workspace.default.student_math_model` in **Databricks Unity Catalog**. The version contains **the preprocessor and the model in a single scikit-learn `Pipeline`**, with its input signature, so it takes raw form data as input. The `@champion` alias (used by the app) only moves if the new version is **strictly better**:

```mermaid
flowchart TD
    A["New training run"] --> B["Best model on CV R²"]
    B --> C["Register new version<br/>(preprocessor + model)"]
    C --> D{"Better test R²<br/>than @champion?"}
    D -- "yes" --> E["🏆 @champion → new version<br/>used by the app"]
    D -- "no" --> F["@challenger → new version<br/>kept for comparison"]
    E --> G["POST /admin/reload-model<br/>the app serves the new champion"]
```

---

## 📁 Project structure

```
mlproject/
├── app.py                        # Flask app: /, /predictdata, /health, /admin/reload-model
├── src/
│   ├── components/
│   │   ├── data_ingection.py     # read + train/test split
│   │   ├── data_transformation.py# ColumnTransformer (impute, scale, one-hot)
│   │   └── model_trainer.py      # 7 models, GridSearchCV, MLflow registry
│   ├── pipeline/
│   │   ├── train_pipeline.py     # entry point of the training
│   │   └── predict_pipeline.py   # model loading (registry → .pkl fallback)
│   ├── mlflow_config.py          # tracking / registry URIs, experiment, model name (from env)
│   ├── utils.py                  # save / load objects, model evaluation
│   ├── logger.py                 # timestamped log files in logs/
│   └── exception.py              # CustomException with file + line number
├── templates/                    # HTML pages (Bootstrap 5)
├── artifacts/                    # model.pkl, preprocessor.pkl, train/test CSV
├── notebooks/                    # EDA + model training notebooks, raw data
├── tests/                        # pytest suite
├── docs/architecture.drawio      # architecture diagram (open in app.diagrams.net)
├── .env.example                  # template for Databricks credentials (.env is git-ignored)
├── Dockerfile                    # production image
├── render.yaml                   # Render deployment blueprint
├── .github/workflows/ci.yml      # CI: tests + docker build + smoke test
├── requirements.txt              # training dependencies (pinned)
├── requirements-prod.txt         # serving dependencies only (Docker image)
└── requirements-dev.txt          # + pytest
```

---

## 🚀 Getting started

**Prerequisites:** Python 3.8, and Docker (optional).

```bash
git clone https://github.com/mbarek2002/mlproject.git
cd mlproject

# with conda
conda create -p ./venv python=3.8 -y
conda activate ./venv

# or with venv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements-dev.txt
```

---

## ⚙️ Configuration: local or Databricks registry

The MLflow location is read from environment variables, loaded from a **`.env`** file in local development.

| Mode | When | Tracking & registry |
|---|---|---|
| **Local** | no `.env` (default) | SQLite file `mlflow.db` |
| **Databricks** | `.env` filled from `.env.example` | Databricks MLflow + Unity Catalog |

To use Databricks ([Free Edition](https://www.databricks.com/learn/free-edition) works):

1. Create a **personal access token**: *Settings → Developer → Access tokens*.
2. Copy `.env.example` to `.env` and fill it in:

| Variable | Example |
|---|---|
| `DATABRICKS_HOST` | `https://dbc-xxxx.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | `dapi...` (secret) |
| `MLFLOW_TRACKING_URI` | `databricks` |
| `MLFLOW_REGISTRY_URI` | `databricks-uc` |
| `MLFLOW_EXPERIMENT_NAME` | `/Users/<email>/student-performance` (training only) |
| `MLFLOW_REGISTERED_MODEL_NAME` | `workspace.default.student_math_model` |
| `RELOAD_TOKEN` | random string protecting `/admin/reload-model` (secret) |

3. In production, set the same variables in **Render → Environment** (`MLFLOW_EXPERIMENT_NAME` is not needed).

> 🔒 `.env` is git-ignored and excluded from the Docker image: secrets only reach the app as environment variables.

---

## 🛠 Usage

### Train the models

```bash
python -m src.pipeline.train_pipeline
```

This runs the full pipeline, writes `artifacts/`, logs every model in MLflow (local or Databricks, depending on the configuration) and registers the best one.

### Explore the experiments

- **Databricks:** *Experiments → student-performance* for the runs, *Catalog → workspace → default → Models* for the versions and aliases.
- **Local:**

  ```bash
  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
  ```

  Then open http://127.0.0.1:5001.

### Put a new champion online (no redeploy)

After a training run that promoted a new `@champion`:

```bash
curl -X POST https://student-performance-cgcd.onrender.com/admin/reload-model \
     -H "Authorization: Bearer $RELOAD_TOKEN"
```

The app loads the new version and returns it. Without the right token it answers `401`, and without `RELOAD_TOKEN` configured the route is disabled (`404`). Restarting the service on Render has the same effect.

### Run the web app

```bash
python app.py
```

Then open http://127.0.0.1:5000/predictdata.

### Run with Docker

```bash
docker build -t student-performance .
docker run -d --name student-app --env-file .env -p 8501:5000 student-performance
```

Then open http://127.0.0.1:8501/predictdata. Without `--env-file .env`, the container serves the `.pkl` fallback.

---

## ✅ Tests & CI/CD

```bash
pytest -v
```

**28 tests** cover the preprocessing, the prediction pipeline and the web app. Among them:

- invalid forms return a **400** with a clear message (never a 500);
- reading and writing scores are **not swapped** between the form and the model;
- the `.pkl` fallback works when the MLflow registry is unavailable;
- `/admin/reload-model` rejects missing or wrong tokens.

Tests always run against the **local** registry, even when a `.env` points to Databricks: they are fast, offline and give the same result on every machine.

**On every push to `main`**, GitHub Actions:

1. runs the tests in `python:3.8-slim`, with the exact production dependencies;
2. builds the Docker image, which **fails if the model cannot be loaded**;
3. starts the container and runs a **smoke test** (`/health` + a real prediction).

**Render deploys only once these checks pass** (`autoDeployTrigger: checksPass`).

---

## 🧠 Technical choices

| Choice | Why |
|---|---|
| Model selected on **CV R²**, not test R² | The test set stays an unbiased estimate of real performance |
| **Preprocessor + model** registered as one `Pipeline` | A registry version is self-contained: raw input in, prediction out |
| **Champion / challenger** aliases | A worse retraining can never replace the model in production |
| **Databricks Unity Catalog** as registry | A shared, managed registry: the deployed app reads the same `@champion` as the training |
| **Training locally, registry remotely** | No Databricks compute used; the training code is unchanged |
| **Hot reload** behind a token | New models go live in seconds, without a commit or an image rebuild |
| **`.pkl` fallback** + short MLflow timeouts | The app keeps working (and never hangs) if Databricks is unreachable |
| Configuration through **environment variables** | Same code and image everywhere; secrets never in git nor in the image |
| **Pinned versions** | A pickled model must be loaded with the scikit-learn version that created it |
| **Separate `requirements-prod.txt`** | No XGBoost / CatBoost in the image: **2 GB → 490 MB** |
| **waitress + tini**, non-root user | Production WSGI server, clean shutdown, least privilege |

---

<div align="center">

**Iheb Mbarek** · [GitHub @mbarek2002](https://github.com/mbarek2002)

</div>
