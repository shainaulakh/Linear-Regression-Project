# Linear Regression Architecture Workshop

## Project Overview

This project demonstrates the implementation of univariate linear regression and the transition from a notebook-based machine learning workflow to a modular, configuration-driven MLOps architecture.

The project uses California housing data for linear regression modeling and Ontario housing data to demonstrate multiple data ingestion methods, including CSV files, the Statistics Canada API, and a PostgreSQL database hosted on Neon.

Linear regression is implemented both from scratch using gradient descent and with Scikit-learn. The two implementations are evaluated using MAE, RMSE, and R².

The project also demonstrates MLOps concepts including:

- Modular Python components
- Configuration-driven experiments using YAML
- Experiment tracking
- Reproducible dependency management
- Separation of data loading, preprocessing, modeling, and evaluation

## Project Architecture

The project separates the machine learning workflow into reusable components.

```text
Data Sources
├── California Housing CSV
├── Ontario Housing CSV
├── Statistics Canada API
└── Neon PostgreSQL Database
        │
        ▼
src/data_loader.py
        │
        ▼
src/preprocessing.py
        │
        ▼
src/model.py
        │
        ▼
src/evaluation.py
        │
        ▼
experiments/results.csv


The notebooks use these modular Python components instead of containing the entire machine learning workflow directly inside notebook cells.

The experiment settings are stored in `configs/experiment_config.yaml`, allowing parameters such as the selected feature, train/test split, learning rate, and number of iterations to be changed without modifying the model implementation.

```

## Project Structure
```text

LinearRegressionArchitecture_Workshop/
│
├── configs/
│   └── experiment_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── california_housing.csv
│   │   └── ontario_housing.csv
│   └── processed/
│       ├── ontario_housing_clean.csv
│       └── ontario_housing_api.csv
│
├── experiments/
│   └── results.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── linear_regression.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Sources

This project uses housing data from multiple sources to demonstrate different data ingestion methods.

### California Housing Data

The California housing dataset is used for the linear regression experiment. Median income (`MedInc`) is selected as the predictor variable and median house value (`MedHouseVal`) is used as the target variable.

The dataset is stored locally as:

`data/raw/california_housing.csv`

### Ontario Housing Data

Ontario housing data is obtained from Statistics Canada's New Housing Price Index dataset. The project focuses on Ontario and the `Total (house and land)` price index.

The Ontario data is used to demonstrate three data ingestion methods:

1. **CSV** — Housing data is loaded from a local CSV file.
2. **API** — Housing observations are retrieved from the Statistics Canada API.
3. **Database** — Cleaned Ontario housing data is stored in and retrieved from a Neon-hosted PostgreSQL database.

Processed Ontario datasets are stored in the `data/processed/` directory.

Database credentials are stored using environment variables and are not committed to the repository.

## Modeling Approach

The project implements univariate linear regression using two approaches.

### Linear Regression From Scratch

A custom linear regression model is implemented in `src/model.py` using gradient descent.

The hypothesis function is:

h(x) = θ₀ + θ₁x

The model iteratively updates the intercept (θ₀) and slope (θ₁) using a configurable learning rate and number of iterations.

The model parameters are controlled through `configs/experiment_config.yaml`, which helps separate experiment configuration from model implementation.

### Scikit-learn Linear Regression

A Scikit-learn `LinearRegression` model is trained using the same training and testing data as the from-scratch implementation.

This provides a comparison between the custom gradient descent implementation and a standard machine learning library implementation.

### Model Evaluation

Both models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The evaluation logic is contained in `src/evaluation.py`, and experiment results are recorded in `experiments/results.csv`.

## Configuration and Experiment Tracking

### Configuration-Driven Experiments

Experiment settings are stored in:

`configs/experiment_config.yaml`

The configuration file controls important experiment parameters, including:

- Dataset paths
- Selected feature and target
- Train/test split
- Random state
- Learning rate
- Number of gradient descent iterations

Keeping these values in a YAML configuration file reduces hard-coded experiment settings and makes experiments easier to modify and reproduce.

### Experiment Tracking

Model results are stored in:

`experiments/results.csv`

Each experiment records information such as:

- Experiment ID
- Timestamp
- Model name
- Selected feature
- Test size
- Learning rate
- Number of iterations
- MAE
- RMSE
- R²

New experiment results are appended to the results file instead of replacing previous experiments. This allows different experiment runs and model configurations to be compared over time.

### Reproducibility

The project uses a fixed random state for train/test splitting to help produce consistent results across runs.

Python dependencies and their versions are recorded in `requirements.txt` so the project environment can be recreated.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <https://github.com/shainaulakh/Linear-Regression-Project.git>
cd LinearRegressionArchitecture_Workshop
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Database Configuration

The project uses a Neon-hosted PostgreSQL database for the database ingestion demonstration.

Create a `.env` file in the project root and add your database connection string:

```text
DATABASE_URL=your_neon_postgresql_connection_string
```

The `.env` file is excluded from version control to protect database credentials.

## Running the Project

The notebooks are located in the `notebooks/` directory.

Run the notebooks in the following order:

1. `EDA.ipynb` — loads and explores the datasets and demonstrates CSV, API, and database ingestion.
2. `linear_regression.ipynb` — performs preprocessing, trains the from-scratch and Scikit-learn linear regression models, evaluates them, and records experiment results.

Experiment parameters can be modified in:

`configs/experiment_config.yaml`

Results from model experiments are stored in:

`experiments/results.csv`

## Design Decisions

The project was structured to demonstrate the transition from an exploratory notebook workflow to a more modular and reproducible machine learning architecture.

### Separation of Concerns

The machine learning workflow is separated into individual Python modules:

- `data_loader.py` handles data ingestion from CSV files, APIs, and databases.
- `preprocessing.py` handles train/test splitting and feature scaling.
- `model.py` contains the custom linear regression implementation.
- `evaluation.py` contains reusable model evaluation logic.

This separation makes individual components easier to understand, test, reuse, and modify without changing the entire workflow.

### Configuration-Driven Design

Experiment parameters are stored in `experiment_config.yaml` instead of being hard-coded throughout the notebook. This allows experiment settings to be changed independently from the model implementation.

### Environment Variables

Database credentials are stored in a `.env` file rather than directly in the source code. The `.env` file is excluded from Git version control to prevent sensitive credentials from being published.

### Experiment Tracking

Experiment parameters and evaluation metrics are recorded in `experiments/results.csv`. Each experiment is assigned an ID and timestamp so different runs can be identified and compared.

### Modular Notebooks

The notebooks remain responsible for exploration, visualization, and orchestration, while reusable functionality is imported from the `src/` modules. This keeps the notebooks easier to follow while moving core functionality into reusable Python components.

