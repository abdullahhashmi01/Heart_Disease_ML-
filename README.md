## End to End ML Project  

A professional, production-grade End-to-End Machine Learning project engineered to predict the likelihood of clinical heart disease using clinical attributes from the UCI Dataset. This repository implements clean software engineering architectural patterns, modular data processing components, structured feature engineering pipelines, advanced hyperparameter tuning via cross-validation grids, and automated experiment tracking integration using MLflow and DagsHub.



## 🚀 Key Features
* **Modular Pipeline Architecture:** Enforces strict separation of concerns into isolated executable blocks (Data Ingestion $\rightarrow$ Data Transformation $\rightarrow$ Model Training).
* **Centralized Orchestration:** Features an automated runner pipeline (`app.py`) that sequentially drives artifact creation, preprocessing array bindings, and model registration.
* **Production-Grade Infrastructure:** Custom execution runtime logs dynamically structured inside timestamped files, combined with contextual file-and-line specific custom exceptions.
* **Robust Feature Transformation:** Comprehensive multi-channel standard scaling, median imputations for missing segments, and zero-leak categorical one-hot mappings via reusable pipelines.
* **Advanced Experiment Tracking:** Continuous automated logging of model hyperparameters, baseline weights, training arrays, evaluation charts, and localized artifacts natively inside online **MLflow** dashboards tracking directly to **DagsHub**.

---

## 📂 Project Directory Layout
```bash
.
├── artifacts/                          # Sequential execution tracking & serialized data
│   ├── raw.csv                         # Local database snapshot copy
│   ├── train.csv                       # 80% Stratified training set 
│   ├── test.csv                        # 20% Automated split test set
│   ├── preprocessor.pkl                # Serialized ColumnTransformer preprocessing object
│   └── model.pkl                       # Final certified best production model binary
├── Heart_Disease_Prediction.egg-info/  # Package discovery distribution metadata
├── logs/                               # Dynamic chronological application diaries
│   └── <YYYY-MM-DD_HH-MM-SS>.log       # Real-time execution tracking log captures
├── notebook/                           # Experimental R&D sandbox environment
│   ├── EDA_Heart_Disease.ipynb         # Exploratory analysis & statistical charts
│   ├── Model_Training.ipynb            # Baseline testing & feature importance analysis
│   └── data/
│       └── heart.csv                   # Source raw dataset files (UCI Repository)
├── src/                                # Production application codebase
│   ├── Heart_Disease_Prediction/
│   │   ├── components/
│   │   │   ├── data_ingestion.py       # Manages file reading, verification, and splits
│   │   │   ├── data_transformation.py  # Constructs transformers, processes matrices, saves states
│   │   │   └── model_trainer.py        # Optimizes estimators, prints reports, ships to MLflow
│   │   ├── exception.py                # Contextual error capture module
│   │   ├── logger.py                   # Central run logging setup
│   │   └── utils.py                    # Object handlers and grid estimation functions
│   └── __init__.py
├── .gitignore                          # Git asset filtration constraints
├── app.py                              # Central orchestration runner framework
├── README.md                           # Comprehensive user documentation center
├── requirements.txt                    # Project prerequisite modules
├── setup.py                            # Python package generation management
└── streamlit_app.py                    # Production Streamlit UI Dashboard framework

 ** End-to-End Pipeline Workflow **
1. Centralized Orchestration Runner (app.py)
The system utilizes a central main file execution wrapper to coordinate the workflow pipelines step-by-step. Instead of firing separate decoupled manual components, triggering app.py sets up a programmatic flow:

[app.py Entry Point]
       │
       ├──► 1. Data Ingestion ──► Generates (raw.csv, train.csv, test.csv)
       │
       ├──► 2. Data Transformation ──► Preprocesses arrays & exports (preprocessor.pkl)
       │
       └──► 3. Model Training ──► Cross-validates parameters, logs to MLflow & saves (model.pkl)



Detailed Component Operations: 

Data Ingestion (data_ingestion.py): Loads the raw underlying dataset from its source environment directory, verifies directories, saves a copy as raw.csv inside artifacts, performs a deterministic 80/20 train-test split based on strict seeds (random_state=42), and provisions partitioned segments.

Data Transformation (data_transformation.py): Builds isolated data preprocessing streams:

Numerical Fields (age, trestbps, chol, thalch, oldpeak, ca): Median missing value imputation followed by standard scaling.

Categorical Fields (sex, dataset, cp, fbs, restecg, exang, slope, thal): Most frequent value imputation, one-hot encoding with unknown handling ignored, and standard deviation scaling.

Extracted transformers are packed into an exportable pipeline tracking block saved as preprocessor.pkl.

Model Training & Logging (model_trainer.py): Converts medical diagnosis multi-class markers into a unified risk tracking binary target ($num > 0 \rightarrow 1$). It triggers iterative GridSearchCV routines via 5-Fold stratified cross-validations for Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, and XGBoost.

📊 Evaluation Summary Output Matrix 

During pipeline training, the system logs metrics summaries straight into standard out streams and archives them into tracking log frameworks:


=================================================================
Model                     Accuracy    ROC AUC  Precision     Recall
=================================================================
Logistic Regression         0.8424     0.9040     0.8774     0.8532
Decision Tree               0.8152     0.8179     0.8713     0.8073
Random Forest               0.8587     0.9166     0.8952     0.8624
Gradient Boosting           0.8261     0.9024     0.8738     0.8257
XGBoost                     0.8478     0.9117     0.8785     0.8624
=================================================================

🏆 Best Model Selected: Random Forest Classifier
   ROC AUC Metrics    : 0.9166
   Optimal Parameters : {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}


Installation & Execution Guide
Clone the Space Workspace
git clone [https://github.com/abdullahhashmi01/Heart_Disease_ML-]
cd Heart_Disease_ML-

 Build Internal Project Package Requirements

pip install -r requirements.txt

📈 Remote MLflow & DagsHub Logging Configuration
The model training lifecycle uses mlflow.sklearn.autolog() and mlflow.xgboost.autolog() to track metrics remotely. To protect your access tokens from being exposed in public code blocks, set your DagsHub workspace secrets in your terminal session before launching code runs:
# Windows Command Prompt / Terminal Environment Setup
$env:MLFLOW_TRACKING_USERNAME="abdullahhashmi01"
$env:MLFLOW_TRACKING_PASSWORD="your_dagshub_token_here"

# Linux / Bash Shell / macOS Terminal Configurations
export MLFLOW_TRACKING_USERNAME="abdullahhashmi01"
export MLFLOW_TRACKING_PASSWORD="your_dagshub_token_here"

Triggering the Central Pipeline Orchestration
To execute data collection routines, transformation preprocessing pipelines, model cross-validations, log performance graphs onto DagsHub, and write all required artifacts seamlessly, run:

python app.py

Launch the Streamlit User Interface
Once the pipeline has completed execution and saved the artifact models successfully, initialize the web application backend using:
streamlit run streamlit_app.py

👤 Project Author
Lead Engineer: Abdullah Hashmi

Email Contact: theabdullahhashmi90@gmail.com


