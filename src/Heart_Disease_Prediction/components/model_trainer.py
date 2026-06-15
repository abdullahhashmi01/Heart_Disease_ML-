import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

# MLflow + DagsHub
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import dagshub

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from src.Heart_Disease_Prediction.exception import CustomException
from src.Heart_Disease_Prediction.logger import logging
from src.Heart_Disease_Prediction.utils import save_object, evaluate_models

# ✅ DagsHub + MLflow Setup
os.environ['MLFLOW_TRACKING_USERNAME'] = 'abdullahhashmi01'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '68e154b026de9de6af153f16866e3e24b4c3ffbb'  # ← Sirf Ye Badlo

dagshub.init(
    repo_owner='abdullahhashmi01',
    repo_name='Heart_Disease_ML-',
    mlflow=True
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and testing input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # ✅ y ko binary integer banao (num > 0 = disease)
            y_train = (y_train > 0).astype(int)
            y_test  = (y_test  > 0).astype(int)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree"      : DecisionTreeClassifier(random_state=42),
                "Random Forest"      : RandomForestClassifier(random_state=42),
                "Gradient Boosting"  : GradientBoostingClassifier(random_state=42),
                "XGBoost"            : XGBClassifier(eval_metric='logloss', random_state=42)
            }

            param_grid = {
                "Logistic Regression": {
                    'C'     : [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                },
                "Decision Tree": {
                    'max_depth'        : [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators'     : [100, 200],
                    'max_depth'        : [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "Gradient Boosting": {
                    'n_estimators' : [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth'    : [3, 5]
                },
                "XGBoost": {
                    'n_estimators' : [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth'    : [3, 5]
                }
            }

            def evaluate_models(models, param_grid, X_train, y_train, X_test, y_test):
                """Evaluate all models with GridSearchCV hyperparameter tuning"""
                model_report = {}
                tuned_models = {}
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # ✅ MLflow Experiment Set
                mlflow.set_experiment("Heart_Disease_Model_Training")

                for model_name, model in models.items():
                    logging.info(f'Tuning {model_name} with GridSearchCV...')

                    with mlflow.start_run(run_name=model_name):

                        # ✅ Framework specific autolog
                        if model_name == "XGBoost":
                            mlflow.xgboost.autolog(log_models=True)
                        else:
                            mlflow.sklearn.autolog(log_models=True)

                        # ✅ GridSearchCV
                        gs = GridSearchCV(
                            estimator  = model,
                            param_grid = param_grid[model_name],
                            cv         = cv,
                            scoring    = 'roc_auc',
                            n_jobs     = -1,
                            verbose    = 0
                        )
                        gs.fit(X_train, y_train)

                        best_model  = gs.best_estimator_
                        best_params = gs.best_params_
                        tuned_models[model_name] = best_model

                        logging.info(f'{model_name} best params: {best_params}')

                        # ✅ Metrics calculate karo
                        y_pred = best_model.predict(X_test)
                        y_prob = best_model.predict_proba(X_test)[:, 1]

                        accuracy  = accuracy_score(y_test, y_pred)
                        roc_auc   = roc_auc_score(y_test, y_prob)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall    = recall_score(y_test, y_pred, zero_division=0)

                        # ✅ MLflow pe log karo
                        mlflow.log_metric("test_accuracy",  accuracy)
                        mlflow.log_metric("test_roc_auc",   roc_auc)
                        mlflow.log_metric("test_precision", precision)
                        mlflow.log_metric("test_recall",    recall)

                        model_report[model_name] = {
                            'accuracy'   : round(accuracy,   4),
                            'roc_auc'    : round(roc_auc,    4),
                            'precision'  : round(precision,  4),
                            'recall'     : round(recall,     4),
                            'best_params': best_params
                        }

                        logging.info(
                            f'{model_name} → Acc: {accuracy:.4f} | AUC: {roc_auc:.4f} | '
                            f'Precision: {precision:.4f} | Recall: {recall:.4f}'
                        )

                # ✅ Best model select karo
                best_model_name = max(model_report, key=lambda x: model_report[x]['roc_auc'])
                best_model      = tuned_models[best_model_name]

                # ✅ Final Best Model MLflow pe log karo
                with mlflow.start_run(run_name="Final_Best_Model"):
                    mlflow.log_param("best_model_name",    best_model_name)
                    mlflow.log_metric("best_test_roc_auc", model_report[best_model_name]['roc_auc'])

                    if best_model_name == "XGBoost":
                        mlflow.xgboost.log_model(best_model, "best_production_model")
                    else:
                        mlflow.sklearn.log_model(best_model, "best_production_model")

                # ✅ Model save karo
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                logging.info(
                    f'Best model: {best_model_name} | '
                    f'ROC AUC: {model_report[best_model_name]["roc_auc"]} | '
                    f'Params: {model_report[best_model_name]["best_params"]}'
                )

                return model_report

            # ✅ evaluate_models call karo
            model_report = evaluate_models(models, param_grid, X_train, y_train, X_test, y_test)

            logging.info(f'Model evaluation results: {model_report}')

            # ✅ Summary Table Print
            print(f"\n{'='*65}")
            print(f"{'Model':<22} {'Accuracy':>10} {'ROC AUC':>10} {'Precision':>10} {'Recall':>10}")
            print(f"{'='*65}")
            for name, metrics in model_report.items():
                print(f"{name:<22} {metrics['accuracy']:>10} {metrics['roc_auc']:>10} "
                      f"{metrics['precision']:>10} {metrics['recall']:>10}")
            print(f"{'='*65}")

            best = max(model_report, key=lambda x: model_report[x]['roc_auc'])
            print(f"\n🏆 Best Model : {best}")
            print(f"   ROC AUC   : {model_report[best]['roc_auc']}")
            print(f"   Params    : {model_report[best]['best_params']}")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)