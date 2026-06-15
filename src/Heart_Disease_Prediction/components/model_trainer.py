import os
import sys
from dataclasses import dataclass 

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

            # ✅ FIX 1: y ko binary integer banao (num > 0 = disease)
            y_train = (y_train > 0).astype(int)
            y_test  = (y_test  > 0).astype(int)

            models = {
                # ✅ FIX 2: LogisticRegression — multi_class hata do, max_iter add karo
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree"      : DecisionTreeClassifier(random_state=42),
                "Random Forest"      : RandomForestClassifier(random_state=42),
                "Gradient Boosting"  : GradientBoostingClassifier(random_state=42),
                # ✅ FIX 3: XGBoost — eval_metric add karo
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

                for model_name, model in models.items():
                    logging.info(f'Tuning {model_name} with GridSearchCV...')

                    # ✅ FIX 4: GridSearchCV — param_grid actually use ho raha hai
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

                    # ✅ FIX 5: roc_auc ke liye predict_proba use karo
                    y_pred = best_model.predict(X_test)
                    y_prob = best_model.predict_proba(X_test)[:, 1]

                    accuracy  = accuracy_score(y_test, y_pred)
                    roc_auc   = roc_auc_score(y_test, y_prob)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall    = recall_score(y_test, y_pred, zero_division=0)

                    model_report[model_name] = {
                        'accuracy'   : round(accuracy,  4),
                        'roc_auc'    : round(roc_auc,   4),
                        'precision'  : round(precision,  4),
                        'recall'     : round(recall,     4),
                        'best_params': best_params
                    }

                    logging.info(
                        f'{model_name} → Acc: {accuracy:.4f} | AUC: {roc_auc:.4f} | '
                        f'Precision: {precision:.4f} | Recall: {recall:.4f}'
                    )

                # ✅ FIX 6: Best tuned model save karo
                best_model_name = max(model_report, key=lambda x: model_report[x]['roc_auc'])
                best_model      = tuned_models[best_model_name]

                save_object(
                    file_path=ModelTrainerConfig.trained_model_file_path,
                    obj=best_model
                )

                logging.info(
                    f'Best model: {best_model_name} | '
                    f'ROC AUC: {model_report[best_model_name]["roc_auc"]} | '
                    f'Params: {model_report[best_model_name]["best_params"]}'
                )

                return model_report

            # Call evaluate_models
            model_report = evaluate_models(models, param_grid, X_train, y_train, X_test, y_test)

            logging.info(f'Model evaluation results: {model_report}')

            # ✅ Print summary table
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
