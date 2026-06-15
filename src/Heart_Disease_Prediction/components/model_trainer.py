import os
import sys
from dataclasses import dataclass 

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
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

            # FIX 1: y ko binary integer banao (num > 0 = disease)
            y_train = (y_train > 0).astype(int)
            y_test  = (y_test  > 0).astype(int)
            
            models = {
                # FIX 2: LogisticRegression — multi_class hata do, max_iter add karo
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree"      : DecisionTreeClassifier(random_state=42),
                "Random Forest"      : RandomForestClassifier(random_state=42),
                "Gradient Boosting"  : GradientBoostingClassifier(random_state=42),
                # FIX 3: XGBoost — eval_metric add karo
                "XGBoost" : XGBClassifier(eval_metric='logloss', random_state=42)
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
                """Evaluate all models and return metrics"""
                model_report = {}
                for model_name, model in models.items():
                    logging.info(f'Training {model_name}...')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # ✅ FIX 4: roc_auc ke liye predict_proba use karo
                    y_prob = model.predict_proba(X_test)[:, 1]

                    accuracy  = accuracy_score(y_test, y_pred)
                    roc_auc   = roc_auc_score(y_test, y_prob)   #  y_prob use karo
                    precision = precision_score(y_test, y_pred, zero_division=0)  #  zero_division
                    recall    = recall_score(y_test, y_pred, zero_division=0)     #  zero_division

                    model_report[model_name] = {
                        'accuracy' : round(accuracy,  4),
                        'roc_auc'  : round(roc_auc,   4),
                        'precision': round(precision,  4),
                        'recall'   : round(recall,     4)
                    }

                    logging.info(f'{model_name} → Acc: {accuracy:.4f} | AUC: {roc_auc:.4f} | '
                                 f'Precision: {precision:.4f} | Recall: {recall:.4f}')

                # FIX 5: Best model save karo
                best_model_name = max(model_report, key=lambda x: model_report[x]['roc_auc'])
                best_model      = models[best_model_name]

                save_object(
                    file_path=ModelTrainerConfig.trained_model_file_path,
                    obj=best_model
                )
                logging.info(f'Best model: {best_model_name} saved.')

                return model_report

            # Call evaluate_models
            model_report = evaluate_models(models, param_grid, X_train, y_train, X_test, y_test)

            logging.info(f'Model evaluation results: {model_report}')

            return model_report

        except Exception as e:
            raise CustomException(e, sys)