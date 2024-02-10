import os
import logging
from typing import Dict
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, log_loss, classification_report, make_scorer, matthews_corrcoef
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def def_features_n_outcome(data: pd.DataFrame):
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    X = data.drop(columns=['group'])
    y = data['group']

    return X, y

def feature_selection(X_train, y_train, X_test):
    sffs = SequentialFeatureSelector(LogisticRegression(),
                                      n_features_to_select="auto",
                                      direction="forward",
                                      scoring='accuracy',
                                      cv=5)
    
    # Fit the SFFS object to your training data
    sffs.fit(X_train, y_train)
    
    # Get the indices of the selected features
    selected_feature_indices = sffs.get_support(indices=True)
    
    feature_names = X_train.columns
    # Use the indices to retrieve the corresponding feature names
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]
    
    # Transform the training set based on the selected features
    X_train_sffs = sffs.transform(X_train)
    X_test_sffs = sffs.transform(X_test)
    
    # Create DataFrames with selected features and set column names
    X_train_sffs_df = pd.DataFrame(X_train_sffs, columns=selected_feature_names)
    X_test_sffs_df = pd.DataFrame(X_test_sffs, columns=selected_feature_names)

    return X_train_sffs_df, X_test_sffs_df


# def feature_selection(X_train, y_train, X_test):
#     """
#     Perform feature selection using Recursive Feature Elimination (RFE).

#     Parameters:
#     - X_train: The training feature matrix.
#     - y_train: The training target variable.
#     - X_test: The test feature matrix.
#     - feature_names: The names of the features.

#     Returns:
#     - X_train_rfe: The training set transformed with selected features.
#     - X_test_rfe: The test set transformed with selected features.
#     """
#     n_features_to_select = 4
#     rfe = RFE(estimator=LogisticRegression(), n_features_to_select=n_features_to_select)
    
#     # Fit the RFE object to your training data
#     X_train_rfe = rfe.fit_transform(X_train, y_train)
#     X_test_rfe = rfe.transform(X_test)

#     feature_names = X_train.columns
#     # Use the selected feature indices to retrieve corresponding feature names
#     selected_feature_indices = rfe.support_
#     selected_feature_names = [feature_names[i] for i, selected in enumerate(selected_feature_indices) if selected]

#     # Create DataFrames with selected features and set column names
#     X_train_rfe_df = pd.DataFrame(X_train_rfe, columns=selected_feature_names)
#     X_test_rfe_df = pd.DataFrame(X_test_rfe, columns=selected_feature_names)

#     return X_train_rfe_df, X_test_rfe_df


# def feature_selection(X_train, y_train, X_test):
#     # Standardize the features (important for Lasso regularization)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Create a Lasso model
#     lasso = Lasso(alpha=0.1)  # Adjust alpha as needed; higher alpha values result in sparser solutions

#     # Fit the Lasso model
#     lasso.fit(X_train_scaled, y_train)

#     # Get the coefficients and their magnitudes
#     coefficients = lasso.coef_
#     magnitude = np.abs(coefficients)

#     # Identify non-zero coefficients (selected features)
#     selected_feature_indices = np.where(magnitude > 0)[0]

#     feature_names = X_train.columns
#     # Use the indices to retrieve the corresponding feature names
#     selected_feature_names = [feature_names[i] for i in selected_feature_indices]

#     # Transform your data using the selected features
#     X_train_lasso = X_train_scaled[:, selected_feature_indices]
#     X_test_lasso = X_test_scaled[:, selected_feature_indices]

#     # Create DataFrames with selected features and set column names
#     X_train_lasso_df = pd.DataFrame(X_train_lasso, columns=selected_feature_names)
#     X_test_lasso_df = pd.DataFrame(X_test_lasso, columns=selected_feature_names)

#     return X_train_lasso_df, X_test_lasso_df



def train_model_and_save_trials_xgb(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        params = {
                'max_depth': trial.suggest_int('max_depth', 1, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
                'subsample': trial.suggest_float('subsample', 0.01, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
                'eval_metric': 'mlogloss',
                'enable_categorical':True,
                'tree_method': 'approx', # 'gpu_hist',
                'missing' : np.nan,
                'eval_metric' : 'logloss',
            }
        model = xgb.XGBClassifier(**params)

        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') 
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    for i, trial in enumerate(study.trials):
        params = trial.params
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))

    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training


def train_model_and_save_trials_knn(X_train: pd.DataFrame, y_train: pd.Series):
    
    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)   
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))

    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training

def train_model_and_save_trials_svm(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        svc_c = trial.suggest_float("C", 1e-2, 1e2, log=True)
        svc_gamma = trial.suggest_float("gamma", 1e-2, 1e-2, log=True)
        svc_kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        model = svm.SVC(C=svc_c, kernel=svc_kernel, gamma=svc_gamma)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = svm.SVC(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = svm.SVC(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))


    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training

def train_model_and_save_trials_rf(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # X_train = X_train.fillna(0)
        n_estimators = trial.suggest_int("n_estimators", 50, 1000, log=True)
        max_depth = trial.suggest_int("max_depth", 10, 100, log=True)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))


    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training


def train_model_and_save_trials_lr(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        params = {
        # 'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
        }
        model = LogisticRegression(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))


    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training

def train_model_and_save_trials_lgb(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        params = {
        'objective': 'binary',  # For binary classification
        'metric': 'binary_logloss',  # Logarithmic loss is a common choice
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 10, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        model = lgb.LGBMClassifier(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))


    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training


def train_model_and_save_trials_gnb(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        params = {
        # 'max_depth': trial.suggest_int('max_depth', 1, 32),
        # 'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
        # 'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
        # 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
        }
        model = GaussianNB(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = GaussianNB(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = GaussianNB(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))

    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training


def train_model_and_save_trials_ada(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
                'random_state': trial.suggest_int('random_state', 1, 100),
            }
        model = AdaBoostClassifier(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = AdaBoostClassifier(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = AdaBoostClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))


    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training

def train_model_and_save_trials_lda(X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        params = {
        'solver': trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
        # 'shrinkage': trial.suggest_categorical('shrinkage', [None, 'auto', 0.1, 0.5, 0.9]),
        'priors': trial.suggest_categorical('priors', [None, [0.2, 0.8], [0.5, 0.5]]),
        'n_components': trial.suggest_int('n_components', 1, min(X_train.shape[1],2-1)),
        'store_covariance': trial.suggest_categorical('store_covariance', [True, False]),
        'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
        # 'covariance_estimator': trial.suggest_categorical('covariance_estimator', [None, 'empirical', 'shrinkage']),
        }
        model = LinearDiscriminantAnalysis(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = LinearDiscriminantAnalysis(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = LinearDiscriminantAnalysis(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))


    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training

def train_model_and_save_trials_dt(X_train: pd.DataFrame, y_train: pd.Series):

    # X_train = X_train.fillna(0)

    def objective(trial):
        """Define the objective function"""
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        params = {
        'max_depth': trial.suggest_int('max_depth', 1, 32),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
        }
        model = DecisionTreeClassifier(**params)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') #fit_params={'classifier__sample_weight': train_class_weights}, scoring=make_scorer(matthews_corrcoef)) #scoring='accuracy')#
        mean_score = scores.mean()

        return mean_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Number of finished trials: {}'.format(len(study.trials)))

    # Save all models and their trial information
    for i, trial in enumerate(study.trials):
        params = trial.params
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        # Log trial information
        print(f"Trial {i}:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print()

    # Save the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = DecisionTreeClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Train Accuracy: accuracy_score(y_train, best_model.predict(X_train))
    clf_report_training = classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_train, best_model.predict(X_train), target_names=['IDH-wt', 'IDH-mut']))


    print('Best trial:')
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    return best_model, clf_report_training


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the coefficient of determination.

    Args:
        xg_bin: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for discharge_destination
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, target_names=['IDH-wt', 'IDH-mut'], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['IDH-wt', 'IDH-mut']))
    
    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.2f on test data.", accuracy*100)

    return clf_report