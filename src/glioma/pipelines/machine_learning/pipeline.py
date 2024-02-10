"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    def_features_n_outcome,
    feature_selection,
    train_model_and_save_trials_xgb,
    train_model_and_save_trials_knn,
    train_model_and_save_trials_svm,
    train_model_and_save_trials_rf,
    train_model_and_save_trials_lr,
    train_model_and_save_trials_dt,
    train_model_and_save_trials_gnb,
    train_model_and_save_trials_lda,
    train_model_and_save_trials_ada,
    train_model_and_save_trials_lgb,
    evaluate_model,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=def_features_n_outcome,
                inputs=["train_dataset"],
                outputs=["X_train", "y_train"],
            ),
            node(
                func=def_features_n_outcome,
                inputs=["test_dataset"],
                outputs=["X_test", "y_test"],
            ), 
            node(
                func=def_features_n_outcome,
                inputs=["train_dataset_filled"],
                outputs=["X_train_filled", "y_train_filled"],
            ),
            node(
                func=def_features_n_outcome,
                inputs=["test_dataset_filled"],
                outputs=["X_test_filled", "y_test_filled"],
            ), 
            node(
                func=feature_selection,
                inputs=["X_train", "y_train", "X_test"],
                outputs=["X_train_selected", "X_test_selected"],
            ),     
            node(
                func=feature_selection,
                inputs=["X_train_filled", "y_train_filled", "X_test_filled"],
                outputs=["X_train_filled_selected", "X_test_filled_selected"],
            ),   
            node(
                func=train_model_and_save_trials_xgb,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_xgb_model", 'classification_report_xgb_training'],
                name="train_xgb_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_xgb_model", "X_test_filled_selected", "y_test_filled"],
                outputs="xgb_classification_report",
                name="evaluate_model_xgb",
            ),
            node(
                func=train_model_and_save_trials_knn,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_knn_model", 'classification_report_knn_training'],
                name="train_knn_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_knn_model", "X_test_filled_selected", "y_test_filled"],
                outputs="knn_classification_report",
                name="evaluate_model_knn",
            ),
            node(
                func=train_model_and_save_trials_svm,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_svm_model", 'classification_report_svm_training'],
                name="train_svm_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_svm_model", "X_test_filled_selected", "y_test_filled"],
                outputs="svm_classification_report",
                name="evaluate_model_svm",
            ),
            node(
                func=train_model_and_save_trials_rf,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_rf_model", 'classification_report_rf_training'],
                name="train_rf_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_rf_model", "X_test_filled_selected", "y_test_filled"],
                outputs="rf_classification_report",
                name="evaluate_model_rf",
            ),

            node(
                func=train_model_and_save_trials_lr,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_lr_model", 'classification_report_lr_training'],
                name="train_lr_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_lr_model", "X_test_filled_selected", "y_test_filled"],
                outputs="lr_classification_report",
                name="evaluate_model_lr",
            ),

            node(
                func=train_model_and_save_trials_dt,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_dt_model", 'classification_report_dt_training'],
                name="train_dt_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_dt_model", "X_test_filled_selected", "y_test_filled"],
                outputs="dt_classification_report",
                name="evaluate_model_dt",
            ),

            node(
                func=train_model_and_save_trials_lgb,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_lgb_model", 'classification_report_lgb_training'],
                name="train_lgb_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_lgb_model", "X_test_filled_selected", "y_test_filled"],
                outputs="lgb_classification_report",
                name="evaluate_model_lgb",
            ),
            node(
                func=train_model_and_save_trials_ada,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_ada_model", 'classification_report_ada_training'],
                name="train_ada_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_ada_model", "X_test_filled_selected", "y_test_filled"],
                outputs="ada_classification_report",
                name="evaluate_model_ada",
            ),
            node(
                func=train_model_and_save_trials_lda,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_lda_model", 'classification_report_lda_training'],
                name="train_lda_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_lda_model", "X_test_filled_selected", "y_test_filled"],
                outputs="lda_classification_report",
                name="evaluate_model_lda",
            ),
            node(
                func=train_model_and_save_trials_gnb,
                inputs=["X_train_filled_selected", "y_train_filled"],
                outputs=["best_gnb_model", 'classification_report_gnb_training'],
                name="train_gnb_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_gnb_model", "X_test_filled_selected", "y_test_filled"],
                outputs="gnb_classification_report",
                name="evaluate_model_gnb",
            ),
    ])
