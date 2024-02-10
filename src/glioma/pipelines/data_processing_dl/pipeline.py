"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline


from .nodes import (
    split_train_test_valid, 
    def_features_n_outcome,
    normalizer,
    smoother,
    scaler,
    transformer
)

def create_pipeline(**kwargs) -> Pipeline:

    pipeline_instance = pipeline(
        [
            node(
                func=split_train_test_valid,
                inputs=["data_dl"],
                outputs=["train_dataset_dl", "test_dataset_dl", "valid_dataset_dl"],
            ),
            node(
                func=def_features_n_outcome,
                inputs=["train_dataset_dl"],
                outputs=["X_train_dl", "Age_train", "y_train_dl"],
            ),
            node(
                func=def_features_n_outcome,
                inputs=["test_dataset_dl"],
                outputs=["X_test_dl", "Age_test", "y_test_dl"],
            ),  
            node(
                func=def_features_n_outcome,
                inputs=["valid_dataset_dl"],
                outputs=["X_valid_dl", "Age_valid", "y_valid_dl"],
            ),
            node(
                func=normalizer,
                inputs=["X_train_dl"],
                outputs="X_train_normalized",
            ),
            node(
                func=smoother,
                inputs=["X_train_normalized"],
                outputs="X_train_smoothed",
            ),               
            node(
                func=transformer,
                inputs=["X_train_smoothed"],
                outputs=["X_train_transformed", "transformer"]
            ), 
            node(
                func=scaler,
                inputs=["X_train_transformed"], 
                outputs=["X_train_processed", "scaler"] 
            ), 
            node(
                func=normalizer,
                inputs=["X_test_dl"],
                outputs="X_test_normalized",
            ),
            node(
                func=smoother,
                inputs=["X_test_normalized"],
                outputs="X_test_smoothed",
            ),            
            node(
                func=transformer,
                inputs=["X_test_smoothed", "transformer"],
                outputs=["X_test_transformed", "transformer_not_saved"]
            ), 
            node(
                func=scaler,
                inputs=["X_test_transformed", "scaler"],
                outputs=["X_test_processed", "scaler_not_saved"]
            ),   
            node(
                func=normalizer,
                inputs=["X_valid_dl"],
                outputs="X_valid_normalized",
            ),
            node(
                func=smoother,
                inputs=["X_valid_normalized"],
                outputs="X_valid_smoothed",
            ),            
            node(
                func=transformer,
                inputs=["X_valid_smoothed", "transformer"],
                outputs=["X_valid_transformed", "transformer_not_saved_val"]
            ), 
            node(
                func=scaler,
                inputs=["X_valid_transformed", "scaler"],
                outputs=["X_valid_processed", "scaler_not_saved_val"]
            ),  
        ]
    )

    return pipeline_instance