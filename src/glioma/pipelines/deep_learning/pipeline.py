"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    optimize_n_train_1dcnn
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=optimize_n_train_1dcnn,
                inputs=["X_train_processed", "y_train_dl", "Age_train", "parameters"],
                outputs=None,
            )
    ])
