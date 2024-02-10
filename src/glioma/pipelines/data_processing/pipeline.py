"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline


from .nodes import (
    split_train_test_valid,
    processing,
)

def create_pipeline(**kwargs) -> Pipeline:

    pipeline_instance = pipeline(
        [
            node(
                func=processing,
                inputs=["data"],
                outputs=["data_clean", "data_filled_NaNs"],
            ),
            node(
                func=split_train_test_valid,
                inputs=["data_clean"],
                outputs=["train_dataset", "test_dataset"],
            ),
            node(
                func=split_train_test_valid,
                inputs=["data_filled_NaNs"],
                outputs=["train_dataset_filled", "test_dataset_filled"],
            ),  
        ]
    )

    return pipeline_instance