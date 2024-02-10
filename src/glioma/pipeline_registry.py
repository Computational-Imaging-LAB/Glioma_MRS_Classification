"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from glioma.pipelines import data_processing as dp
from glioma.pipelines import data_processing_dl as dp_dl
from glioma.pipelines import machine_learning as ml
from glioma.pipelines import deep_learning as dl


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = dp.create_pipeline()
    machine_learning_pipeline = ml.create_pipeline()
    deep_learning_pipeline = dl.create_pipeline()
    dl_data_processing_pipeline = dp_dl.create_pipeline()

    return {"data_processing":data_processing_pipeline, 
            "data_processing_dl":dl_data_processing_pipeline, 
            "machine_learning":machine_learning_pipeline,
            "deep_learning":deep_learning_pipeline,
            "__default__": deep_learning_pipeline,
            }

