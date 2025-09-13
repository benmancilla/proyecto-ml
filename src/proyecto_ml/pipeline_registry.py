from __future__ import annotations

from kedro.pipeline import Pipeline
from proyecto_ml.pipelines import data_preparation

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": data_understanding.create_pipeline() + data_preparation.create_pipeline(),
        "data_understanding": data_understanding.create_pipeline(),
        "data_preparation": data_preparation.create_pipeline(),
    }
from proyecto_ml.pipelines import data_preparation, data_understanding