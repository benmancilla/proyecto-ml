from __future__ import annotations

from kedro.pipeline import Pipeline
from proyecto_ml.pipelines import reporting
from proyecto_ml.pipelines import data_science 
from proyecto_ml.pipelines import data_engineering
from proyecto_ml.pipelines import classification
from proyecto_ml.pipelines import regression

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": (
            data_engineering.create_pipeline() + 
            data_science.create_pipeline() + 
            classification.create_pipeline() + 
            regression.create_pipeline() +
            reporting.create_pipeline()
        ),
        "data_engineering": data_engineering.create_pipeline(),
        "data_science": data_science.create_pipeline(),
        "classification": classification.create_pipeline(),
        "regression": regression.create_pipeline(),
        "reporting": reporting.create_pipeline(),
    }
