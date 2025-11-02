"""
Regression pipeline for machine learning project.
Trains multiple regression models with GridSearchCV and k-fold cross-validation.
"""
from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
