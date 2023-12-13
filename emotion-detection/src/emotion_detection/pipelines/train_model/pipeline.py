"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model  # , save_keras_model, load_keras_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "X_val",
                    "y_train",
                    "y_val",
                    "params:model_params",
                    "params:train_params",
                ],
                outputs="dummy",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["X_test", "y_test", "params:model_params", "dummy"],
                outputs=None,
                name="evaluate_model",
            ),
        ]
    )
