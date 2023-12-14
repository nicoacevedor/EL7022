from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_data, prepare_data, process_data, select_features, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs="params:data_raw_path",
                outputs="raw_data",
                name="load_data",
            ),
            node(
                func=select_features,
                inputs="raw_data",
                outputs=["video_names", "target", "other_data"],
                name="select_features",
            ),
            node(
                func=prepare_data,
                inputs=["video_names", "params:recreate_data"],
                outputs="mel_images",
                name="prepare_data",
            ),
            node(
                func=split_data,
                inputs=[
                    "mel_images",
                    "other_data",
                    "target",
                    "params:split_data_params",
                ],
                outputs=[
                    "X_train_pre",
                    "X_val_pre",
                    "X_test_pre",
                    "y_train_pre",
                    "y_val_pre",
                    "y_test_pre",
                ],
                name="split_data",
            ),
            node(
                func=process_data,
                inputs=[
                    "X_train_pre",
                    "X_val_pre",
                    "X_test_pre",
                    "y_train_pre",
                    "y_val_pre",
                    "y_test_pre",
                ],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="process_data",
            ),
        ]
    )
