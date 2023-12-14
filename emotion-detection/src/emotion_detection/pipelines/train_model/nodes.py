from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    GlobalMaxPooling1D,
    Input,
    Layer,
    MaxPooling1D,
)
from keras.metrics import AUC
from keras.models import Model
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)


def encoded_to_label(encoded_array: np.ndarray, label_dict: dict) -> list:
    return [list(label_dict.keys())[i] for i in encoded_array]


def create_dense_model(input_shape: int, n_classes: int, layers: list[int]) -> Model:
    input_main = Input(shape=input_shape)
    layer = input_main
    for n_neurons in layers:
        layer = Dense(n_neurons, activation="relu", kernel_regularizer="l2")(layer)

    predictions = Dense(n_classes, activation="softmax")(layer)
    model = Model(inputs=[input_main], outputs=predictions)

    return model


def identity_block(X: Layer, f: Any, filters: tuple) -> Layer:
    F1, F2 = filters
    X_shortcut = X
    X = Conv2D(
        filters=F1,
        kernel_size=f,
        strides=1,
        padding="same",
        kernel_initializer=glorot_uniform(seed=0),
    )(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation("relu")(X)
    X = Conv2D(
        filters=F2,
        kernel_size=f,
        strides=1,
        padding="same",
        kernel_initializer=glorot_uniform(seed=0),
    )(X)
    X = BatchNormalization(axis=-1)(X)
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def convolutional_block(X: Layer, f: Any, filters: tuple, s: int = 2) -> Layer:
    F1, F2 = filters
    X_shortcut = X
    X = Conv2D(
        F1, f, strides=s, padding="same", kernel_initializer=glorot_uniform(seed=0)
    )(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation("relu")(X)
    X = Conv2D(
        F2, f, strides=1, padding="same", kernel_initializer=glorot_uniform(seed=0)
    )(X)
    X = BatchNormalization(axis=-1)(X)
    X_shortcut = Conv2D(
        F2, 1, strides=s, padding="valid", kernel_initializer=glorot_uniform(seed=0)
    )(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1)(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def create_resnet_model(input_shape: int, n_classes: int, layers: list) -> Model:
    X_input = Input((input_shape, 1))
    X = Conv1D(64, 7, strides=2, kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation("relu")(X)
    X = MaxPooling1D(3, strides=2)(X)
    for layer in layers:
        X = convolutional_block(X, f=3, filters=layer, s=1)
        X = identity_block(X, 3, layer)
    X = GlobalMaxPooling1D()(X)
    X = Dense(
        n_classes, activation="softmax", kernel_initializer=glorot_uniform(seed=0)
    )(X)
    model = Model(inputs=X_input, outputs=X)
    return model


def compile_model(model: Model) -> Model:
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=[AUC()],
    )
    print(model.summary())
    return model


def create_model(model_params: dict) -> Model:
    model_type = model_params["model_type"]
    input_shape = model_params["input_shape"]
    n_classes = model_params["n_classes"]
    dense_layers = model_params["dense_layers"]
    resnet_layers = model_params["resnet_layers"]

    accepted_models = ["dense", "resnet"]
    if model_type not in accepted_models:
        raise ValueError(f"The only accepted models are {accepted_models}")

    if model_type == "dense":
        if dense_layers is None:
            raise ValueError("Dense Layers not given...")
        model = create_dense_model(input_shape, n_classes, dense_layers)
    elif model_type == "resnet":
        if resnet_layers is None:
            raise ValueError("ResNet Layers not given...")
        model = create_resnet_model(input_shape, n_classes, resnet_layers)

    return compile_model(model)


def store_model(experiment: str, name: str) -> Path:
    model_folder = Path(f"models/{experiment}/")
    model_folder.mkdir(parents=True, exist_ok=True)
    model_path = Path("model.keras")
    model_path.rename(model_folder / f"{name}.keras")
    return model_path


def store_results(experiment: str, name: str) -> Path:
    folder = Path(f"results/{experiment}/")
    folder.mkdir(parents=True, exist_ok=True)
    results_path = folder / f"{name}.csv"
    return results_path


def train_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    model_params: dict,
    train_params: dict,
) -> str:
    model = create_model(model_params)
    batch_size = train_params["batch_size"]
    epochs = train_params["epochs"]

    print(tf.config.list_physical_devices("GPU"))

    with tf.device("/gpu:0"):
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
        )

    model.save("model.keras")
    return "dummy"


def evaluate_model(X: pd.DataFrame, y: pd.DataFrame, params: dict, dummy: str) -> None:
    trained_model = tf.keras.models.load_model("model.keras")
    label_dict = {
        "neutral": 0,
        "angry": 1,
        "happy": 2,
        "sad": 3,
        "disgust": 4,
        "fear": 5,
    }

    now = datetime.now()
    experiment = mlflow.set_experiment(
        experiment_name=f"experiment_{now.strftime('%d_%m_%y')}"
    )
    run_name = f"{params['model_type']}_{now.strftime('%d_%m_%y_%H_%M_%S')}"
    model_path = store_model(experiment.name, run_name)

    mlflow.autolog()
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        mlflow.log_metric("auc", trained_model.evaluate(X, y, verbose=False)[1])
        y_pred_hot = trained_model.predict(X, verbose=2)
        y_pred = np.argmax(y_pred_hot, axis=1)
        y_pred = encoded_to_label(y_pred, label_dict)
        y = y.to_numpy()
        y_label = np.argmax(y, axis=1)
        y_label = encoded_to_label(y_label, label_dict)
        labels = list(label_dict.keys())
        cm = compute_confusion_matrix(y_label, y_pred, labels)
        metrics_dict = compute_metrics(cm, labels)
        metrics_df = pd.DataFrame(
            metrics_dict["metrics_by_label"],
            index=["label", "precision", "recall", "f1_score", "support"],
        ).T
        results_path = store_results(experiment.name, run_name)
        roc_auc = dict()
        for label, index in label_dict.items():
            roc_auc[label] = compute_roc_curve(y[:, index], y_pred_hot[:, index], label)

        metrics_df = pd.merge(
            metrics_df,
            pd.DataFrame(roc_auc.items(), columns=["label", "auc"]),
            on="label",
        )
        metrics_df = metrics_df[
            ["label", "precision", "recall", "f1_score", "auc", "support"]
        ]
        metrics_df.to_csv(results_path)
        mlflow.log_artifact(results_path, artifact_path="metrics_by_label")
        mlflow.log_metrics(
            {
                metric: metrics_dict[metric]
                for metric in ["accuracy", "macro_precision", "macro_recall"]
            }
        )
        mlflow.log_params(params)
        mlflow.tensorflow.log_model(trained_model, str(model_path))


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_disp.plot()
    mlflow.log_figure(cm_disp.figure_, "confusion_matrix.png")
    return cm


def compute_metrics(cm: np.ndarray, labels: list) -> dict:
    output = dict()
    metrics_by_label = dict()
    macro_precision = 0
    macro_recall = 0
    for index, label in enumerate(labels):
        precision, recall, f1_score, support = metrics_from_confussion_matrix(cm, index)
        metrics_by_label[label] = [label, precision, recall, f1_score, support]
        macro_precision += precision
        macro_recall += recall
    output["macro_precision"] = macro_precision / len(labels)
    output["macro_recall"] = macro_recall / len(labels)
    output["accuracy"] = np.trace(cm) / np.sum(cm)
    output["metrics_by_label"] = metrics_by_label
    return output


def metrics_from_confussion_matrix(cm: np.ndarray, label: int) -> tuple:
    tp = cm[label, label]
    fp = np.sum(cm[:, label]) - tp
    fn = np.sum(cm[label, :]) - tp
    precision = tp / (fp + tp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-6)
    support = tp + fn
    return precision, recall, f1_score, support


def compute_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=label)
    roc_disp.plot()
    mlflow.log_figure(roc_disp.figure_, f"roc_curve_{label}.png")
    return roc_auc
