import numpy as np
import matplotlib.pyplot as plt
import hydra  # for configurations
from omegaconf.omegaconf import OmegaConf  # configs
import mlflow  # for tracking
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from src.cnn_models import (
    dense_net_model,
    vgg16_model,
    efficient_net_model,
    compile_model,
)
from src.data import DataLoader
import tensorflow as tf

EXPERIMENT_NAME = "Eggplant Disease Classification"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
MLFLOW_TRACKING_URI = "https://dagshub.com/Marshall-mk/EggPlantDisease.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.tensorflow.autolog()


@hydra.main(config_path="../configs", config_name="configs")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)
    # Load the data
    data_loader = DataLoader()
    train_data = data_loader.load_train_data(cfg.model.traing_data_path)
    val_data = data_loader.load_val_data(cfg.model.traing_data_path)
    test_data = data_loader.load_test_data(cfg.model.test_data_path)

    # Load the model
    model = dense_net_model(
        input_shape=(cfg.model.image_size, cfg.model.image_size, 3),
        classes=cfg.model.classes,
    )
    model = compile_model(
        model,
        optimizer=cfg.train.optimizer,
        loss=cfg.train.loss,
        metrics=cfg.train.metrics,
        weighted_metrics=None,
    )
    """Model callbacks"""
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=5
    )
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{cfg.model.ckpt_path}model.h5",
        save_weights_only=True,
        save_best_only=True,
    )

    # Train the model
    with mlflow.start_run(experiment_id=EXPERIMENT_ID):
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=cfg.train.epoch,
            batch_size=cfg.train.batch_size,
            call_backs=[earlystopping, checkpointer],
        )

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(test_data)
        print(f"Test accuracy: {test_accuracy}")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_params(cfg)
        mlflow.log_artifact(f"{cfg.model.ckpt_path}model.h5")
        mlflow.end_run()

    # Log the confusion matrix
    y_pred = model.predict(test_data)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = test_data.classes
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=test_data.class_indices
    )
    disp.plot(cmap="viridis", values_format="d")
    plt.xticks(rotation=90)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log the classification report
    class_names = [k for k, v in test_data.class_indices.items()]
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    report = {k: v for k, v in report.items() if k in class_names}
    report = {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report.items()}
    report = {
        k: dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
        for k, v in report.items()
    }
    report = {
        k: {k2: v2 for k2, v2 in v.items() if k2 != "support"}
        for k, v in report.items()
    }
    report = {
        k: {k2: v2 for k2, v2 in v.items() if k2 != "macro avg"}
        for k, v in report.items()
    }
    report = {
        k: {k2: v2 for k2, v2 in v.items() if k2 != "weighted avg"}
        for k, v in report.items()
    }
    mlflow.log_params(report)
    _model_history(history, cfg)


def _model_history(model_info, cfg):
    accuracy = model_info.history["accuracy"]
    val_accuracy = model_info.history["val_accuracy"]
    loss = model_info.history["loss"]
    val_loss = model_info.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(20, 10))
    plt.plot(epochs, accuracy, "g-", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.grid()
    plt.savefig(f"{cfg.model.history_path}accuracy.png", dpi=300, bbox_inches="tight")

    plt.legend()

    plt.figure(figsize=(20, 10))
    plt.plot(epochs, loss, "g-", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{cfg.model.history_path}loss.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()