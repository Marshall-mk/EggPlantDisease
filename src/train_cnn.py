import numpy as np
import matplotlib.pyplot as plt
import hydra  # for configurations
from omegaconf.omegaconf import OmegaConf  # configs
import mlflow  # for tracking
import json  # for saving the classification report
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from cnn_models import (
    dense_net_model,
    vgg16_model,
    efficient_net_model,
    dense_net_model_FT,
    efficient_net_model_FT,
    vgg16_model_FT,
)
from data_load import DataLoader
import tensorflow as tf

# EXPERIMENT_NAME = "Eggplant Disease Classification exp"
# EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
# MLFLOW_TRACKING_URI = "https://dagshub.com/Marshall-mk/EggPlantDisease.mlflow"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@hydra.main(config_path="../configs", config_name="configs", version_base="1.2")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)
    # Load the data
    data_loader = DataLoader()
    train_data = data_loader.load_train_data(cfg.model.train_data_path)
    val_data = data_loader.load_val_data(cfg.model.train_data_path)
    test_data = data_loader.load_test_data(cfg.model.test_data_path)
    model_name = cfg.model.model_name

    if model_name == "vgg16":
        model = vgg16_model(
            input_shape=(224, 224, 3),
            classes=7,
        )
    elif model_name == "vgg16_ft":
        model = vgg16_model_FT(
            input_shape=(224, 224, 3),
            classes=7,
        )
    elif model_name == "efficient_net_ft":
        model = efficient_net_model_FT(
            input_shape=(224, 224, 3),
            classes=7,
        )
    elif model_name == "efficient_net":
        model = efficient_net_model(
            input_shape=(224, 224, 3),
            classes=7,
        )
    elif model_name == "dense_net_ft":
        model = dense_net_model_FT(
            input_shape=(224, 224, 3),
            classes=7,
        )
    elif model_name == "dense_net":
        model = dense_net_model(
            input_shape=(224, 224, 3),
            classes=7,
        )
    else:
        print("Please specify a valid model name")

    model.compile(
        optimizer=cfg.train.optimizer,
        loss=cfg.train.loss,
        metrics=["accuracy", "precision", "recall", "f1_score"],
    )
    """Model callbacks"""
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", verbose=1, patience=5
    )
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{cfg.model.ckpt_path}{model_name}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
    )

    # Train the model
    # with mlflow.start_run(experiment_id=EXPERIMENT_ID):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=[earlystopping, checkpointer],
    )

    # Evaluate the model
    model.save(f"{cfg.model.save_path}{model_name}_model.keras")
    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1_score,
    ) = model.evaluate(test_data)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_loss}")
    print(f"Test precision: {test_precision}")
    print(f"Test recall: {test_recall}")
    print(f"Test f1_score: {test_f1_score}")
        # mlflow.log_metric("test_accuracy", test_accuracy)
        # mlflow.log_metric("test_loss", test_loss)
        # mlflow.log_metric("test_precision", test_precision)
        # mlflow.log_metric("test_recall", test_recall)
        # mlflow.log_metric("test_f1_score", test_f1_score)
        # mlflow.log_params(cfg)
        # # mlflow.log_artifact(f"{cfg.model.ckpt_path}{model_name}_model.h5")
        # mlflow.end_run()

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
    plt.savefig(f"{cfg.model.history_path}{model_name}_confusion_matrix.png")
    # mlflow.log_artifact("confusion_matrix.png")

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
    json_report = json.dumps(report, indent=4)
    with open(
        f"{cfg.model.history_path}{model_name}_classification_report.json", "w"
    ) as f:
        f.write(json_report)
    # mlflow.log_params(report)
    _model_history(history, cfg)


def _model_history(model_info, cfg):
    accuracy = model_info.history["accuracy"]
    val_accuracy = model_info.history["val_accuracy"]
    loss = model_info.history["loss"]
    val_loss = model_info.history["val_loss"]
    precision = model_info.history["precision"]
    val_precision = model_info.history["val_precision"]
    recall = model_info.history["recall"]
    val_recall = model_info.history["val_recall"]
    f1_score = model_info.history["f1_score"]
    val_f1_score = model_info.history["val_f1_score"]
    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(20, 10))
    plt.plot(epochs, accuracy, "g-", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.grid()
    plt.savefig(
        f"{cfg.model.history_path}{cfg.model.model_name}_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.legend()

    plt.figure(figsize=(20, 10))
    plt.plot(epochs, loss, "g-", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{cfg.model.history_path}{cfg.model.model_name}_loss.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.figure(figsize=(20, 10))
    plt.plot(epochs, precision, "g-", label="Training precision")
    plt.plot(epochs, val_precision, "b", label="Validation precision")
    plt.title("Training and validation precision")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{cfg.model.history_path}{cfg.model.model_name}_precision.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure(figsize=(20, 10))
    plt.plot(epochs, recall, "g-", label="Training recall")
    plt.plot(epochs, val_recall, "b", label="Validation recall")
    plt.title("Training and validation recall")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{cfg.model.history_path}{cfg.model.model_name}_recall.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure(figsize=(20, 10))
    plt.plot(epochs, f1_score, "g-", label="Training f1_score")
    plt.plot(epochs, val_f1_score, "b", label="Validation f1_score")
    plt.title("Training and validation f1_score")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{cfg.model.history_path}{cfg.model.model_name}_f1_score.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
