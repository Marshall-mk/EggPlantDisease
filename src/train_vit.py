import numpy as np
import matplotlib.pyplot as plt
import hydra  # for configurations
from omegaconf.omegaconf import OmegaConf  # configs

from keras import layers, ops
import keras
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_load import DataLoader
from gcvit_model import GCViT
from swin_model import PatchEmbedding, SwinTransformer, PatchMerging, patch_extract


@hydra.main(config_path="../configs", config_name="configs", version_base="1.2")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)
    # Load the data
    data_loader = DataLoader()
    train_data = data_loader.load_train_data(cfg.model.train_data_path)
    val_data = data_loader.load_val_data(cfg.model.train_data_path)
    test_data = data_loader.load_test_data(cfg.model.test_data_path)
    model_name = cfg.model.model_name

    if model_name == "swinvit":
        input_shape = (224, 224, 3)
        num_classes = 7
        patch_size = (2, 2)  # 2-by-2 sized patches
        dropout_rate = 0.03  # Dropout rate
        num_heads = 8  # Attention heads
        embed_dim = 12  # Embedding dimension
        num_mlp = 256  # MLP layer size
        qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
        window_size = 2  # Size of attention window
        shift_size = 1  # Size of shifting window
        image_dimension = input_shape[0]  # Initial image size
         

        num_patch_x = input_shape[0] // patch_size[0]
        num_patch_y = input_shape[1] // patch_size[1]

        learning_rate = 1e-3
        weight_decay = 0.0001
        label_smoothing = 0.1

        class PatchExtract(layers.Layer):
            def call(self, images):
                return patch_extract(images)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], num_patch_x * num_patch_y, embed_dim)

            def get_config(self):
                return {}

        inputs = keras.Input(shape=input_shape)
        input = PatchExtract()(inputs)
        x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(input)
        x = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )(x)
        x = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )(x)
        x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
        x = layers.GlobalAveragePooling1D()(x)
        output = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, output)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            optimizer=keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            ),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
                "precision",
                "recall",
                "f1_score",
            ],
        )
        history = model.fit(
            train_data,
            batch_size=cfg.train.batch_size,
            epochs=cfg.train.epochs,
            validation_data=val_data,
        )
        model.save(f"{cfg.model.save_path}{model_name}_model.keras")
        model.save_weights(f"{cfg.model.save_path}{model_name}.weights.h5")
        loss, accuracy, top_5_accuracy, precision, recall, f1_score = model.evaluate(
            test_data
        )
        _model_history(history, cfg)

        print(f"Test loss: {round(loss, 2)}")
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
        # print(f"Test precision: {round(precision * 100, 2)}%")
        # print(f"Test recall: {round(recall * 100, 2)}%")
        # print(f"Test f1_score: {round(f1_score * 100, 2)}%")

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

        # Log the classification report
        class_names = [k for k, v in test_data.class_indices.items()]
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        report = {k: v for k, v in report.items() if k in class_names}
        report = {
            k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report.items()
        }
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
   

    elif model_name == "gcvit":
        # Model Params
        config = {
            "window_size": (7, 7, 14, 7),
            "embed_dim": 64,
            "depths": (2, 2, 6, 2),
            "num_heads": (2, 4, 8, 16),
            "mlp_ratio": 3.0,
            "path_drop": 0.2,
        }

        # Re-Build Model
        model = GCViT(**config, num_classes=7)
        model.compile(
            loss=cfg.train.loss,
            optimizer=cfg.train.optimizer,
            metrics=["accuracy", "precision", "recall", "f1_score"],
        )
        history = model.fit(
            train_data,
            batch_size=cfg.train.batch_size,
            validation_data=val_data,
            epochs=cfg.train.epochs,
            verbose=1,
        )
        model.save(f"{cfg.model.save_path}{model_name}_model.keras")
        model.save_weights(f"{cfg.model.save_path}{model_name}.weights.h5")
        loss, accuracy, precision, recall, f1_score = model.evaluate(test_data)
        _model_history(history, cfg)
        print(f"Test loss: {round(loss, 2)}")
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test precision: {round(precision * 100, 2)}%")
        print(f"Test recall: {round(recall * 100, 2)}%")
        print(f"Test f1_score: {round(f1_score * 100, 2)}%")

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

        # Log the classification report
        class_names = [k for k, v in test_data.class_indices.items()]
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        report = {k: v for k, v in report.items() if k in class_names}
        report = {
            k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report.items()
        }
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

