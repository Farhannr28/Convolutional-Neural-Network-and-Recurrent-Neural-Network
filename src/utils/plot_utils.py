import matplotlib.pyplot as plt
from typing import Dict, List


def plot_loss(
    history: Dict[str, List[float]], title: str = "Training and Validation Loss"
) -> None:
    """
    Plot training loss and validation loss over epochs.

    Parameters:
    -----------
    history : Dict[str, List[float]]
        Dictionary containing 'loss' and 'val_loss' lists from Keras History object.
    title : str, optional
        Title of the plot (default is "Training and Validation Loss").
    """
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    if not loss or not val_loss:
        print("Warning: History does not contain 'loss' or 'val_loss' keys.")
        return

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multi_loss(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = "Training and Validation Loss Comparison",
) -> None:
    """
    Plot multiple training and validation losses in two subplots.

    Parameters:
    -----------
    histories : Dict[str, Dict[str, List[float]]]
        Dictionary where keys are experiment names and values are history dicts
        containing 'loss' and 'val_loss' lists.
    title : str, optional
        Title of the whole figure (default is "Training and Validation Loss Comparison").
    """
    plt.figure(figsize=(14, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        loss = history.get("loss", [])
        if loss:
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, marker="o", label=name)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot validation loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        val_loss = history.get("val_loss", [])
        if val_loss:
            epochs = range(1, len(val_loss) + 1)
            plt.plot(epochs, val_loss, marker="o", label=name)
    plt.title("Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
