import matplotlib.pyplot as plt
from typing import Dict, List


def plot_loss(history: Dict[str, List[float]], title: str = "Training and Validation Loss") -> None:
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
