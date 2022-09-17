import os
from queue import Queue
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader

from datasets import *
import metrics
from models import *


def plot_loss(figure, epochs: list, loss: List[list], labels: List[str], start_epoch: int = None) -> None:
    """
    Plot loss values over epochs on the given figure.
    Parameters:
    `figure`: A figure to plot on.
    `epochs`: A sequence of epoch numbers.
    `loss`: A list of lists of loss values, each of which are plotted as separate lines. Each nested list must have the same length as `epochs`.
    `labels`: A list of strings to display in the legend for each item in `loss`.
    `start_epoch`: The epoch number at which to display a horizontal line to indicate the start of the current training session.
    """
    figure.clear()
    axis = figure.add_subplot(1, 1, 1)  # Number of rows, number of columns, index
    
    # markers = (".:", ".-")
    # colors = (Colors.BLUE, Colors.ORANGE)

    # Plot each set of loss values.
    for i, loss_i in enumerate(loss):
        if not len(loss_i):
            continue
        # color = colors[i % len(colors)]
        axis.plot(epochs[:len(loss_i)], loss_i, ".-", label=labels[i])
        axis.annotate(f"{loss_i[-1]:,.2f}", (epochs[-1 - (len(epochs)-len(loss_i))], loss_i[-1]), fontsize=10)
    
    # # Plot a vertical line indicating when the current training session began.
    # if start_epoch:
    #     axis.vlines(start_epoch - 0.5, 0, max([max(_) for _ in loss]), colors=(Colors.GRAY,), label="Current session starts")
    
    axis.legend()
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Epochs")
    axis.set_ylabel("Loss")
    axis.grid(axis="y")

def save_model(filepath: str, **kwargs) -> None:
    """Save model parameters to a file."""
    torch.save(kwargs, filepath)
    print(f"Saved model parameters to {filepath}.")

def load_model(filepath: str, device: str) -> dict:
    """Return a dictionary of model parameters from a file."""
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except FileNotFoundError:
        print(f"{filepath} not found.")
    else:
        print(f"Loaded model from {filepath} trained for {checkpoint['epoch']} epochs.")
        return checkpoint

def train_regression(
    device: str, epoch_count: int, checkpoint: dict, filepath_model: str, save_model_every: int,
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_function: nn.Module,
    train_dataloader: DataLoader, validate_dataloader: DataLoader,
    scheduler = None,
    queue=None, queue_to_main=None, info_gui: dict=None,
) -> nn.Module:
    """Train and validate the given regression model. Return the model after finishing training."""

    # Load the previous training history.
    if checkpoint is not None:
        epoch = checkpoint["epoch"] + 1
        previous_training_loss = checkpoint["training_loss"]
        previous_validation_loss = checkpoint["validation_loss"]
    else:
        epoch = 1
        previous_training_loss = []
        previous_validation_loss = []
    epochs = range(epoch, epoch+epoch_count)

    # Initialize values to send to the GUI, to be updated throughout training.
    if queue:
        info_gui["progress_epoch"] = (epoch, epochs[-1])
        info_gui["progress_batch"] = (0, 0)
        info_gui["epochs"] = epochs
        info_gui["training_loss"] = []
        info_gui["previous_training_loss"] = previous_training_loss
        info_gui["validation_loss"] = []
        info_gui["previous_validation_loss"] = previous_validation_loss
        queue.put(info_gui)

    # Initialize the loss values for the current training session.
    training_loss = []
    validation_loss = []

    # Main training-validation loop.
    for epoch in epochs:
        print(f"\nEpoch {epoch}/{epochs[-1]} ({time.strftime('%I:%M %p')})")
        time_start = time.time()
        
        # Train on the training dataset.
        model.train(True)
        loss = 0

        for batch, (input_data, label_data) in enumerate(train_dataloader, 1):
            input_data = input_data.to(device).float()
            label_data = label_data.to(device).float()
            
            # Predict an output from the model with the given input.
            output_data = model(input_data)
            # Calculate the loss.
            loss_current = loss_function(output_data, label_data)
            loss += loss_current.item()

            if loss_current is torch.nan:
                print(f"Stopping due to nan loss.")
                break
            
            # Reset gradients of model parameters.
            optimizer.zero_grad()
            # Backpropagate the prediction loss.
            loss_current.backward()
            # Adjust model parameters.
            optimizer.step()

            if batch % 1 == 0:
                print(f"Batch {batch}/{len(train_dataloader)}: {loss/batch:,.2e}...", end="\r")
                if queue:
                    info_gui["progress_batch"] = (batch, len(train_dataloader)+len(validate_dataloader))
                    info_gui["training_loss"] = [*training_loss, loss/batch]
                    queue.put(info_gui)
            
            # Requested to stop from GUI.
            if queue_to_main and not queue_to_main.empty():
                break
        
        print()
        loss /= batch
        training_loss.append(loss)
        print(f"Training loss: {loss:,.2e}")

        # Adjust the learning rate if a scheduler is used.
        if scheduler:
            scheduler.step()
            learning_rate = optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {learning_rate}")
            if queue:
                info_gui["info_training"]["Learning Rate"] = learning_rate
                queue.put(info_gui)

        # Test on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
        model.train(False)
        loss = 0
        outputs = []
        labels = []
        with torch.no_grad():
            for batch, (input_data, label_data) in enumerate(validate_dataloader, 1):
                input_data = input_data.to(device).float()
                label_data = label_data.to(device).float()
                output_data = model(input_data)
                loss += loss_function(output_data, label_data).item()

                # Convert to NumPy arrays for evaluation metric calculations.
                output_data = output_data.cpu().numpy()
                label_data = label_data.cpu().numpy()

                outputs.append(output_data)
                labels.append(label_data)

                if batch % 50 == 0:
                    print(f"Batch {batch}/{len(validate_dataloader)}...", end="\r")
                    if queue:
                        info_gui["progress_batch"] = (len(train_dataloader)+batch, len(train_dataloader)+len(validate_dataloader))
                        queue.put(info_gui)
                
                # Requested to stop from GUI.
                if queue_to_main and not queue_to_main.empty():
                    break
        
        print()
        loss /= batch
        validation_loss.append(loss)
        print(f"Validation loss: {loss:,.2e}")

        # Calculate evaluation metrics on validation results.
        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        results = metrics.evaluate(outputs, labels)
        for metric, value in results.items():
            print(f"{metric}: {value:.3f}")

        # Save the model parameters periodically and in the last iteration of the loop.
        if epoch % save_model_every == 0 or epoch == epochs[-1]:
            save_model(
                filepath_model,
                epoch = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                learning_rate = optimizer.param_groups[0]["lr"],
                training_loss = [*previous_training_loss, *training_loss],
                validation_loss = [*previous_validation_loss, *validation_loss],
            )
        
        # Show the elapsed time during the epoch.
        time_end = time.time()
        duration = time_end - time_start
        if duration >= 60:
            duration_text = f"{duration/60:.1f} minutes"
        else:
            duration_text = f"{duration:.1f} seconds"
        print(f"Finished epoch {epoch} in {duration_text}.")

        if queue:
            info_gui["progress_epoch"] = (epoch, epochs[-1])
            info_gui["training_loss"] = training_loss
            info_gui["validation_loss"] = validation_loss
            info_gui["info_training"]["Epoch Runtime"] = duration_text
            queue.put(info_gui)
        
        # Requested to stop from GUI.
        if queue_to_main and not queue_to_main.empty():
            queue_to_main.queue.clear()
            break
    
    # Plot the loss history.
    if not queue:
        figure = plt.figure()
        
        all_training_loss = [*previous_training_loss, *training_loss]
        all_validation_loss = [*previous_validation_loss, *validation_loss]
        plot_loss(figure, range(1, epochs[-1]+1), [all_training_loss, all_validation_loss], ["Training", "Validation"], start_epoch=epochs[0])

        plt.show()
    
    return model

def test_regression(
    device: str, model: nn.Module, loss_function: nn.Module, dataset: Dataset, test_dataloader: DataLoader,
    queue=None, queue_to_main=None, info_gui: dict=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test the given regression model and return its outputs and corresponding labels and inputs."""

    # Initialize values to send to the GUI.
    if queue:
        info_gui["progress_batch"] = (0, 0)
        info_gui["info_metrics"] = {}
        queue.put(info_gui)

    model.train(False)
    
    loss = 0
    inputs = []
    outputs = []
    labels = []

    with torch.no_grad():
        for batch, (input_data, label_data) in enumerate(test_dataloader, 1):
            input_data = input_data.to(device).float()
            label_data = label_data.to(device).float()
            output_data = model(input_data)
            loss += loss_function(output_data, label_data).item()

            # Convert to NumPy arrays for evaluation metric calculations.
            input_data = input_data.cpu().detach().numpy()
            output_data = output_data.cpu().detach().numpy()
            label_data = label_data.cpu().numpy()

            inputs.append(input_data)
            labels.append(label_data)
            outputs.append(output_data)
            
            if batch % 1 == 0:
                print(f"Batch {batch}/{len(test_dataloader)}...", end="\r")
                if queue:
                    info_gui["progress_batch"] = (batch, len(test_dataloader))
                    queue.put(info_gui)
    
    print()
    loss /= batch
    print(f"Testing loss: {loss:,.2e}")
    
    # Concatenate testing results from all batches into a single array.
    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    if queue:
        info_gui["info_metrics"] = {f"Loss ({loss_function})": loss}
        info_gui["test_inputs"] = inputs
        info_gui["test_outputs"] = outputs
        info_gui["test_labels"] = labels
        info_gui["test_max_value"] = dataset.max_value
        queue.put(info_gui)

    return outputs, labels, inputs

def evaluate_regression(outputs: np.ndarray, labels: np.ndarray, inputs: np.ndarray, dataset: Dataset, queue=None, info_gui: dict=None):
    """Calculate and return evaluation metrics."""

    results = metrics.evaluate(outputs, labels)
    for metric, value in results.items():
        print(f"{metric}: {value:,.3f}")

    # Initialize values to send to the GUI.
    if queue:
        info_gui["info_metrics"] = results
        queue.put(info_gui)
    
    return results


def main(
    epoch_count: int, learning_rate: float, decay_learning_rate: bool, batch_sizes: Tuple[int, int, int], training_split: Tuple[float, float, float], Model: nn.Module,
    filename_model: str, train_existing: bool, save_model_every: int,
    dataset: Dataset,
    train: bool, test: bool, evaluate: bool,
    Optimizer: torch.optim.Optimizer = torch.optim.SGD, Loss: nn.Module = nn.MSELoss,
    queue: Queue = None, queue_to_main: Queue = None,
):
    """
    Function run directly by this file and by the GUI.

    Parameters:
    `train`: Train the model.
    `test`: Test the model.
    `evaluate`: Evaluate the test results, if testing.
    `train_existing`: Load a previously saved model and continue training it.

    `epoch_count`: Number of epochs to train.
    `learning_rate`: Learning rate for the optimizer.
    `batch_sizes`: Tuple of batch sizes for the training, validation, and testing datasets.
    `Model`: A Module subclass to instantiate, not an instance of the class.
    `Optimizer`: An Optimizer subclass to instantiate, not an instance of the class.
    `Loss`: A Module subclass to instantiate, not an instance of the class.

    `dataset`: The Dataset to train on.
    `training_split`: A tuple of three floats in [0, 1] of the training, validation, and testing ratios.
    `filename_model`: Name of the .pth file to load and save to during training.
    
    `queue`: A Queue used to send information to the GUI.
    `queue_to_main`: A Queue used to receive information from the GUI.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    # Initialize values to send to the GUI.
    info_gui = {
        "info_training": {},
        "info_metrics": {},
    } if queue else None
    
    filepath_model = os.path.join('.', filename_model)

    if (test and not train) or (train and train_existing):
        checkpoint = load_model(filepath=filepath_model, device=device)
        # Load the last learning rate used.
        if checkpoint and decay_learning_rate:
            learning_rate = checkpoint["learning_rate"]
            print(f"Using last learning rate {learning_rate}.")
    else:
        checkpoint = None

    # Create the dataset and split it into training, validation, and testing datasets.
    sample_size = len(dataset)
    train_size, validate_size, test_size = [int(split * sample_size) for split in training_split]
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, validate_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split {sample_size:,} samples into {train_size:,} training / {validate_size:,} validation / {test_size:,} testing.")

    batch_size_train, batch_size_validate, batch_size_test = batch_sizes
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size_validate, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)    

    # Initialize the model, optimizer, and loss function.
    model = Model()
    model.to(device)
    optimizer = Optimizer(model.parameters(), lr=learning_rate)
    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = None
    loss_function = Loss()

    # Load previously saved model and optimizer parameters.
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if queue:
            queue.put({
                "epochs": range(1, checkpoint["epoch"]+1),
                "training_loss": checkpoint["training_loss"],
                "validation_loss": checkpoint["validation_loss"],
            })
    
    if queue:
        info_gui["info_training"]["Training Size"] = train_size
        info_gui["info_training"]["Validation Size"] = validate_size
        info_gui["info_training"]["Testing Size"] = test_size
        info_gui["info_training"]["Learning Rate"] = learning_rate
        queue.put(info_gui)

    if train:
        model = train_regression(
            device = device,
            epoch_count = epoch_count,
            checkpoint = checkpoint,
            filepath_model = filepath_model,
            save_model_every = save_model_every,
            model = model,
            optimizer = optimizer,
            loss_function = loss_function,
            train_dataloader = train_dataloader,
            validate_dataloader = validate_dataloader,
            scheduler = scheduler,
            queue = queue,
            queue_to_main = queue_to_main,
            info_gui = info_gui,
            )
    
    if test:
        outputs, labels, inputs = test_regression(
            device = device,
            model = model,
            loss_function = loss_function,
            dataset = dataset,
            test_dataloader = test_dataloader,
            queue = queue,
            queue_to_main = queue_to_main,
            info_gui = info_gui,
        )

        if evaluate:
            results = evaluate_regression(outputs, labels, inputs, dataset, queue=queue, info_gui=info_gui)


if __name__ == "__main__":
    kwargs = {
        "filename_model": "model.pth",
        "train_existing": not True,
        "save_model_every": 5,

        "epoch_count": 5,
        "learning_rate": 1e-2,
        "decay_learning_rate": not True,
        "batch_sizes": (8, 64, 64),
        "training_split": (0.8, 0.1, 0.1),
        "Model": LatticeCnn,
        "Optimizer": torch.optim.SGD,
        "Loss": nn.MSELoss,
        
        "dataset": LatticeDataset(),
        
        "train": True,
        "test": True,
        "evaluate": True,
    }

    main(**kwargs)