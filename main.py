import os
from queue import Queue
import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
# import torch_geometric

from datasets import *
import metrics
from models import *
from preprocessing import DATASET_FOLDER
from visualization import *


random.seed(42)


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

    # Plot each set of loss values.
    for i, loss_i in enumerate(loss):
        if not len(loss_i):
            continue
        axis.plot(epochs[:len(loss_i)], loss_i, ".-", label=labels[i])
        axis.annotate(f"{loss_i[-1]:,.2e}", (epochs[-1 - (len(epochs)-len(loss_i))], loss_i[-1]), fontsize=10)
    
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
    print(f"Saved model to {filepath}.")

def load_model(filepath: str, device: str) -> dict:
    """Return a dictionary of model parameters from a file."""
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except FileNotFoundError:
        print(f"{filepath} not found.")
    else:
        print(f"Loaded model from {filepath} trained for {checkpoint['epoch']} epochs.")
        return checkpoint

def train(
    device: str, epoch_count: int, checkpoint: dict, filepath_model: str, save_model_every: int,
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_function: nn.Module,
    train_dataloader: DataLoader, validate_dataloader: DataLoader,
    scheduler = None,
    queue=None, queue_to_main=None, info_gui: dict=None,
) -> nn.Module:
    """Train and validate the given model and return the model after finishing training."""

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

        try:
            for batch, (input_data, coordinates, label_data) in enumerate(train_dataloader, 1):
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                output_data = model(input_data, coordinates)

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

                if batch % 10 == 0:
                    print(f"\rBatch {batch}/{len(train_dataloader)}: {loss_current.item():,.2e}...", end="")
                    if queue:
                        info_gui["progress_batch"] = (batch, len(train_dataloader)+len(validate_dataloader))
                        info_gui["training_loss"] = [*training_loss, loss/batch]
                        queue.put(info_gui)
                
                # Requested to stop from GUI.
                if queue_to_main and not queue_to_main.empty():
                    break

        except KeyboardInterrupt:
            break
        
        loss /= batch
        training_loss.append(loss)
        print(f"\nTraining loss: {loss:,.2e}")

        # Adjust the learning rate if a scheduler is used.
        if scheduler:
            scheduler.step()
            learning_rate = optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {learning_rate}")

        # Test on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
        model.train(False)
        loss = 0
        outputs = []
        labels = []
        with torch.no_grad():
            for batch, (input_data, coordinates, label_data) in enumerate(validate_dataloader, 1):
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                output_data = model(input_data, coordinates)
                loss += loss_function(output_data, label_data).item()

                # Convert to NumPy arrays for evaluation metric calculations.
                output_data = output_data.cpu().numpy()
                label_data = label_data.cpu().numpy()

                outputs.append(output_data)
                labels.append(label_data)

                if batch % 10 == 0:
                    print(f"\rBatch {batch}/{len(validate_dataloader)}...", end="")
                    if queue:
                        info_gui["progress_batch"] = (len(train_dataloader)+batch, len(train_dataloader)+len(validate_dataloader))
                        queue.put(info_gui)
                
                # Requested to stop from GUI.
                if queue_to_main and not queue_to_main.empty():
                    break
        
        loss /= batch
        validation_loss.append(loss)
        print(f"\nValidation loss: {loss:,.2e}")

        # Calculate evaluation metrics on validation results.
        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        # train_dataloader.dataset.dataset.unnormalize(outputs)
        # train_dataloader.dataset.dataset.unnormalize(labels)
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
        
        # # Show weights as a 3D cube, if training MLP
        # if epoch % 50 == 0:
        #     with torch.no_grad():
        #         w = torch.clone(model.get_parameter('linear.0.weight')).view((11, 11, 11))
        #         w -= w.min()
        #         w /= w.max()
        #         w *= 255
        #         visualize_input(w[:6, :, :], opacity=1)

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

@torch.no_grad()
def test(
    device: str, model: nn.Module, loss_function: nn.Module, test_dataloader: DataLoader,
    queue=None, queue_to_main=None, info_gui: dict=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test the given model and return its outputs and corresponding labels and inputs."""

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

    for batch, (input_data, coordinates, label_data) in enumerate(test_dataloader, 1):
        try:
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            output_data = model(input_data, coordinates)
            loss += loss_function(output_data, label_data).item()

            # Convert to NumPy arrays for evaluation metric calculations.
            input_data = input_data.cpu().detach().numpy()
            output_data = output_data.cpu().detach().numpy()
            label_data = label_data.cpu().numpy()

            inputs.append(input_data)
            labels.append(label_data)
            outputs.append(output_data)
            
            if batch % 10 == 0:
                print(f"\rBatch {batch}/{len(test_dataloader)}...", end="")
                if queue:
                    info_gui["progress_batch"] = (batch, len(test_dataloader))
                    queue.put(info_gui)
        
        except KeyboardInterrupt:
            break
    
    loss /= batch
    print(f"\nTesting loss: {loss:,.2e}")

    return outputs, labels, inputs

def evaluate(outputs: np.ndarray, labels: np.ndarray, inputs: np.ndarray, dataset: Dataset, queue=None, info_gui: dict=None):
    """Calculate and return evaluation metrics."""

    results = metrics.evaluate(outputs, labels)
    for metric, value in results.items():
        print(f"{metric}: {value:,.3f}")

    # Initialize values to send to the GUI.
    if queue:
        info_gui["info_metrics"] = results
        queue.put(info_gui)
    
    return results

@torch.no_grad()
def infer(model: nn.Module, filename_model: str, dataset: Dataset, batch_size: int):
    """Make predictions using a trained model on a dataset without corresponding labels. Defined as a generator function to allow visualizing intermediate results one by one."""

    checkpoint = load_model(os.path.join(DATASET_FOLDER, filename_model), 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    loader = DataLoader(dataset, batch_size=batch_size)

    # Make diameter predictions.
    diameters = torch.tensor([0.05] * len(dataset))
    locations_1 = list([_[0] for _ in dataset.indices])
    locations_2 = list([_[1] for _ in dataset.indices])
    yield locations_1, locations_2, diameters

    for i, input_ in enumerate(loader, 1):
        diameter = model(input_)
        diameters[(i-1)*batch_size:i*batch_size] = diameter.squeeze()
        yield locations_1, locations_2, diameters


def main(
    epoch_count: int, learning_rate: float, decay_learning_rate: bool, batch_sizes: Tuple[int, int, int], data_split: Tuple[float, float, float], dataset: Dataset, Model: nn.Module,
    filename_model: str, train_existing: bool, save_model_every: int,
    train_model: bool, test_model: bool, visualize_results: bool,
    Optimizer: torch.optim.Optimizer = torch.optim.SGD, loss_function: nn.Module = nn.MSELoss(),
    queue: Queue = None, queue_to_main: Queue = None,
):
    """
    Parameters:
    `train_model`: Train the model.
    `test_model`: Test the model.
    `visualize_results`: Show plots or visualizations.
    `train_existing`: Load a previously saved model and continue training it.

    `epoch_count`: Number of epochs to train.
    `learning_rate`: Learning rate for the optimizer.
    `batch_sizes`: Tuple of batch sizes for the training, validation, and testing datasets.
    `data_split`: Tuple of three floats in [0, 1] of the relative training, validation, and testing dataset sizes.
    `dataset`: The Dataset to train on.
    `filename_model`: Name of the .pth file to load and save to during training.
    `Model`: A Module subclass to instantiate, not an instance of the class.
    `Optimizer`: An Optimizer subclass to instantiate, not an instance of the class.
    `loss_function`: A callable as an instantiated Module subclass.

    `queue`: A Queue used to send information to the GUI.
    `queue_to_main`: A Queue used to receive information from the GUI.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device.")

    # Initialize values to send to the GUI.
    info_gui = {
        "info_metrics": {},
    } if queue else None
    
    filepath_model = os.path.join(DATASET_FOLDER, filename_model)

    if (test_model and not train_model) or (train_model and train_existing):
        checkpoint = load_model(filepath=filepath_model, device=device)
        # Load the last learning rate used.
        if checkpoint and decay_learning_rate:
            learning_rate = checkpoint["learning_rate"]
            print(f"Using last learning rate {learning_rate}.")
    else:
        checkpoint = None

    # Split the dataset into training, validation, and testing datasets. Split by input image instead of by struts to ensure that input images in the training set do not appear in the validation or testing sets.
    image_indices = list(range(dataset.inputs.size(0)))
    random.shuffle(image_indices)

    train_size, validate_size, test_size = [int(split * dataset.inputs.size(0)) for split in data_split]
    train_image_indices = image_indices[:train_size]
    validate_image_indices = image_indices[train_size:train_size+validate_size]
    test_image_indices = image_indices[-test_size:]

    train_indices = dataset.outputs_for_images(train_image_indices)
    validate_indices = dataset.outputs_for_images(validate_image_indices)
    test_indices = dataset.outputs_for_images(test_image_indices)

    train_dataset = Subset(dataset, train_indices)
    validate_dataset = Subset(dataset, validate_indices)
    test_dataset = Subset(dataset, test_indices)
    train_size, validate_size, test_size = len(train_dataset), len(validate_dataset), len(test_dataset)
    dataset_size = sum([train_size, validate_size, test_size])

    # dataset_size = len(dataset)
    # train_size, validate_size, test_size = [int(split * dataset_size) for split in data_split]
    # train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset,
    #     [train_size, validate_size, test_size],
    #     generator=torch.Generator().manual_seed(42),
    # )

    print(f"\nSplit {dataset_size:,} samples into {train_size:,} training / {validate_size:,} validation / {test_size:,} testing.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, drop_last=False)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_sizes[1], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False)

    # Initialize the model and optimizer.
    model = Model()
    model.to(device)
    print(f"Using model {type(model).__name__} with {get_parameter_count(model):,} parameters.")
    optimizer = Optimizer(model.parameters(), lr=learning_rate)
    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = None

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

    if train_model:
        model = train(
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
    
    if test_model:
        outputs, labels, inputs = test(
            device = device,
            model = model,
            loss_function = loss_function,
            test_dataloader = test_dataloader,
            queue = queue,
            queue_to_main = queue_to_main,
            info_gui = info_gui,
        )

        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        inputs = np.concatenate(inputs, axis=0)

        results = evaluate(outputs, labels, inputs, dataset, queue=queue, info_gui=info_gui)

        if visualize_results:
            # Calculate (x, y, z) coordinates of each node.
            locations_1 = []
            locations_2 = []
            for i in range(inputs.shape[0]):
                coordinates = np.argwhere(inputs[i, 1, ...])
                assert coordinates.shape[0] == 2
                locations_1.append(tuple(coordinates[0, :]))
                locations_2.append(tuple(coordinates[1, :]))

            metrics.plot_histograms(outputs, labels, bins=20)
            metrics.plot_predicted_vs_true(outputs, labels)
            metrics.plot_error_by_angle(outputs, labels, locations_1, locations_2)
            metrics.plot_error_by_edge_distance(outputs, labels, locations_1, locations_2)
            metrics.plot_error_by_xy_edge_distance(outputs, labels, locations_1, locations_2)

            # # If predicting local strut diameters, visualize the predictions on all struts in a single lattice structure.
            # dataset.p = 1.0
            # indices = dataset.outputs_for_images([test_image_indices[0]])
            # loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
            # outputs, labels, inputs = test(
            #     device = device,
            #     model = model,
            #     loss_function = loss_function,
            #     test_dataloader = loader,
            #     queue = queue,
            #     queue_to_main = queue_to_main,
            #     info_gui = info_gui,
            # )

            # # Calculate (x, y, z) coordinates of each node.
            # locations_1 = []
            # locations_2 = []
            # for input_ in inputs:
            #     coordinates = np.argwhere(input_[0, 1, ...])
            #     assert coordinates.shape[0] == 2
            #     locations_1.append(tuple(coordinates[0, :]))
            #     locations_2.append(tuple(coordinates[1, :]))

            # visualize_lattice(locations_1, locations_2, outputs) #, true_diameters=labels)
            # visualize_lattice(locations_1, locations_2, labels)


            # graph = copy.deepcopy(test_dataset[0])
            # i = 1527
            # graph.edge_attr = outputs[:i, :]
            # graph.y = graph.y[:i, :]

            # i = 3

            # graph = test_dataset[i]
            # graph.edge_attr = outputs[i]
            # graph.y = labels[i]
            # lattice = convert_graph_to_lattice(graph)
            # visualize_lattice(*lattice, [_.item() for _ in graph.y])


if __name__ == "__main__":
    kwargs = {
        "train_model": True,
        "test_model": True,
        "visualize_results": True,

        "train_existing": not True,
        "filename_model": "model.pth",
        "save_model_every": 1,

        "epoch_count": 5,
        "learning_rate": 1e-3,
        "decay_learning_rate": False,
        "batch_sizes": (64, 64, 64),
        "data_split": (0.8, 0.1, 0.1),
        
        "dataset": StrutDataset(p=0.25, normalize_inputs=True),
        "Model": ResNetMasked,
        "Optimizer": torch.optim.Adam,
        "loss_function": nn.MSELoss(),
    }
    
    main(**kwargs)


    # generator = infer(
    #     model=ResNet(),
    #     filename_model="model_5conv_res.pth",
    #     dataset=InferenceDataset('circle'),
    #     batch_size=100,
    # )

    # tic = time.time()
    # for locations_1, locations_2, diameters in generator:
    #     visualize_lattice(locations_1, locations_2, diameters, gui=False, screenshot_filename=f"{i:03}")
    # toc = time.time()
    # print(f"Generated {len(dataset)} struts in {toc - tic:.1f} seconds.")
    # visualize_lattice(locations_1, locations_2, diameters, gui=True)