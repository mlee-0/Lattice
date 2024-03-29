import os
from queue import Queue
import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import *
from metrics import *
from models import *
from visualization import *


CHECKPOINTS_FOLDER = 'Checkpoints'


def plot_loss(losses_training: List[float], losses_validation: List[float]) -> None:
    plt.figure()
    plt.semilogy(range(1, len(losses_training)+1), losses_training, '-', label='Training')
    plt.semilogy(range(1, len(losses_validation)+1), losses_validation, '-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def save_model(filepath: str, **kwargs) -> None:
    """Save model parameters to a file."""
    torch.save(kwargs, filepath)
    print(f"Saved model to {filepath}.")

def load_model(filepath: str, device: str='cpu') -> dict:
    """Return a dictionary of model parameters from a file."""
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except FileNotFoundError:
        print(f"{filepath} not found.")
    else:
        print(f"Loaded model from {filepath} trained for {checkpoint['epoch']} epochs.")
        return checkpoint

def train(
    device: str, epoch_count: int, checkpoint: dict, filepath_model: str, save_model_every: int, save_best_separately: bool,
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_function: nn.Module,
    train_dataloader: DataLoader, validate_dataloader: DataLoader,
    scheduler = None,
    queue=None, queue_to_main=None, info_gui: dict=None,
) -> nn.Module:
    """Train and validate the given model and return the model after finishing training."""

    # Load the previous training history.
    epoch = checkpoint.get('epoch', 0) + 1
    epochs = range(epoch, epoch+epoch_count)
    training_loss = checkpoint.get('training_loss', [])
    validation_loss = checkpoint.get('validation_loss', [])

    # Initialize values to send to the GUI, to be updated throughout training.
    if queue:
        info_gui["progress_epoch"] = (epoch, epochs[-1])
        info_gui["progress_batch"] = (0, 0)
        info_gui["epochs"] = epochs
        info_gui["training_loss"] = training_loss
        info_gui["validation_loss"] = validation_loss
        queue.put(info_gui)

    # Main training-validation loop.
    for epoch in epochs:
        print(f"\nEpoch {epoch}/{epochs[-1]} ({time.strftime('%I:%M %p')})")
        time_start = time.time()
        
        # Train on the training dataset.
        model.train(True)
        loss = 0

        try:
            for batch, (input_data, label_data) in enumerate(train_dataloader, 1):
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                output_data = model(input_data)

                # Calculate the loss.
                loss_current = loss_function(output_data, label_data)
                loss += loss_current.item() * label_data.size(0)

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
        
        loss /= len(train_dataloader.dataset)
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
            for batch, (input_data, label_data) in enumerate(validate_dataloader, 1):
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                output_data = model(input_data)
                loss += loss_function(output_data, label_data).item() * label_data.size(0)

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
        
        loss /= len(validate_dataloader.dataset)
        validation_loss.append(loss)
        print(f"\nValidation loss: {loss:,.2e}")

        # Calculate evaluation metrics on validation results.
        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        # train_dataloader.dataset.dataset.unnormalize(outputs)
        # train_dataloader.dataset.dataset.unnormalize(labels)
        outputs /= LatticeDataset.DIAMETER_SCALE
        labels /= LatticeDataset.DIAMETER_SCALE
        results = evaluate(outputs, labels)

        # Save the model periodically and in the last epoch.
        if epoch % save_model_every == 0 or epoch == epochs[-1]:
            save_model(
                filepath_model,
                epoch = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                learning_rate = optimizer.param_groups[0]['lr'],
                training_loss = training_loss,
                validation_loss = validation_loss,
            )
        # Save the model if the model achieved the lowest validation loss so far.
        if save_best_separately and validation_loss[-1] <= min(validation_loss):
            save_model(
                f"{filepath_model[:-4]}[best]{filepath_model[-4:]}",
                epoch = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                learning_rate = optimizer.param_groups[0]['lr'],
                training_loss = training_loss,
                validation_loss = validation_loss,
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
            queue.put(info_gui)
        
        # Requested to stop from GUI.
        if queue_to_main and not queue_to_main.empty():
            queue_to_main.queue.clear()
            break

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

    for batch, (input_data, label_data) in enumerate(test_dataloader, 1):
        try:
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            output_data = model(input_data)
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

def evaluate(outputs: np.ndarray, labels: np.ndarray):
    """Print and return evaluation metrics."""

    results = {
        'Mean Error': me(outputs, labels),
        'MAE': mae(outputs, labels),
        'MSE': mse(outputs, labels),
        'MRE': mre(outputs, labels),
        'Nonzero MAE': mae(outputs[labels > 0], labels[labels > 0]),
        'Nonzero MSE': mse(outputs[labels > 0], labels[labels > 0]),
        'Nonzero MRE': mre(outputs[labels > 0], labels[labels > 0]),
        'Min error': min_error(outputs, labels),
        'Max error': max_error(outputs, labels),
        # 'Zeros correct': fraction_of_zeros_correct(outputs, labels),
        # 'Zeros incorrect': fraction_of_zeros_incorrect(outputs, labels),
        # 'Number of values out of bounds': ((outputs < 0) + (outputs > 1)).sum(),
        # 'Fraction of values out of bounds': ((outputs < 0) + (outputs > 1)).sum() / outputs.size,
    }
    for metric, value in results.items():
        print(f"{metric}: {value:,.5f}")

    return results

@torch.no_grad()
def infer_lattice(model: nn.Module, filename_model: str, dataset: Dataset) -> torch.Tensor:
    """Make predictions using a trained model on a dataset without labels."""

    checkpoint = load_model(os.path.join(CHECKPOINTS_FOLDER, filename_model))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)
    output = model(dataset[0:1])

    return output

@torch.no_grad()
def infer_strut(model: nn.Module, filename_model: str, dataset: Dataset, batch_size: int) -> Tuple[list, list, torch.Tensor]:
    """Make predictions using a trained model on a dataset without labels. Defined as a generator function to allow visualizing intermediate results one by one."""

    checkpoint = load_model(os.path.join(CHECKPOINTS_FOLDER, filename_model))
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

def infer_femur():
    dataset = FemurDataset()

    checkpoint = load_model(os.path.join(CHECKPOINTS_FOLDER, 'LatticeNet.pth'))
    model = LatticeNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        output = model(dataset.inputs[0:1])
        output = output[0].numpy()
        output /= 100
    
    # Remove invalid struts.
    for channel, x, y, z in zip(*np.nonzero(output)):
        dx, dy, dz = DIRECTIONS[channel]
        try:
            if dataset.inputs[0, 1, x + dx, y + dy, z + dz] < 0:
                output[channel, x, y, z] = 0
        except IndexError:
            output[channel, x, y, z] = 0
    
    actor = make_actor_lattice(*convert_array_to_lattice(output), resolution=3)
    visualize_actors(actor, gui=True)

def main(
    epoch_count: int, learning_rate: float, decay_learning_rate: bool, batch_sizes: Tuple[int, int, int], data_split: Tuple[float, float, float], dataset: Dataset, model: nn.Module,
    filename_model: str, train_existing: bool, save_model_every: int, save_best_separately: bool,
    train_model: bool, test_model: bool, show_loss: bool, show_parity: bool, show_predictions: bool,
    Optimizer: torch.optim.Optimizer = torch.optim.SGD, loss_function: nn.Module = nn.MSELoss(),
    queue: Queue = None, queue_to_main: Queue = None,
):
    """
    Parameters:
    `train_model`: Train the model.
    `test_model`: Test the model.
    `show_predictions`: Show plots or visualizations.
    `train_existing`: Load a previously saved model and continue training it.

    `epoch_count`: Number of epochs to train.
    `learning_rate`: Learning rate for the optimizer.
    `batch_sizes`: Tuple of batch sizes for the training, validation, and testing datasets.
    `data_split`: Tuple of three floats in [0, 1] of the relative training, validation, and testing dataset sizes.
    `dataset`: The Dataset to train on.
    `filename_model`: Name of the .pth file to load and save to during training.
    `model`: The network, as an instance of Module.
    `Optimizer`: An Optimizer subclass to instantiate, not an instance of the class.
    `loss_function`: A callable as an instantiated Module subclass.

    `queue`: A Queue used to send information to the GUI.
    `queue_to_main`: A Queue used to receive information from the GUI.
    """

    device = 'cpu'  #"cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device.")

    # Initialize values to send to the GUI.
    info_gui = {
        "info_metrics": {},
    } if queue else None
    
    filepath_model = os.path.join(CHECKPOINTS_FOLDER, filename_model)

    if (test_model and not train_model) or (train_model and train_existing):
        checkpoint = load_model(filepath=filepath_model, device=device)
        # Load the last learning rate used.
        if checkpoint and decay_learning_rate:
            learning_rate = checkpoint["learning_rate"]
            print(f"Using last learning rate {learning_rate}.")
    else:
        checkpoint = {}

    # Split the dataset into training, validation, and testing datasets. Split by input image instead of by struts to ensure that input images in the training set do not appear in the validation or testing sets.
    image_indices = list(range(dataset.inputs.size(0)))
    random.shuffle(image_indices)

    train_dataset, validate_dataset, test_dataset = random_split(
        dataset,
        lengths=[int(split * len(dataset)) for split in data_split],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"\nSplit {len(dataset):,} data into {len(train_dataset):,} training / {len(validate_dataset):,} validation / {len(test_dataset):,} testing.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, drop_last=False)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_sizes[1], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False)

    # Initialize the model and optimizer.
    model.to(device)
    optimizer = Optimizer(model.parameters(), lr=learning_rate)
    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = None

    # Load previously saved model and optimizer parameters.
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            save_best_separately = save_best_separately,
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

    # Show the loss history.
    if show_loss:
        checkpoint = load_model(filepath=filepath_model)
        training_loss = checkpoint.get('training_loss', [])
        validation_loss = checkpoint.get('validation_loss', [])
        plot_loss(training_loss, validation_loss)

    # Load the best model.
    checkpoint = load_model(f"{filepath_model[:-4]}[best]{filepath_model[-4:]}")
    model.load_state_dict(checkpoint['model_state_dict'])

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

        # Scale predictions and labels [0, 1].
        outputs /= LatticeDataset.DIAMETER_SCALE
        labels /= LatticeDataset.DIAMETER_SCALE

        results = evaluate(outputs, labels)

        # # Remove invalid struts.
        # for i, channel, x, y, z in zip(*np.nonzero(outputs)):
        #     dx, dy, dz = DIRECTIONS[channel]
        #     try:
        #         if inputs[i, 1, x + dx, y + dy, z + dz] < 0:
        #             outputs[i, channel, x, y, z] = 0
        #     except IndexError:
        #         continue
        #         # outputs[i, channel, x, y, z] = 0

        # Show a parity plot.
        if show_parity:
            plt.plot(labels.flatten(), outputs.flatten(), '.')
            plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.show()

        # Show the prediction and label side-by-side, with label shown on right.
        if show_predictions:
            for index in random.sample(range(len(test_dataset)), k=1):
                actor_output = make_actor_lattice(*convert_array_to_lattice(outputs[index, ...]))
                actor_label = make_actor_lattice(*convert_array_to_lattice(labels[index, ...]), translation=(12, 0, 0))
                visualize_actors(actor_output, actor_label, gui=True)


if __name__ == "__main__":
    main(
        train_model = not True,
        test_model = True,
        show_loss = True,
        show_parity = not True,
        show_predictions = True,

        train_existing = True,
        filename_model = 'LatticeNet.pth',
        save_model_every = 5,
        save_best_separately = True,

        epoch_count = 50,
        learning_rate = 1e-3,
        decay_learning_rate = False,
        batch_sizes = (64, 64, 64),
        data_split = (0.8, 0.1, 0.1),
        
        dataset = LatticeDataset(normalize_inputs=True),
        model = LatticeNet(),
        Optimizer = torch.optim.Adam,
        loss_function = nn.MSELoss(),
    )