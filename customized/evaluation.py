"""
Evaluation phase customized to this dataset.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch
import numpy as np


def _convert_output(out):
    # convert 2D output to 1D a single class label (4000 nodes into a single number per output)
    out = np.argmax(out, axis=1)  # column with max value in each row is the index of the predicted label

    return out


def _calculate_num_hits(out, actual):
    """
    out: 2D tensor (torch.FloatTensor), each row has 71 columns (one for each possible label)
    actual: 1D tensor (torch.LongTensor)
    """
    print('out', out)

    # retrieve labels from device by converting to numpy arrays
    actual = actual.cpu().detach().numpy()
    print('actual', actual)
    # convert output to class labels
    pred = _convert_output(out)
    print('pred', pred)
    # compare predictions against actual
    n_hits = np.sum(pred == actual)
    print('hits', n_hits)
    return n_hits


class Evaluation:

    def __init__(self, val_loader, criterion_func, devicehandler):
        logging.info('Loading evaluation phase...')
        self.val_loader = val_loader
        self.criterion_func = criterion_func
        self.devicehandler = devicehandler

    def evaluate_model(self, epoch, num_epochs, model):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')
        val_loss = 0
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for (inputs, targets) in self.val_loader:
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # calculate validation loss
                loss = self.criterion_func(out, targets)
                val_loss += loss.item()

                # calculate number of accurate predictions for this batch
                out = out.cpu().detach().numpy()  # extract from gpu
                num_hits += _calculate_num_hits(out, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = num_hits / len(self.val_loader.dataset)

            return val_loss, val_acc
