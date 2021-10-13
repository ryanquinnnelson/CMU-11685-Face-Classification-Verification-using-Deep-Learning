"""
Defines the testing phase of model training.
"""
__author__ = 'ryanquinnnelson'

import logging
import torch
import numpy as np


class Testing:
    def __init__(self, test_loader, devicehandler):
        logging.info('Loading testing phase...')
        self.test_loader = test_loader
        self.devicehandler = devicehandler

    def test_model(self, epoch, num_epochs, model):
        logging.info(f'Running epoch {epoch}/{num_epochs} of testing...')
        output = []

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, batch in enumerate(self.test_loader):
                # print(i)
                #     print(batch[:2])

                if type(batch) is tuple:
                    # loader contains inputs and targets
                    inputs = batch[0]
                    targets = batch[1]
                else:
                    # loader contains only inputs
                    inputs = batch
                    targets = None

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # capture output for mini-batch
                out = out.cpu().detach().numpy()  # extract from gpu if necessary
                output.append(out)

                # if i < 2:
                #     print('out\n', np.argmax(out, axis=1))
                #     print()
                # print(i, 'output length', out.shape)

        combined = np.concatenate(output, axis=0)
        # print('combined', combined.shape)
        return combined
