"""
Defines the testing phase of model training.
"""
__author__ = 'ryanquinnnelson'

import logging
import torch
import numpy as np
import pandas as pd


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


class Testing2:
    def __init__(self, test_loader, devicehandler):
        logging.info('Loading testing phase...')
        self.test_loader = test_loader
        self.devicehandler = devicehandler

    def test_model(self, epoch, num_epochs, model):
        logging.info(f'Running epoch {epoch}/{num_epochs} of testing...')
        output = []
        filenames_list = []

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, filenames) in enumerate(self.test_loader):
                targets = None

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # capture output for mini-batch
                out = out.cpu().detach().numpy()  # extract from gpu if necessary
                out = np.argmax(out, axis=1)
                output.append(out)
                filenames_list.append(filenames)

        combined_filenames = np.concatenate(filenames_list, axis=0)
        combined = np.concatenate(output, axis=0)

        # create df of contents
        df_output = pd.DataFrame(combined)
        df_output = df_output.rename(columns={0: "label"})
        print(df_output.head())

        df_filenames = pd.DataFrame(combined_filenames)
        df_filenames = df_filenames.rename(columns={0: "id"})
        print(df_filenames.head())

        df = pd.concat([df_output, df_filenames], axis=1)
        print(df.head())

        return df[['id','label']]
