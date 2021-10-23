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
    out: 2D tensor (torch.FloatTensor)
    actual: 1D tensor (torch.LongTensor)
    """
    # print('out', out)

    # retrieve labels from device by converting to numpy arrays
    actual = actual.cpu().detach().numpy()
    # print('actual', actual)
    # convert output to class labels
    pred = _convert_output(out)
    # print('pred', pred)
    # compare predictions against actual
    n_hits = np.sum(pred == actual)
    # print('hits', n_hits)
    return n_hits


class Evaluation:

    def __init__(self, val_loader, criterion_func, devicehandler):
        logging.info('Loading evaluation phase...')
        self.val_loader = val_loader
        self.criterion_func = criterion_func
        self.devicehandler = devicehandler

    def evaluate_model(self, epoch, num_epochs, model):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')
        # output_list = []
        # targets_list = []
        val_loss = 0
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # if i == 0 or i == 1:
                #     print()
                #     print(i, 'target', targets.shape, targets[:20])

                # forward pass
                out = model.forward(inputs)
                # if i == 0 or i == 1:
                #     # print('out', out.shape, out[:2])
                #     print('pred', _convert_output(out).shape, _convert_output(out[:20]))
                #     print()

                # calculate validation loss
                loss = self.criterion_func(out, targets)
                val_loss += loss.item()

                # calculate number of accurate predictions for this batch
                out = out.cpu().detach().numpy()  # extract from gpu
                num_hits += _calculate_num_hits(out, targets)

                # # debug save output
                # output_list.append(_convert_output(out))
                # targets_list.append(targets.cpu().detach().numpy())

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = num_hits / len(self.val_loader.dataset)

            # # debug write output
            # combined_output = np.concatenate(output_list, axis=0)
            # combined_targets = np.concatenate(targets_list, axis=0)
            # df_output = pd.DataFrame(combined_output)
            # df_targets = pd.DataFrame(combined_targets)
            # df = pd.concat([df_output, df_targets],axis=1)
            # fname = f'/home/ubuntu/evaluation.epoch{epoch}.{datetime.now().strftime("%Y%m%d.%H.%M.%S")}.csv'
            # logging.info(f'evaluation filename:{fname}')
            # df.to_csv(fname, header=False, index=False)

            return val_loss, val_acc


class EvaluationCenterLoss:

    def __init__(self, val_loader, label_criterion_func, centerloss_func, centerloss_weight, devicehandler):
        logging.info('Loading evaluation phase...')
        self.val_loader = val_loader
        self.label_criterion_func = label_criterion_func
        self.centerloss_func = centerloss_func
        self.centerloss_weight = centerloss_weight
        self.devicehandler = devicehandler

    def evaluate_model(self, epoch, num_epochs, model):
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')
        val_loss = 0
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                feature, outputs = model.forward(inputs, return_embedding=True)

                # calculate validation loss
                l_loss = self.label_criterion_func(outputs, targets)
                c_loss = self.centerloss_func(feature, targets)
                loss = l_loss + c_loss * self.centerloss_weight
                val_loss += loss.item()

                # calculate number of accurate predictions for this batch
                outputs = outputs.cpu().detach().numpy()  # extract from gpu
                num_hits += _calculate_num_hits(outputs, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = num_hits / len(self.val_loader.dataset)

            return val_loss, val_acc
