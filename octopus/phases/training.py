"""
Defines the training phase of model training.
"""
__author__ = 'ryanquinnnelson'

import logging
import torch


class Training:

    def __init__(self, train_loader, criterion_func, devicehandler):
        logging.info('Loading training phase...')
        self.train_loader = train_loader
        self.criterion_func = criterion_func
        self.devicehandler = devicehandler

    def train_model(self, epoch, num_epochs, model, optimizer):
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')
        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
            # prep
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

            # compute forward pass
            out = model.forward(inputs)

            # calculate loss
            loss = self.criterion_func(out, targets)
            train_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss


class TrainingCenterLoss:

    def __init__(self, train_loader, label_criterion_func, centerloss_func, centerloss_weight, devicehandler):
        logging.info('Loading training phase...')
        self.train_loader = train_loader
        self.label_criterion_func = label_criterion_func
        self.centerloss_func = centerloss_func
        self.devicehandler = devicehandler
        self.centerloss_weight = centerloss_weight

    def train_model(self, epoch, num_epochs, model, optimizer_class, optimizer_label):
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')
        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
            # prep
            optimizer_class.zero_grad()
            optimizer_label.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

            # compute forward pass
            feature, outputs = model.forward(inputs, return_embedding=True)

            # calculate loss
            l_loss = self.label_criterion_func(outputs, targets)
            c_loss = self.centerloss_func(feature, targets)
            loss = l_loss + self.centerloss_weight * c_loss
            train_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer_class.step()
            for param in self.centerloss_func.parameters():
                param.grad.data *= (1.0 / self.centerloss_weight)
            optimizer_class.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss
