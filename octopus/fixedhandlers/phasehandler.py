"""
All things related to managing training, evaluation, and testing phases.
"""
__author__ = 'ryanquinnnelson'

import logging
import time


class PhaseHandler:

    def __init__(self, num_epochs, outputhandler, devicehandler, statshandler, checkpointhandler,
                 schedulerhandler, wandbconnector, formatter, load_from_checkpoint, checkpoint_file=None):
        logging.info('Initializing phase handling...')
        self.load_from_checkpoint = load_from_checkpoint
        self.checkpoint_file = checkpoint_file
        self.first_epoch = 1
        self.num_epochs = num_epochs

        # handlers
        self.outputhandler = outputhandler
        self.devicehandler = devicehandler
        self.statshandler = statshandler
        self.checkpointhandler = checkpointhandler
        self.schedulerhandler = schedulerhandler
        self.wandbconnector = wandbconnector

        # formatter for test output
        self.formatter = formatter

    def _load_checkpoint(self, model, optimizer, scheduler):
        device = self.devicehandler.get_device()
        checkpoint = self.checkpointhandler.load(self.checkpoint_file, device, model, optimizer, scheduler)

        # restore stats
        self.statshandler.stats = checkpoint['stats']

        # set which epoch to start from
        self.first_epoch = checkpoint['next_epoch']

    def process_epochs(self, model, optimizer, scheduler, training, evaluation, testing):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self._load_checkpoint(model, optimizer, scheduler)

            # submit old stats to wandb to align with other runs
            self.statshandler.report_previous_stats(self.wandbconnector)

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_loss = training.train_model(epoch, self.num_epochs, model, optimizer)

            # validate
            val_loss, val_metric = evaluation.evaluate_model(epoch, self.num_epochs, model)

            # test
            out = testing.test_model(epoch, self.num_epochs, model)
            out = self.formatter.format_output(out)
            self.outputhandler.save(out, epoch)

            # stats
            end = time.time()
            self.statshandler.collect_stats(epoch, train_loss, val_loss, val_metric, start, end)
            self.statshandler.report_stats(self.wandbconnector)

            # scheduler
            self.schedulerhandler.update_scheduler(scheduler, self.statshandler.stats)

            # save model checkpoint
            self.checkpointhandler.save(model, optimizer, scheduler, epoch + 1, self.statshandler.stats)

            # check if early stopping criteria is met
            if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
                logging.info('Early stopping criteria is met. Stopping the training process...')
                break  # stop running epochs

    def process_epochs_centerloss(self, model, label_optimizer, centerloss_optimizer, label_scheduler,
                                  centerloss_scheduler, training, evaluation, testing):

        # # load checkpoint if necessary
        # if self.load_from_checkpoint:
        #     self._load_checkpoint(model, optimizer, scheduler)

            # # submit old stats to wandb to align with other runs
            # self.statshandler.report_previous_stats(self.wandbconnector)

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_loss = training.train_model(epoch, self.num_epochs, model, centerloss_optimizer, label_optimizer)

            # validate
            val_loss, val_metric = evaluation.evaluate_model(epoch, self.num_epochs, model)

            # test
            out = testing.test_model(epoch, self.num_epochs, model)
            out = self.formatter.format_output(out)
            self.outputhandler.save(out, epoch)

            # stats
            end = time.time()
            self.statshandler.collect_stats(epoch, train_loss, val_loss, val_metric, start, end)
            self.statshandler.report_stats(self.wandbconnector)

            # scheduler
            self.schedulerhandler.update_scheduler(label_scheduler, self.statshandler.stats)
            self.schedulerhandler.update_scheduler(centerloss_scheduler, self.statshandler.stats)

            # # save model checkpoint
            # self.checkpointhandler.save(model, optimizer, scheduler, epoch + 1, self.statshandler.stats)

            # check if early stopping criteria is met
            if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
                logging.info('Early stopping criteria is met. Stopping the training process...')
                break  # stop running epochs
