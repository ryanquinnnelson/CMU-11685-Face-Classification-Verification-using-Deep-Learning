"""
Performs environment setup for deep learning and runs a deep learning pipeline.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import sys

# execute before loading torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # better error tracking from gpu

# local modules
from octopus.helper import _to_string_list, _to_float_dict, _to_int_dict
from octopus.connectors.kaggleconnector import KaggleConnector
from octopus.connectors.wandbconnector import WandbConnector
from octopus.fixedhandlers.checkpointhandler import CheckpointHandler
from octopus.fixedhandlers.devicehandler import DeviceHandler
from octopus.fixedhandlers.criterionhandler import CriterionHandler
from octopus.fixedhandlers.optimizerhandler import OptimizerHandler
from octopus.fixedhandlers.schedulerhandler import SchedulerHandler
from octopus.fixedhandlers.statshandler import StatsHandler
from octopus.fixedhandlers.phasehandler import PhaseHandler
from octopus.fixedhandlers.dataloaderhandler import DataLoaderHandler
from octopus.fixedhandlers.outputhandler import OutputHandler
from octopus.datasethandlers.imagedatasethandler import ImageDatasetHandler
from octopus.modelhandlers.cnnhandler import CnnHandler
from octopus.phases.training import Training
from octopus.phases.testing import Testing

# customized to this data
from customized.formatters import OutputFormatter
from customized.evaluation import Evaluation


class Octopus:

    def __init__(self, config, config_file):
        # save configuration
        self.config = config

        # logging
        _setup_logging(config['debug']['debug_file'])
        _draw_logo()
        logging.info('Initializing octopus...')
        logging.info(f'Parsing configuration from {config_file}...')

        # connectors
        self.kaggleconnector, self.wandbconnector = initialize_connectors(config)

        # fixed handlers
        self.checkpointhandler, self.criterionhandler, self.dataloaderhandler, self.devicehandler, \
        self.optimizerhandler, self.outputhandler, self.schedulerhandler, self.statshandler, \
        self.phasehandler = initialize_fixed_handlers(config, self.wandbconnector)

        # variable handlers
        self.inputhandler, self.modelhandler = initialize_variable_handlers(config)

        logging.info('octopus initialization is complete.')

    def setup_environment(self):
        logging.info('octopus is setting up the environment...')

        # wandb
        self.wandbconnector.setup()

        # kaggle
        if self.kaggleconnector:
            self.kaggleconnector.setup()

        # checkpoint directory
        self.checkpointhandler.setup()

        # output directory
        self.outputhandler.setup()

        # device
        self.devicehandler.setup()

        # dataloaders
        self.dataloaderhandler.setup(self.devicehandler.device)

        logging.info('octopus has finished setting up the environment.')

    def download_data(self):
        if self.kaggleconnector:
            logging.info('octopus is downloading data...')
            self.kaggleconnector.download_and_unzip()
            logging.info('octopus has finished downloading data.')
        else:
            logging.info('octopus is not downloading data.')
            logging.info(f'octopus expects data to be available in {self.inputhandler.data_dir}.')

    def run_pipeline(self):
        """
        Note 1:
        Reason behind moving model to device first:
        https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
        """
        logging.info('octopus is running the pipeline...')

        # initialize model
        model = self.modelhandler.get_model()
        self.devicehandler.move_model_to_device(model)  # move model before initializing optimizer - see Note 1
        self.wandbconnector.watch(model)

        # initialize model components
        loss_func = self.criterionhandler.get_loss_function()
        optimizer = self.optimizerhandler.get_optimizer(model)
        scheduler = self.schedulerhandler.get_scheduler(optimizer)

        # load data
        train_loader, val_loader, test_loader = self.dataloaderhandler.load(self.inputhandler)

        # load phases
        training = Training(train_loader, loss_func, self.devicehandler)
        evaluation = Evaluation(val_loader, loss_func, self.devicehandler)
        testing = Testing(test_loader, self.devicehandler)

        # run epochs
        self.phasehandler.process_epochs(model, optimizer, scheduler, training, evaluation, testing)

        logging.info('octopus has finished running the pipeline.')

    def cleanup(self):
        logging.info('octopus shutdown complete.')


def _setup_logging(debug_file):
    # create directory if it doesn't exist
    debug_path = os.path.dirname(debug_file)
    if not os.path.isdir(debug_path):
        os.mkdir(debug_path)

    # delete older debug file if it exists
    if os.path.isfile(debug_file):
        os.remove(debug_file)

    # define basic logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.handlers = []  # clear out previous handlers

    # write to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # write to debug file
    handler = logging.FileHandler(debug_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _draw_logo():
    logging.info('              _---_')
    logging.info('            /       \\')
    logging.info('           |         |')
    logging.info('   _--_    |         |    _--_')
    logging.info('  /__  \\   \\  0   0  /   /  __\\')
    logging.info('     \\  \\   \\       /   /  /')
    logging.info('      \\  -__-       -__-  /')
    logging.info('  |\\   \\    __     __    /   /|')
    logging.info('  | \\___----         ----___/ |')
    logging.info('  \\                           /')
    logging.info('   --___--/    / \\    \\--___--')
    logging.info('         /    /   \\    \\')
    logging.info('   --___-    /     \\    -___--')
    logging.info('   \\_    __-         -__    _/')
    logging.info('     ----               ----')
    logging.info('')
    logging.info('       O  C  T  O  P  U  S')
    logging.info('')


def initialize_connectors(config):
    # kaggle
    if config['kaggle'].getboolean('download_from_kaggle'):
        kaggleconnector = KaggleConnector(config['kaggle']['kaggle_dir'],
                                          config['kaggle']['content_dir'],
                                          config['kaggle']['token_file'],
                                          config['kaggle']['competition'],
                                          config['kaggle'].getboolean('delete_zipfiles_after_unzipping'))
    else:
        kaggleconnector = None

    # wandb
    hyper_dict = dict(config['hyperparameters'])
    hyper_dict.update(dict(config['model']))  # get all hyperparameters from different parts of config
    wandbconnector = WandbConnector(config['wandb']['wandb_dir'],
                                    config['wandb']['entity'],
                                    config['DEFAULT']['run_name'],
                                    config['wandb']['project'],
                                    config['wandb']['notes'],
                                    _to_string_list(config['wandb']['tags']),
                                    hyper_dict)

    return kaggleconnector, wandbconnector


def initialize_fixed_handlers(config, wandbconnector):
    # checkpoints
    checkpointhandler = CheckpointHandler(config['checkpoint']['checkpoint_dir'],
                                          config['checkpoint'].getboolean('delete_existing_checkpoints'),
                                          config['DEFAULT']['run_name'],
                                          config['checkpoint'].getboolean('load_from_checkpoint'))

    # criterion
    criterionhandler = CriterionHandler(config['hyperparameters']['criterion_type'])

    # device
    devicehandler = DeviceHandler()

    # dataloader
    dataloaderhandler = DataLoaderHandler(config['dataloader'].getint('batch_size'),
                                          config['dataloader'].getint('num_workers'),
                                          config['dataloader'].getboolean('pin_memory'))

    # optimizer
    optimizerhandler = OptimizerHandler(config['hyperparameters']['optimizer_type'],
                                        _to_float_dict(config['hyperparameters']['optimizer_kwargs']))

    # output
    outputhandler = OutputHandler(config['DEFAULT']['run_name'],
                                  config['output']['output_dir'])

    # scheduler
    schedulerhandler = SchedulerHandler(config['hyperparameters']['scheduler_type'],
                                        _to_float_dict(config['hyperparameters']['scheduler_kwargs']),
                                        config['hyperparameters']['scheduler_plateau_metric'])

    # statshandler
    statshandler = StatsHandler(config['stats']['val_metric_name'],
                                config['stats']['comparison_metric'],
                                config['stats'].getboolean('comparison_best_is_max'),
                                config['stats'].getint('comparison_patience'))

    # phasehandler
    if config.has_option('checkpoint', 'checkpoint_file'):
        checkpoint_file = config['checkpoint']['checkpoint_file']
    else:
        checkpoint_file = None

    phasehandler = PhaseHandler(config['hyperparameters'].getint('num_epochs'),
                                outputhandler,
                                devicehandler,
                                statshandler,
                                checkpointhandler,
                                schedulerhandler,
                                wandbconnector,
                                OutputFormatter(),
                                config['checkpoint'].getboolean('load_from_checkpoint'),
                                checkpoint_file)

    return checkpointhandler, criterionhandler, dataloaderhandler, devicehandler, \
           optimizerhandler, outputhandler, schedulerhandler, statshandler, phasehandler


# TODO add alternative input and model handlers for MLP
def initialize_variable_handlers(config):
    # input
    if config['data']['data_type'] == 'image':
        inputhandler = ImageDatasetHandler(config['data']['data_dir'],
                                           config['data']['train_dir'],
                                           config['data']['val_dir'],
                                           config['data']['test_dir'],
                                           _to_string_list(config['data']['transforms_list']))
    else:
        inputhandler = None

    # model
    if config['model']['model_type'] == 'CNN2d':
        modelhandler = CnnHandler(config['model']['model_type'],
                                  config['model'].getint('input_size'),
                                  config['model'].getint('output_size'),
                                  config['model']['activation_func'],
                                  config['model'].getboolean('batch_norm'),
                                  _to_int_dict(config['model']['conv_kwargs']),
                                  config['model']['pool_class'],
                                  _to_int_dict(config['model']['pool_kwargs']))
    else:
        modelhandler = None

    return inputhandler, modelhandler
