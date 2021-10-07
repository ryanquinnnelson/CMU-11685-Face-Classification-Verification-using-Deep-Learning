"""
All things related to data output.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
from datetime import datetime
import octopus.helper as utilities


class OutputHandler:
    def __init__(self, run_name, output_dir):
        logging.info('Initializing output handling...')
        self.run_name = run_name
        self.output_dir = output_dir

    def setup(self):
        logging.info('Preparing output directory...')
        utilities.create_directory(self.output_dir)

    def save(self, df, epoch):
        # generate filename
        filename = f'{self.run_name}.epoch{epoch}.{datetime.now().strftime("%Y%m%d.%H.%M.%S")}.output.csv'
        path = os.path.join(self.output_dir, filename)

        logging.info(f'Saving test output to {path}...')

        # save output
        df.to_csv(path, header=True, index=False)
