"""
Contains all Formatter objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import pandas as pd
import numpy as np
import json
import logging
import os


def _convert_output(out):
    # convert 2D output to 1D a single class label (4000 nodes into a single number per output)
    out = np.argmax(out, axis=1)  # column with max value in each row is the index of the predicted label

    return out


class OutputFormatter:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _read_class_to_idx_json(self):
        source = os.path.join(self.data_dir, 'class_to_idx.json')
        logging.info(f'Reading class_to_idx mapping from {source}...')
        with open(source) as mapping_source:
            mapping = json.load(mapping_source)
            return mapping

    def format_output(self, out):
        labels = _convert_output(out)

        # add an index column
        df = pd.DataFrame(labels).reset_index(drop=False)
        # print(df.columns)
        # change column names
        df = df.rename(columns={0: "output_label", 'index': 'idprefix'})

        # add .jpg to the id column
        df = df.astype({'idprefix': 'str'})
        df['idsuffix'] = '.jpg'
        df['id'] = df['idprefix'] + df['idsuffix']

        # drop extra columns generated
        df = df.drop(['idprefix', 'idsuffix'], axis=1)

        # remap output labels to correct labels based on ImageFolder (see Note 1 in imagedatasethandler)
        mapping = self._read_class_to_idx_json()
        df['label'] = df.apply(lambda x: list(mapping.keys())[list(mapping.values()).index(x['output_label'])], axis=1)

        # ensure id is first column
        df = df[['id', 'label', 'output_label']]

        return df
