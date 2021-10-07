"""
All things related to models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models.CNN import CNN2d


def _convert_dict_to_dicts(d):
    # determine number of dictionaries to create
    layer_number_set = set()
    for key in d:
        layer_number, layer_parm = key.strip().split('.')
        layer_number_set.add(int(layer_number))
    num_layers = len(layer_number_set)

    # create a dictionary of parmeters for each layer, in layer order (1,2,3...)
    layer_dicts = []
    for i in range(1, num_layers + 1):  # extract values in order

        layer_dict = {}

        # find all dictionary entries that start with this layer name
        # extract parameter names and values and place into a dictionary for this layer
        for key in d:
            layer_number, layer_parm = key.strip().split('.')
            if int(layer_number) == i:
                layer_dict[layer_parm] = d[key]
        layer_dicts.append(layer_dict)

    return layer_dicts


class CnnHandler:
    def __init__(self,
                 model_type, input_size, output_size, activation_func, batch_norm, conv_dict, pool_class, pool_dict):
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.batch_norm = batch_norm
        self.conv_dicts = _convert_dict_to_dicts(conv_dict)
        self.pool_class = pool_class
        self.pool_dicts = _convert_dict_to_dicts(pool_dict)

    def get_model(self):
        logging.info('Initializing model...')
        model = None

        if self.model_type == 'CNN2d':
            model = CNN2d(self.input_size, self.output_size, self.activation_func, self.batch_norm, self.conv_dicts,
                          self.pool_class, self.pool_dicts)

        logging.info(f'Model initialized:\n{model}')
        return model
