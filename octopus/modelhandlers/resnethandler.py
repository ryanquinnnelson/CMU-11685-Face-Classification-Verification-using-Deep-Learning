import logging

import octopus.models.resnets as resnets


class ResnetHandler:
    def __init__(self, model_type, in_features, num_classes):
        logging.info('Initializing model handling...')
        self.model_type = model_type
        self.in_features = in_features
        self.num_classes = num_classes

    def get_model(self):
        logging.info('Initializing model...')
        model = None

        if self.model_type == 'Resnet18':
            model = resnets.Resnet18(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet34':
            model = resnets.Resnet34_v2(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet50':
            model = resnets.Resnet50(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet101':
            model = resnets.Resnet101(self.in_features, self.num_classes)

        elif self.model_type == 'Resnet152':
            model = resnets.Resnet152(self.in_features, self.num_classes)

        logging.info(f'Model initialized:\n{model}')
        return model


class ResnetHandlerCenterLoss:
    def __init__(self, model_type, in_features, num_classes, feat_dim):
        logging.info('Initializing model handling...')
        self.model_type = model_type
        self.in_features = in_features
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def get_model(self):
        logging.info('Initializing model...')
        model = None

        if self.model_type == 'Resnet18':
            model = resnets.Resnet18(self.in_features, self.num_classes, self.feat_dim)

        elif self.model_type == 'Resnet34':
            model = resnets.Resnet34(self.in_features, self.num_classes, self.feat_dim)

        logging.info(f'Model initialized:\n{model}')
        return model
