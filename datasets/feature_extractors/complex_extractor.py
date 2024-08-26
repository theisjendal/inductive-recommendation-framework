import inspect
import os.path

import dgl
import numpy as np
import torch

from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration
from shared.enums import FeatureEnum
from shared.utility import load_entities


class ComplEXFeatureExtractor(FeatureExtractionBase):
    """
    Traines ComplEX and saves features for entities.
    """
    def __init__(self, model_name='transe', use_cuda=False, **model_kwargs):
        super_kwargs = {}
        for k in model_kwargs:
            if k in inspect.signature(FeatureExtractionBase.__init__).parameters:
                super_kwargs[k] = model_kwargs.pop(k)
        if len(model_kwargs) == 0:
            model_kwargs = {'embedding_dim': 32}  # default kwargs
        super().__init__(use_cuda, 'pykeen', **super_kwargs)
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.batch_size = 16384
        self.num_workers = 2

    def extract_features(self, data_path: str, feature_configuration: FeatureConfiguration,
                         experiment: ExperimentConfiguration, cold_experiment: ExperimentConfiguration=None,
                         fold: str = 0) -> np.ndarray:

        try:
            from pykeen.pipeline import pipeline
            from pykeen.triples import CoreTriplesFactory
            import logging
            logging.disable(logging.INFO)
        except ModuleNotFoundError:
            print('WARNING: Switch python venv if using complex extractor or install pykeen.'
                  'If using docker installation guide, then simply use complex_env')
            exit(1)

        model_kwargs = self.model_kwargs
        training_kwargs = {'batch_size': self.batch_size, 'num_workers': self.num_workers}

        # Can add model specific fixed arguments here
        if self.model_name == 'transe':
            pass
        elif self.model_name == 'transr':
            pass
        elif self.model_name == 'complex':
            pass
        else:
            raise NotImplementedError()

        entities, = self.load_experiment(data_path, experiment, limit=['entities'])
        graph_type = 'train' if feature_configuration.feature_span != FeatureEnum.SUMMARY_NODES else None
        graph = self._load_graph(data_path, fold, feature_configuration, experiment, graph_type=graph_type)
        u, v, eid = graph.edges('all')
        types = graph.edata['type'][eid]
        triples = torch.stack([u, types, v]).T
        n_entities = torch.max(triples) + 1  # Assume num nodes > num relations
        n_relations = torch.max(triples[:, 1]) + 1
        factory = CoreTriplesFactory(triples, n_entities, n_relations, torch.arange(n_entities),
                                     torch.arange(n_relations), create_inverse_triples=True)

        ratios = [0.8, 0.1, 0.1]
        training_factory, testing_factory, validation_factory = factory.split(ratios, random_state=experiment.seed)
        result = pipeline(training=training_factory, validation=validation_factory, testing=testing_factory,
                          model=self.model_name, epochs=1000, random_seed=experiment.seed, stopper='early',
                          stopper_kwargs={'frequency': 5, 'patience': 10, 'relative_delta': 1e-8},
                          model_kwargs=model_kwargs, training_kwargs=training_kwargs)

        print("Train hits@k: ",  result.get_metric('hits@k'))

        if self.model_name == 'transe':
            embeddings = result.model.state_dict()['entity_embeddings._embeddings.weight']
        elif self.model_name in ['complex', 'transr']:
            embeddings = result.model.state_dict()['entity_representations.0._embeddings.weight']
        else:
            raise NotImplementedError()

        if (feature_configuration.feature_span == FeatureEnum.ENTITIES or
                feature_configuration.feature_span == FeatureEnum.SUMMARY_NODES):
            features = embeddings
        elif feature_configuration.feature_span == FeatureEnum.DESC_ENTITIES:
            features = embeddings[torch.tensor([e.index for e in entities if not e.recommendable])]
        else:
            raise NotImplementedError()

        return features.cpu().numpy()
