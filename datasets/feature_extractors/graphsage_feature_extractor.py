from typing import List, Union

import dgl
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk import tokenize

from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from datasets.feature_extractors.utility import extract_degree
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration
from shared.entity import Entity
from shared.enums import FeatureEnum, ExperimentEnum
from shared.utility import load_entities


class GraphSAGEFeatureExtractor(FeatureExtractionBase):
    """
    Extracts features similar to the method described in GraphSAGE
    """

    def __init__(self, attributes, use_cuda=False):
        super().__init__(use_cuda, 'text', return_for_all_settings=True,
                         name_suffix=['train', 'validation', 'test'])
        self._model = SentenceTransformer('stsb-roberta-base')
        self.attributes = attributes
        self.text_dim = 768
        self.batch_size = 1024
        self._model.eval()

        if self._use_cuda:
            self._model.to('cuda')

    def _text_to_vec(self, text: List[str]):
        with torch.no_grad():
            # Get embeddings
            sentence_embedding = self._model.encode(text, self.batch_size)

        return sentence_embedding  # Ensure numpy for later processing

    def flatten(self, s, phase_users):
        if isinstance(s, str):
            return [s]
        elif isinstance(s, tuple):  # Review handling
            if s[0] in phase_users:
                return self.flatten(s[1], phase_users)
            else:
                return []
        elif isinstance(s, list):
            r = []
            for s_ in s:
                r.extend(self.flatten(s_, phase_users))
            return r
        elif s is None:
            return []
        else:
            raise ValueError(f'Invalid type {type(s)} found.')

    def _get_text(self, entity: Entity, phase_users):
        text = [getattr(entity, a) for a in self.attributes]

        text = self.flatten(text, phase_users)
        text = list(filter(lambda x: bool(x), text))

        grouped_text = []
        for t in text:
            grouped_text.append(tokenize.sent_tokenize(t))

        return grouped_text

    def _load_graph(self, dataset_path, fold, feature_configuration, experiment, graph_name=None, graph_type='train'):
        g1 = super()._load_graph(dataset_path, fold, feature_configuration, experiment, graph_name, graph_type)
        g2 = super()._load_graph(dataset_path, fold, feature_configuration, experiment, 'cg_pos', graph_type)
        return g1, g2

    # def extract_features(self, graph: dgl.DGLGraph, feature_configuration: FeatureConfiguration, path: str,
    #                      ws_g: dgl.DGLGraph = None, mappings=None) -> np.ndarray:
    def extract_features(self, data_path: str, feature_configuration: FeatureConfiguration,
                         experiment: ExperimentConfiguration, cold_experiment: ExperimentConfiguration = None,
                         fold: str = 0) -> Union[np.ndarray, List[np.ndarray]]:
        entities, meta = self.load_experiment(data_path, experiment, limit=['entities', 'meta'])

        graphs = [self._load_graph(data_path, fold, feature_configuration, experiment)]

        if experiment.experiment.TEMPORAL == ExperimentEnum.TEMPORAL:
            graphs.append(self._load_graph(data_path, fold, feature_configuration, experiment, graph_type='validation'))
            graphs.append(self._load_graph(data_path, fold, feature_configuration, experiment, graph_type='test'))

        # TODO filter reviews that are not in the graph
        n_entities = len(entities)

        users = []
        items = []
        desc_entities = list(set(meta.entities).difference(meta.items))
        for _, g in graphs:
            e = torch.cat(g.edges()).unique()
            u = e[e >= len(meta.entities)]
            items.append(e[e < len(meta.items)].tolist())
            users.append((u - len(meta.entities)).tolist())

        # Length is the text dim, number of entity types as one hot, and node degree (undirected)
        feature_length = self.text_dim + 1
        all_features = []
        for (graph, _), phase_users, phase_items in zip(graphs, users, items):
            features = np.zeros((n_entities, feature_length), dtype=np.float32)
            batch_size = self.batch_size

            phase_entities = desc_entities + phase_items

            # Only entities that are in the graph
            batches = [[entities[i] for i in phase_entities[b * batch_size:(b + 1) * batch_size]]
                       for b in range(len(phase_entities) // batch_size + 1)]

            for i, batch in tqdm(enumerate(batches), total=len(batches), desc='Extracting features from entities'):
                texts = []
                sum_counts = []
                inner_features = np.zeros((len(batch), feature_length))
                for j, entity in enumerate(batch):
                    # Get text
                    text_list = self._get_text(entity, phase_users)

                    # flatten
                    sum_counts.append([len(tx) for tx in text_list])
                    texts.extend([t for ts in text_list for t in ts])
                    inner_features[j, -1] = graph.out_degrees(entity.index)

                text_features = self._text_to_vec(texts)
                index = 0
                for i, counts in enumerate(sum_counts):
                    f = np.zeros((len(counts), self.text_dim))
                    for j, count in enumerate(counts):
                        f[j] = np.sum(text_features[index:index + count], axis=0)
                        index += count

                    inner_features[i, :self.text_dim] = np.sum(f, axis=0)

                features[[entity.index for entity in batch]] = inner_features

            if feature_configuration.feature_span == FeatureEnum.ITEMS:
                features = features[[entity.index for entity in entities if entity.recommendable]]
            elif feature_configuration.feature_span == FeatureEnum.DESC_ENTITIES:
                features = features[[entity.index for entity in entities if entity.recommendable]]
            elif feature_configuration.feature_span == FeatureEnum.ENTITIES:
                pass  # Already limited to entities
            else:
                raise NotImplementedError()

            all_features.append(features)

        return all_features
