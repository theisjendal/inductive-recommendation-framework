from copy import deepcopy

from models.bert4rec.bert4rec_recommender import BERT4RecRecommender
from models.bpr.bpr_dgl_recommender import BPRDGLRecommender
from models.graphsage.graphsage_dgl_recommender import GraphSAGEDGLRecommender
from models.idcf.idcf_dgl_recommender import IDCFDGLRecommender
from models.inmo.inmo_recommender import INMORecommender
from models.ncf.ncf_recommender import NCFRecommender
from models.ginrec.ginrec_recommender import GInRecRecommender
from models.kgat.kgat_dgl_recommender import KGATDGLRecommender
from models.ngcf.ngcf_dgl_recommender import NGCFDGLRecommender
from models.pinsage.pinsage_recommender import PinSAGERecommender
from models.ppr.ppr_dgl_recommender import PPRDGLRecommender
from models.random.random_dgl_recommender import RandomDGLRecommender
from models.toppop.toppop_dgl_recommender import TopPopDGLRecommender

inductive_methods = {
    'bert4rec': {
        'model': BERT4RecRecommender,
        'graphs': ['cg_pos'],
        'use_cuda': True,
        'hyperparameters': {
            'learning_rates':  [1e-5, 0.01],
            'weight_decays': [1e-5, 100.],
            'dropouts': [0., 0.8],
            'att_dropouts': [0.2],
            'activations': ['gelu'],
            'dims': [64],
            'normalization_ranges': [0.02],
            'layer_dims': [256],
            'max_position_embeddings': [200],
            'num_attention_heads': [2],
            'num_hidden_layers': [2],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous', 'weight_decays': 'continuous',
                         'att_dropouts': 'choice', 'activations': 'choice', 'dims': 'choice',
                         'normalization_ranges': 'choice', 'layer_dims': 'choice', 'max_position_embeddings': 'choice',
                         'num_attention_heads': 'choice', 'num_hidden_layers': 'choice', 'type_vocab_size':'choice',
                         'vocab_size': 'choice'}
    },
    'ginrec': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'n_layer_samples': [25, 10, 5],
        'hyperparameters':  {
            'learning_rates': [0.1, 1e-5],
            'dropouts': [0, 0.5],
            'autoencoder_weights': [1e-8, 2.],
            'gate_types': ['concat'],
            'weight_decays': [1e-12, 1.],
            'aggregators': ['bi-interaction'],
            'use_ntype': [False],
            'sampling_on_inference': [False],
            'l2_loss': [0],
            'disentanglement': [False],
            # 'disentanglement_weight': [0.1, 0.01, 0.001, 1e-12, 0.2, 0.5, 1, 0],
            'sample_time': [False],
            'timed_batching': [True, False],
            'use_global': ['mean', 'none'],
            'normalizations': ['none'],
            'neighbor_sampling_methods': ['none'],
            'local_names': ['mean', 'none'],
            'neg_sampling_methods': ['uniform'],
            'layers': [4, 3, 2],
            'optimizers': ['AdamW'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'continuous', 'aggregators': 'choice', 'l2_loss': 'choice',
                         'disentanglement': 'choice', 'disentanglement_weight': 'choice',
                         'sampling_on_inference': 'choice', 'use_ntype': 'choice',
                         'sample_time': 'choice', 'use_global': 'choice', 'timed_batching': 'choice',
                         'use_local': 'choice', 'normalizations': 'choice', 'neighbor_sampling_methods': 'choice',
                         'local_names': 'choice', 'neg_sampling_methods': 'choice', 'layers': 'choice',
                         'optimizers': 'choice'}
    },
    'simplerec2': {
        'model': GInRecRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'n_layer_samples': [25, 10, 5],
        'hyperparameters':  {
            'learning_rates': [0.1, 1e-5],
            'dropouts': [0, 0.5],
            'autoencoder_weights': [1e-8, 2.],
            'gate_types': ['concat'],
            'weight_decays': [1e-12, 1.],
            'aggregators': ['bi-interaction'],
            'use_ntype': [False],
            'sampling_on_inference': [False],
            'l2_loss': [0],
            'disentanglement': [False],
            # 'disentanglement_weight': [0.1, 0.01, 0.001, 1e-12, 0.2, 0.5, 1, 0],
            'sample_time': [False],
            'timed_batching': [True, False],
            'use_global': ['mean', 'none'],
            'normalizations': ['none'],
            'neighbor_sampling_methods': ['none'],
            'local_names': ['mean', 'none'],
            'neg_sampling_methods': ['uniform'],
            'layers': [4, 3, 2],
            'optimizers': ['AdamW'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'autoencoder_weights': 'continuous', 'gate_types': 'choice',
                         'weight_decays': 'continuous', 'aggregators': 'choice', 'l2_loss': 'choice',
                         'disentanglement': 'choice', 'disentanglement_weight': 'choice',
                         'sampling_on_inference': 'choice', 'use_ntype': 'choice',
                         'sample_time': 'choice', 'use_global': 'choice', 'timed_batching': 'choice',
                         'use_local': 'choice', 'normalizations': 'choice', 'neighbor_sampling_methods': 'choice',
                         'local_names': 'choice', 'neg_sampling_methods': 'choice', 'layers': 'choice',
                         'optimizers': 'choice'}
    },
    'bpr': {
        'model': BPRDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [0.1, 1e-5],
            'latent_factors': [8, 32],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'latent_factors': 'discrete'}
    },
    'graphsage': {
        'model': GraphSAGEDGLRecommender,
        'use_cuda': True,
        'graphs': ['kg', 'cg_pos'],
        'hyperparameters': {
            'learning_rates': [0.01, 1e-5],
            'dropouts': [0., 0.8],
            'aggregators': ['mean', 'pool'],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'aggregators': 'choice'}
    },
    'inmo': {
        'model': INMORecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates':  [1e-4, 0.005],
            'weight_decays': [1e-5, 100.],
            'dropouts': [0., 0.8],
            'n_layers': [1, 2, 3],
            'auxiliary_weight': [1e-4, 1],
            'dim': [64]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous', 'weight_decays': 'continuous',
                         'attention_samples': 'choice', 'dims': 'choice', 'auxiliary_weight': 'continuous',
                         'dim': 'choice', 'n_layers': 'choice'}
    },
    'idcf': {
        'model': IDCFDGLRecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'attention_samples': [200, 500, 2000],
            'learning_rates': [0.1, 1e-5],
            'weight_decays': [1e-8, .2],
            'dims': ['(32, [32, 32, 1])', '(64, [64, 64, 1])']
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous', 'attention_samples': 'choice',
                         'dims': 'choice'}
    },
    'ppr': {
        'model': PPRDGLRecommender,
        'graphs': ['ckg_pos'],
        'hyperparameters': {
            'alphas': [0.25, 0.45, 0.65, 0.85],
        },
        'gridsearch': True,
        'sherpa_types': {'alphas': 'choice'}
    },
    'ppr-cf': {
        'model': PPRDGLRecommender,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'alphas': [0.25, 0.45, 0.65, 0.85],
        },
        'gridsearch': True,
        'sherpa_types': {'alphas': 'choice'}
    },
    'pinsage': {
        'model': PinSAGERecommender,
        'use_cuda': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [1.e-5, 0.1],
            'dropouts': [0., 0.8],
            'weight_decays': [1e-8, .1],
            'deltas': [0.01, 32.]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous', 'deltas': 'continuous'},
        'features': True
    },
    'toppop': {
        'model': TopPopDGLRecommender,
        'graphs': ['cg_pos'],
        'gridsearch': True,
        'hyperparameters': {'cut_off': [-1, 1] + list(range(5, 100, 5))},
        'sherpa_types': {'cut_off': 'choice'}
    },
}

base_models = {
    'lightgcn': {
        'model': NGCFDGLRecommender,
        'graphs': ['cg_pos'],
        'use_cuda': True,
        'hyperparameters': {
            'learning_rates':  [1e-4, 0.005],
            'weight_decays': [1e-5, 100.],
            'dropouts': [0., 0.8]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous'}
    },
    'ngcf': {
        'model': NGCFDGLRecommender,
        'graphs': ['cg_pos'],
        'use_cuda': True,
        'lightgcn': False,
        'hyperparameters': {
            'learning_rates':  [1e-4, 0.005],
            'weight_decays': [0., 100.],
            'dropouts': [0., 0.8]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous'}
    },
    'ncf': {
        'model': NCFRecommender,
        'use_cuda': True,
        'learned_embeddings': True,
        'graphs': ['cg_pos'],
        'hyperparameters': {
            'learning_rates': [0.1, 0.001],
            'weight_decays': [1e-5, 100.],
            'layers': [1, 6]
        },
        'sherpa_types': {'learning_rates': 'continuous', 'layers': 'discrete', 'weight_decays': 'continuous',
                         'anchors': 'choice'}
    },
    'kgat': {
        'model': KGATDGLRecommender,
        'use_cuda': True,
        'graphs': ['ckg_rev'],
        'hyperparameters': {
            'learning_rates': [0.05, 0.001],
            'dropouts': [0., 0.8],
            'weight_decays': [1e-5, 100.],
        },
        'sherpa_types': {'learning_rates': 'continuous', 'dropouts': 'continuous',
                         'weight_decays': 'continuous'}
    },
    'random': {
        'model': RandomDGLRecommender,
        'graphs': [],
        'hyperparameters': {}
    }
}


# Update dictionary with ablation studies
dgl_models = {}
dgl_models.update(inductive_methods)
dgl_models.update(base_models)