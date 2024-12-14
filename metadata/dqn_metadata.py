from torch import nn,optim
from stable_baselines3.common.torch_layers import FlattenExtractor

METADATA = {
        'policy':'MlpPolicy',
        'learning_rate': 0.0001,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'replay_buffer_class': None,
        'replay_buffer_kwargs': None,
        'optimize_memory_usage': False, 
        'target_update_interval': 10000,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'max_grad_norm': 10,
        'stats_window_size': 100,
        'tensorboard_log': None,
        'policy_kwargs': {
                    'net_arch': [100,50,20,10,5],
                    'activation_fn': nn.modules.activation.LeakyReLU,
                    'features_extractor_class':FlattenExtractor,
                    'features_extractor_kwargs': None,
                    'normalize_images': True,
                    'optimizer_class': optim.Adam,
                    'optimizer_kwargs': None
                },
        'verbose': 1,
        'seed': None,
        'device': 'auto',
        '_init_setup_model': True
    }
