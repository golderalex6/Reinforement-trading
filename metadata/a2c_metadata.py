from torch import nn,optim
from stable_baselines3.common.torch_layers import FlattenExtractor

METADATA = {
        'policy':'MlpPolicy',
        'learning_rate': 0.0007,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'rms_prop_eps': 1e-05,
        'use_rms_prop': True,
        'use_sde': False,
        'sde_sample_freq': -1,
        'rollout_buffer_class': None,
        'rollout_buffer_kwargs': None,
        'normalize_advantage': False,
        'stats_window_size': 100,
        'tensorboard_log': None,
        'policy_kwargs': {
                'net_arch': [100,50,20,10,5],
                'activation_fn': nn.modules.activation.Tanh,
                'ortho_init': True,
                'log_std_init': 0.0,
                'full_std': True,
                'use_expln': False,
                'squash_output': False,
                'features_extractor_class': FlattenExtractor,
                'features_extractor_kwargs': None,
                'share_features_extractor': True,
                'normalize_images': True,
                'optimizer_class': optim.Adam,
                'optimizer_kwargs': None
            },
        'verbose': 0,
        'seed': None,
        'device': 'auto',
        '_init_setup_model': True
    }

if __name__ == '__main__':
    print(metadata)
