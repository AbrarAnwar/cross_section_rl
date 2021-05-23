import gym

from envs.simple_cross_section_env_local import SimpleCrossSectionEnv
gym.envs.register(id='simple_cross_section_env_local-v0',
        entry_point='envs.simple_cross_section_env_local:SimpleCrossSectionEnv',
        reward_threshold=40.0
        )

from envs.simple_cross_section_env_weighted_local import SimpleCrossSectionEnv
gym.envs.register(id='simple_cross_section_env_weighted_local-v0',
        entry_point='envs.simple_cross_section_env_weighted_local:SimpleCrossSectionEnv',
        reward_threshold=40.0
        )

from envs.simple_cross_section_env_local_jumpy import SimpleCrossSectionEnv
gym.envs.register(id='simple_cross_section_env_local_jumpy-v0',
        entry_point='envs.simple_cross_section_env_local_jumpy:SimpleCrossSectionEnv',
        reward_threshold=40.0
        )
