import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym



from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.actor_critic_agents.SAC import SAC
from envs.simple_cross_section_env_local_jumpy import SimpleCrossSectionEnv


config = Config()
config.seed = 1
input_file = '/home/abrar/cross_section_rl/data/cross_section_data/sphere_resampled.npz'
env = gym.make('simple_cross_section_env_local_jumpy-v0', input_file='/home/abrar/cross_section_rl/data/cross_section_data/sphere_resampled_10verts_10sections.npz', same_obs_size=True, \
     k_state_neighborhood=2, previous_mesh_neighborhood=2, next_mesh_neighborhood=2)
config.environment = env
config.num_episodes_to_run = 10000
config.file_to_save_data_results = "experiments/results"
config.file_to_save_results_graph = "experiments/results"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [512, 512],
        "final_layer_activation": ['tanh', None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [512, 512],
            "final_layer_activation": 'tanh',
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [512, 512],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 10000,
        "batch_size": 256,
        "discount_rate": 0.95,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == "__main__":
    AGENTS = [SAC]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()






