"""An example of training PPO against OpenAI Gym Atari Envs.

This script is an example of training a PPO agent on Atari envs.

To train PPO for 10M timesteps on Breakout, run:
    python train_ppo_ale.py

To train PPO using a recurrent model on a flickering Atari env, run:
    python train_ppo_ale.py --recurrent --flicker --no-frame-stack
"""
import argparse

import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import atari_wrappers

from envs.simple_cross_section_env import SimpleCrossSectionEnv
from models.point_net_ae import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="simple", help="type of env"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID. Set to -1 to use CPUs only."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of env instances run in parallel.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default='/home/abrar/thesis/cross_sections_rl/data/cross_section_data/sphere_resampled.npz',
        help="directory of file used"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=51*100, help="Total time steps for training."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes ran in an evaluation phase.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=128 * 8,
        help="Interval (in timesteps) between PPO iterations.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32 * 8,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs used for each PPO iteration.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Interval (in timesteps) of printing logs.",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        default=False,
        help="Use a recurrent model. See the code for the model definition.",
    )
    parser.add_argument(
        "--flicker",
        action="store_true",
        default=False,
        help=(
            "Use so-called flickering Atari, where each"
            " screen is blacked out with probability 0.5."
        ),
    )

    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )
    parser.add_argument(
        "--pointnet_load_path",
        type=str,
        default="saved_models/sphere/save_999.pth",
        help=(
            "Directory path to load a saved pointnet network from"
        ),
    )

    
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    pfrl.utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed

        env = SimpleCrossSectionEnv(args.input_file, same_obs_size=True)

        env.seed(env_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        print([(idx,env) for idx, env in enumerate(range(args.num_envs))] )
        vec_env = pfrl.envs.MultiprocessVectorEnv(
            [
                (lambda: make_env(idx, test))
                for idx, env in enumerate(range(args.num_envs))
            ]
        )
        return vec_env

    sample_env = make_batch_env(test=False)
    print("Observation space", sample_env.observation_space)
    print("Action space", sample_env.action_space)

    test_env = SimpleCrossSectionEnv(args.input_file, same_obs_size=True)
    sample_spacing = test_env.sample_spacing
    print('First Quality Metric: ', test_env.calculate_M_reward())
    del test_env

    # n_actions = sample_env.action_space.n
    action_space = sample_env.action_space
    obs_n_channels = sample_env.observation_space.low.shape[0]
    print(obs_n_channels)
    print(action_space.low.size)
    obs = np.array(sample_env.reset())

    print(obs.shape)
    num_points = obs.shape[1] * obs.shape[2]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    autoencoder = PCAutoEncoder(3, num_points)
    state_dict = torch.load(args.pointnet_load_path, map_location=device)
    autoencoder.load_state_dict(state_dict)

    autoencoder = autoencoder.eval()
    del sample_env


    model = RLModel(1024, action_space.low.size)

    # agent model structure:
    #   state input: list of cross sections
    #   get representation E from cross sections from PointNet
    #   feed it into TD-VAE and get full state representation E'
    #   push through linear layer to get action output
    #   next_state, _ = env.step(action) 

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)



    # this is the preprocessing into pointnet!
    def phi(x):
        # turn into points
        pts = np.empty(shape = (0,3))
        for i,m in enumerate(x):
            col_to_add = np.ones(len(m))*i*sample_spacing
            res = np.hstack([m, np.atleast_2d(col_to_add).T])
            pts = np.concatenate([pts, res])

        pts = torch.tensor(pts)
        pts = pts.unsqueeze(0)
        pts = pts.transpose(2, 1).float()
        # now feed these into pointnet
        reconstructed_points, global_feat = autoencoder(pts)
        return global_feat.squeeze()

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=args.recurrent,
        max_grad_norm=0.5,
    )
    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        )

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
        )


class RLModel(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.l1 = torch.nn.Linear(obs_size, 512)
        self.l2 = torch.nn.Linear(512, 512)
        self.l3 = torch.nn.Linear(512, n_actions)

        self.value = torch.nn.Linear(512, 1)

        self.pol = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=self.n_actions,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))

        value = self.value(h)

        h = self.l3(h)
        # ret = pfrl.action_value.DiscreteActionValue(h)
        ret = self.pol(h)
        return ret, value

if __name__ == "__main__":
    main()