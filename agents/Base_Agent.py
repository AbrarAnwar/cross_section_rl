import logging
import os
import sys
import gym
import random
import numpy as np
import torch
import time
# import tensorflow as tf
from nn_builder.pytorch.NN import NN
# from tensorboardX import SummaryWriter
from torch.optim import optimizer

from models.point_net_ae import *
from models.gnn import SpaceTimeElementEncoder
if torch.cuda.is_available():
    from losses.chamfer_distance_gpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
else:
    from losses.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
chamfer_dist = ChamferDistance()

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

class Base_Agent(object):

    def __init__(self, config):
        self.environment = config.environment
        self.environment_title = self.get_environment_title()
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"

        obs =  self.environment.reset()
        self.state_size = obs[0].shape[0]*obs[0].shape[1]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # autoencoder = PCAutoEncoder(3, self.state_size, 256)
        # state_dict = torch.load('/home/abrar/cross_section_rl/saved_models/list_model/save_1999.pth', map_location=self.device)
        # autoencoder.load_state_dict(state_dict)
        

        # self.autoencoder = autoencoder.to(self.device)

        self.state_size = 256

        self.edgeEncoder = SpaceTimeElementEncoder(3, self.state_size, hidden=128).to(self.device)

        # self.autoencoder = autoencoder.eval()
        # self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.0001, betas=(0.9, 0.999))


        # self.state_size = 256 + self.environment.num_points*2

        self.logger = self.setup_logger()
        self.debug_mode = config.debug_mode
        # if self.debug_mode: self.tensorboard = SummaryWriter()
        self.config = config
        self.set_random_seeds(config.seed)

        self.action_size = int(self.get_action_size())

        self.config.action_size = self.action_size

        self.lowest_possible_episode_score = self.get_lowest_possible_episode_score()





        self.hyperparameters = config.hyperparameters
        self.average_score_required_to_win = self.get_score_required_to_win()*len(self.environment.M)
        self.rolling_score_window = self.get_trials()
        # self.max_steps_per_episode = self.environment.spec.max_episode_steps
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning
        self.log_game_info()

    def phi(self, x, train):
        # print("IN PHI")
        # this comes in a list of batches. 
        x_list = np.array(x,dtype=object)

        data_list = []

        all_pts = []
        for vals in x_list:
            pts = np.empty(shape = (0,3))
            for i,m in enumerate(vals[0]):
                col_to_add = np.ones(len(m))*i*self.environment.sample_spacing
                res = np.hstack([m, np.atleast_2d(col_to_add).T])
                pts = np.concatenate([pts, res])
            all_pts.append(pts)
            edges = vals[1].T
            data_list.append(Data(x=torch.tensor(pts, dtype=torch.float).to(self.device), edge_index=torch.tensor(edges, dtype=torch.long).to(self.device)))


        pts = np.array(all_pts)
        batch = Batch.from_data_list(data_list)

        if(train == True):
            # self.autoencoder.train()
            self.edgeEncoder.train()
        else:
            # self.autoencoder.eval()
            self.edgeEncoder.eval()

        # global_feat = None
        # pts = torch.tensor(pts)
        # # pts = pts.unsqueeze(0)
        # pts = pts.transpose(2, 1).float().to(self.device)
        # # now feed these into pointnet
        # reconstructed_points, global_feat = self.autoencoder(pts)
        # dist1, dist2 = chamfer_dist(pts, reconstructed_points)  


        # calculate graph shit
        graphFeat = self.edgeEncoder(batch, train)

        return graphFeat
        # return graphFeat
        # # exit()
        # # feat = global_feat.squeeze()
        # global_feat = global_feat
        # feat = global_feat
        # feat = torch.cat([feat, v_d_1, v_d_2], dim=1)
        # # print('feature shape', feat.shape)
        # feat = feat.squeeze()

        # return feat

        # self.ae_optimizer.zero_grad()
        # if(len(self.phi_buffer) > 1000):
        #     self.phi_buffer.pop(random.randrange(len(self.phi_buffer)))
        # self.phi_buffer.append(pts)

        # if not len(self.phi_buffer) <= 4:
        #     training = random.sample(self.phi_buffer, 3)
        # else:
        #     training = self.phi_buffer

        # training.append(pts)

        # # self.autoencoder.train()
        # training = np.stack(training)
        # train_points = torch.tensor(training).to(self.device).float()
        # train_points = train_points.transpose(2, 1)

        # reconstructed_points, global_feat = self.autoencoder(train_points)

        # dist1, dist2 = chamfer_dist(train_points, reconstructed_points)   # calculate loss
        # ae_train_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        # ae_train_loss.backward()
        # self.ae_optimizer.step()

        # self.autoencoder.eval()

        # global_feat = None
        # # with torch.no_grad(): 
        # pts = torch.tensor(pts)
        # pts = pts.unsqueeze(0)
        # pts = pts.transpose(2, 1).float().to(self.device)
        # # now feed these into pointnet
        # reconstructed_points, global_feat = self.autoencoder(pts)
        # dist1, dist2 = chamfer_dist(pts, reconstructed_points)   # calculate loss
        # self.ae_count +=1

        # feat = global_feat.squeeze().cpu().detach().numpy()
        # feat = np.append(feat, v_d_1)
        # feat = np.append(feat, v_d_2)

        # exit()
        # return feat



    def step(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def get_environment_title(self):
        """Extracts name of environment from it"""
        try:
            name = self.environment.unwrapped.id
        except AttributeError:
            try:
                if str(self.environment.unwrapped)[1:11] == "FetchReach": return "FetchReach"
                elif str(self.environment.unwrapped)[1:8] == "AntMaze": return "AntMaze"
                elif str(self.environment.unwrapped)[1:7] == "Hopper": return "Hopper"
                elif str(self.environment.unwrapped)[1:9] == "Walker2d": return "Walker2d"
                else:
                    name = self.environment.spec.id.split("-")[0]
            except AttributeError:
                name = str(self.environment.env)
                if name[0:10] == "TimeLimit<": name = name[10:]
                name = name.split(" ")[0]
                if name[0] == "<": name = name[1:]
                if name[-3:] == "Env": name = name[:-3]
        return name

    def get_lowest_possible_episode_score(self):
        """Returns the lowest possible episode score you can get in an environment"""
        if self.environment_title == "Taxi": return -800
        return None

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        if "overwrite_action_size" in self.config.__dict__: return self.config.overwrite_action_size
        if "action_size" in self.environment.__dict__: return self.environment.action_size
        if self.action_types == "DISCRETE": return self.environment.action_space.n
        else: return self.environment.action_space.shape[0]

    def get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
        # random_state = self.phi(self.environment.reset())
        random_state = self.environment.reset()

        if isinstance(random_state, dict):
            state_size = random_state["observation"].shape[0] + random_state["desired_goal"].shape[0]
            return state_size
        else:
            return random_state.size

    def get_score_required_to_win(self):
        """Gets average score required to win game"""
        print("TITLE ", self.environment_title)
        if self.environment_title == "FetchReach": return -5
        if self.environment_title in ["AntMaze", "Hopper", "Walker2d"]:
            print("Score required to win set to infinity therefore no learning rate annealing will happen")
            return float("inf")
        try: return self.environment.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.environment.spec.reward_threshold
            except AttributeError:
                return self.environment.unwrapped.spec.reward_threshold

    def get_trials(self):
        """Gets the number of trials to average a score over"""
        if self.environment_title in ["AntMaze", "FetchReach", "Hopper", "Walker2d", "CartPole"]: return 100
        try: return self.environment.unwrapped.trials
        except AttributeError: return self.environment.spec.trials

    def setup_logger(self):
        """Sets up the logger"""
        filename = "Training.log"
        try: 
            if os.path.isfile(filename): 
                os.remove(filename)
        except: pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    def log_game_info(self):
        """Logs info relating to the game"""
        for ix, param in enumerate([self.environment_title, self.action_types, self.action_size, self.lowest_possible_episode_score,
                      self.state_size, self.hyperparameters, self.average_score_required_to_win, self.rolling_score_window,
                      self.device]):
            self.logger.info("{} -- {}".format(ix, param))

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def reset_game(self, is_eval=False):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config.seed)
        self.state = self.environment.reset(is_eval)
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.state))

    def track_episodes_data(self):
        """Saves the data from the recent episodes"""
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        eval_ep = False
        while self.episode_number < num_episodes:
            self.reset_game(is_eval=eval_ep)
            self.step()
            eval_ep = self.episode_number % 10-1 == 0 
            if eval_ep: print("THIS IS THE EVAL ITERATION")
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        # self.next_state = self.phi(self.next_state)
        self.next_state = self.next_state

        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]: self.reward =  max(min(self.reward, 1.0), -1.0)


    def save_and_print_result(self):
        """Saves and prints results of the game"""
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        """Prints out the latest episode results"""
        text = """"\r Episode {0}, Score: {3: .2f}, Max score seen: {4: .2f}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}\n"""
        sys.stdout.write(text.format(len(self.game_full_episode_scores), self.rolling_results[-1], self.max_rolling_score_seen,
                                     self.game_full_episode_scores[-1], self.max_episode_score_seen))
        sys.stdout.flush()

    def show_whether_achieved_goal(self):
        """Prints out whether the agent achieved the environment target goal"""
        index_achieved_goal = self.achieved_required_score_at_index()
        print(" ")
        if index_achieved_goal == -1: #this means agent never achieved goal
            print("\033[91m" + "\033[1m" +
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" +
                  "{} achieved required score at episode {} \n".format(self.agent_name, index_achieved_goal) +
                  "\033[0m" + "\033[0m")

    def achieved_required_score_at_index(self):
        """Returns the episode at which agent achieved goal or -1 if it never achieved it"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required_to_win:
                return ix
        return -1

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        if random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=True):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        # loss += self.ae_train_loss/self.ae_count
        # self.ae_count = 0
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def log_gradient_and_weight_information(self, network, optimizer):

        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.logger.info("Gradient Norm {}".format(total_norm))

        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        self.logger.info("Learning Rate {}".format(learning_rate))


    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]
        if override_seed: seed = override_seed
        else: seed = self.config.seed

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    def freeze_all_but_output_layers(self, network):
        """Freezes all layers except the output layer of a network"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """Unfreezes all layers of a network"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
