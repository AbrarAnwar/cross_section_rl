# cross_section_rl

To start, first create data using 

```
python create_simple_data.py
```
and specifying the file you want. The three basic shapes are already processed in data/cross_section_data

The primary agent to look at is the one trained by running 
```
python experiments/SAC_trainer_local_weights.py
```

This refers to the use of the simple_cross_section_env_local_weighted.py environment, as it was used in my thesis. The other environments are older and are not up-to-date with the most recent problem formulation

## Notes for others looking at this:
- Any new graph model, or changes to the state occur in the agents/actor_critic_agents/Base_Agents.py folder. Here, the function phi() has all state inputs fed into it before producing the action, and thus "preprocesses" the inputs. Here, you can train a graph neural network model, or any other model that creates a representation for the actor to use. 
- The initial simulations are generated in the environment class, but the number of these are set in agents/actor_critic_agents/SAC.py. It is by default set to 500.
- The current implementation only iterates sequentially through the environment; however, modifications to insteading step_t and step_i could allow for moving around. step_i indexes into the list of cross-sections, while step_t is the number of steps taken so far. Look at simple_cross_section_env_local_jumpy.py for inspiration, even though it might not be identical. 



## Misc

This model also works with the DQN and PPO black boxes provided by the PFRL library. These classes might not work anymore as they have not been tested with the updated problem formulation.

We have two approaches to train the model, using a DQN or PPO. The DQN trains faster on one's computer. There are four types of environments:
- simple_cross_section_env: Reward is based on a Delaunay triangulation on a window from the current step
- simple_cross_section_env_weighted: Reward is based on a weighted Delaunay triangulation on a window from the current step. Action space is increased to include weights for the current set of points.
- simple_cross_section_env_local: Reward is based on the Delaunay triangulation of two cross-sections (the current one and previous one). The final mesh is the progressive reconstruction of the faces that were generated
- simple_cross_section_env_local_weighted: Similar to the previous one, but uses a weighted Delaunay triangulation of two cross-sections

To specify an environment, choose one of the following targets (no_weights, weights, local, weights_local). An example of how to run it is:

```
python train_simple_ppo.py --env_type local
```