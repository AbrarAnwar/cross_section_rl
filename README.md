# cross_section_rl

To start, first create data using 

```
python create_simple_data.py
```

Then, train a PointNet model for this simple data using:

```
python model_trainers/point_net_trainer.py
```

We have two approaches to train the model, using a DQN or PPO. The DQN trains faster on one's computer. There are four types of environments:
- simple_cross_section_env: Reward is based on a Delaunay triangulation on a window from the current step
- simple_cross_section_env_weighted: Reward is based on a weighted Delaunay triangulation on a window from the current step. Action space is increased to include weights for the current set of points.
- simple_cross_section_env_local: Reward is based on the Delaunay triangulation of two cross-sections (the current one and previous one). The final mesh is the progressive reconstruction of the faces that were generated
- simple_cross_section_env_local_weighted: Similar to the previous one, but uses a weighted Delaunay triangulation of two cross-sections

To specify an environment, choose one of the following targets (no_weights, weights, local, weights_local). An example of how to run it is:

```
python train_simple_ppo.py --env_type local
```