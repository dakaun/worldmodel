# Explaining Reinforcement Learning through its World Model

This repository is part of the master thesis "Explaining Reinforcement Learning through its World Model" by Daniela Kaun.
It contains a user interface and the underlying logic to explain the behavior of an RL agent with its own world model.
As world model serves the work of [Kaiser et al. (2019)](https://arxiv.org/abs/1903.00374).  
Therefore this repository is split into several directories:
* WorldModelExperiments/dashboarddash - user interface. Here, all code belonging to the user interface is gathered.
* tensor2tensor/tensor2tensor/rl/player.py - world model. Here, all code to train and build the world model of the agent is gathered.
* gym - gym environment. Here, the underlying environment providing the agent and its methods.

#### Gym
The package is from <http://github.com/openai/gym> (VERSION = '0.12.0') and provides the agent-environment ensemble with the standard methods.

#### World Model
The world model applied in this work is from Kaiser et al. (2019). 
The according code is part of a deep learning library from Tensorflow <http://github.com/tensorflow/tensor2tensor> (VERSION='1.13.1').
We use the model-based algorithm and trained the model by ourselves using *tensor2tensor/tensor2tensor/rl/trainer_model_based.py* and a stochastic-discrete model.   
Pretrained models are also available using 
```
gsutil -m cp -r gs://tensor2tensor-checkpoints/modelrl_experiments/train_sd/142/ $OUTPUT_DIR
```
The code underlying of the world model and agent underlying the user interface can be found in *ensor2tensor/tensor2tensor/rl/player.py*
#### User Interface
The user interface is written with [dash plotly](https://dash.plotly.com/) and available in the directory *WorldModelsExperiments/dashboarddash* running 
 ```
 python dashboard.py
```
Within the user interface several modules are available which focus on different sub-topics of the agent or world model. For further information we refer to the master thesis.
