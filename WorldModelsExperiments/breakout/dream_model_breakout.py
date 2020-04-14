import sys
import numpy as np
import random
from copy import deepcopy
from PIL import Image
import gym
import json
import argparse

from WorldModelsExperiments.breakout.vae.vae import ConvVAE
from WorldModelsExperiments.breakout.rnn.rnn import RNNModel, hps_sample, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
from WorldModelsExperiments.breakout.dream_env_breakout import make_env

INPUT_SHAPE = (64,64)

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x), axis=0)

def _process_frame(frame): # converts into (64,64,3)
  img = Image.fromarray(frame)
  img = img.resize(INPUT_SHAPE)  # resize
  obs = np.array(img)
  obs = obs / 255.
  return obs

def make_model(load_model, rnn_path, vae_path, frame_limit):
    model = DreamModel(load_model, rnn_path, vae_path, frame_limit)
    return model

class DreamModel():
    def __init__(self, load_model=True,   rnn_path='tf_rnn/rnn.json', vae_path='tf_vae/vae.json', frame_limit=1200):
        self.env_name = 'Breakout-v0'

        self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
        self.rnn = RNNModel(hps_sample, gpu_mode=False, reuse=True)

        if load_model:
            self.vae.load_json(vae_path)  # vae_path)
            self.rnn.load_json(rnn_path)

        self.frame_limit=frame_limit
        self._make_env()

        self.state = rnn_init_state(self.rnn)
        self.rnn_mode = True

        self.input_size = rnn_output_size()
        self.z_size = 32

        self.render_mode = False

    def _make_env(self, seed=-1, render_mode =False):
        self.render_mode = render_mode
        self.env = make_env(self.env_name, agent = self, frame_limit= self.frame_limit, seed = seed)

    def reset(self):
        self.state = rnn_init_state(self.rnn)

    def encode_obs(self, obs):
        result = np.expand_dims(obs, axis=0)
        mu, logvar = self.vae.encode_mu_logvar(result)
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        z = mu + np.exp(logvar / 2.0) * np.random.randn(*s)
        return z, mu, logvar

    def get_action(self, z):
        h = rnn_output(self.state, z)
        action = softmax(np.matmul(h, self.weight) + self.bias)
        action = np.argmax(action)
        action_one_hot = np.zeros(self.rnn.hps.num_actions)
        action_one_hot[action] = 1
        self.state = rnn_next_state(self.rnn, z, action_one_hot, self.state)
        return action_one_hot, action

    def set_model_params(self, model_params):
        self.bias = np.array(model_params[:4])
        self.weight = np.array(model_params[4:]).reshape(self.input_size, 4)

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        # return np.random.randn(self.param_count)*stdev
        return np.random.standard_cauchy(self.param_count) * stdev  # spice things up!

    def init_random_model_params(self ,stdev=0.1):
        rnn_params = self.rnn.get_random_model_params(stdev=stdev)
        self.rnn.set_model_params(rnn_params)


def simulate(model, train_mode=False, render_mode=False, num_episode=1, seed=-1, max_len=-1):
    reward_list = []
    t_list= []
    max_episode_length = 2100

    if max_len>0:
        max_episode_length = max_len

    if (seed >= 0):
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    for episode in range(num_episode):
        model.reset()
        z = model.env.reset()

        for t in range(max_episode_length):
            action_one_hot, action = model.get_action(z)

            if render_mode:
                model.env.render('human')
            else:
                model.env.render('rgb_array')


            z, reward, done, info = model.env.step(action_one_hot) # two problems here: where to get reward and done info ?

            print('Action ', action, ' sum.square.z ', np.sum(np.square(z)))

            if done:
                break
        t_list.append(t)

    return reward_list, t_list


def main():
    use_model = False

    #parser = argparse.ArgumentParse(description='Run Dream World with given trained models')
    #parser.add_argument('-f', '--file', type=str, help='path to best json file') # file: log/carracing.cma.16.64.best.json
    #parser.add_argument('--vae', type=str, help='path to vae model')
    #parser.add_argument('--rnn', type=str, help='path to rnn model')
    #parser.add_argument('--render', type=bool, help='Boolean to show images')
    #args = parser.parse_args()
    #vae_path = args.vae
    #rnn_path = args.rnn

    render_mode = True #args.render

    rnn_path='/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v2/200228/tf_rnn_10000/rnn.json'
    vae_path='/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v2/200228/tf_vae/vae.json'

    file = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v2/200228/log/breakout.cma.16.64.best.json'

    if file:#args.file:
        use_model = True
        filename = file#args.file
        print('filename: ', filename)

    model = make_model(load_model=use_model, rnn_path=rnn_path, vae_path=vae_path, frame_limit=500)

    if use_model:
        model.load_model(filename)
    else:
        model.init_random_model_params(stdev=np.random.rand()*0.01)

    while True:
        reward, steps_taken = simulate(model, train_mode=False, render_mode=render_mode, num_episode=1)
        print('terminal reward: ', reward, ' ,average steps taken: ', np.mean(steps_taken)+1)

if __name__ == "__main__":
    main() # run: python dream_model_breakout.py path_to_file.json