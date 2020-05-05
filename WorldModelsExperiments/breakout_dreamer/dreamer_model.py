import numpy as np
import random
import json
import sys
import config
import time
import tensorflow as tf
import os
from PIL import Image
from gym.envs.atari.atari_env import AtariEnv
from gym.spaces.box import Box
from gym.utils import seeding
from gym.envs.classic_control import rendering
from WorldModelsExperiments.breakout_dreamer.dreamer_vae.dreamer_vae import ConvVAE
from WorldModelsExperiments.breakout_dreamer.dreamer_rnn.dreamer_rnn import RNNModel, default_hps

final_mode = True
render_mode = True
RENDER_DELAY = False
TEMPERATURE = 1.25

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  random_value = np.random.randint(N)
  #print('error with sampling ensemble, returning random', random_value)
  return random_value

class BreakoutWrapper(AtariEnv):
    metadata = {
        'render.modes':['human', 'rgb_array']
    }
    def __init__(self, game_name, fullgame_name):
        frameskip = 1 if 'Frameskip' in fullgame_name else (2,5)
        game_name = game_name.lower()
        #super(BreakoutWrapper, self).__init__() # pong env, nicht breakout..
        super().__init__(game=game_name, obs_type='image', frameskip=frameskip)
        self.observation_space = Box(low=0, high=255., shape=(64,64,3))
        self.viewer = rendering.SimpleImageViewer()

        with open(os.path.join('tf_dreamerinitial_z', 'initial_z.json'), 'r') as f:
            [initial_mu, initial_logvar] = json.load(f)

        self.initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]

        self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
        self.rnn = RNNModel(hps=hps_sample, gpu_mode=False)

        if load_model:
            self.vae.load_json(os.path.join('tf_vaedream', 'vae.json'))
            self.rnn.load_json(os.path.join('tf_rnndream', 'rnn.json'))

        self.outwidth = self.rnn.hps.seq_width
        self.zero_state = self.rnn.sess.rn(self.rnn.zero_state)
        self.seed()
        self.rnn_state = None
        self.z = None
        self.restart = None
        self.temperature = None
        self.frame_count = None
        self.max_frame = 2100

        self.reset()

    def sample_init_z(self):
        idx = self.np_random.randint(0,len(self.initial_mu_logvar))
        init_mu, init_logvar = self.initial_mu_logvar[idx]
        init_mu = np.array(init_mu) / 10000.
        init_logvar = np.array(init_logvar) / 10000.
        init_z = init_mu + np.exp(init_logvar / 2.0) * self.np_random.randn(*init_logvar.shape)
        return init_z

    def _current_state(self):
        return np.concatenate([self.z, self.rnn_state.h.flatten()], axis=0) # todo h oder h[0] wie in "normal"

    def _reset(self):
        self.temperature = TEMPERATURE
        self.rnn_state = self.zero_state
        self.z = self.sample_init_z()
        self.restart = 1
        self.frame_count = 0
        return self._current_state()

    def _seed(self, seed=None):
        if seed:
            tf.set_random_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, a):
        self.frame_count += 1

        prev_z = self.zeros((1,1,64))
        prev_z[0][0] = self.z
        prev_action = np.zeros((1,1,4))
        prev_action[0][0] = a
        prev_restart = np.ones((1,1))
        prev_restart[0], self.restart

        s_model = self.rnn
        temperature = self.temperature

        feed = {s_model.input_z: prev_z,
                s_model.input_action: prev_action,
                s_model.input_restart: prev_restart,
                s_model.initial_state: self.rnn_state
                }
        [logmix, mean, logstd, logrestart, next_state] = s_model.sess.run([s_model.out_logmix,
                                                                           s_model.out_mean,
                                                                           s_model.out_logstd,
                                                                           s_model.out_restart_logits,
                                                                           s_model.final_state],
                                                                          feed)
        OUTWIDTH = self.outwidth
        # adjust temperatures
        logmix2 = np.copy(logmix) / temperature
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)
        mixture_idx = np.zeros(OUTWIDTH)
        chosen_mean = np.zeros(OUTWIDTH)
        chosen_logstd = np.zeros(OUTWIDTH)

        for j in range(OUTWIDTH):
            idx = get_pi_idx(self.np_random.rand(), logmix2[j])
            mixture_idx[j] = idx
            chosen_mean[j] = mean[j][idx]
            chosen_logstd[j] = logstd[j][idx]

        rand_gaussian = self.np_random.randn(OUTWIDTH) * np.sqrt(temperature)
        next_z = chosen_mean + np.exp(chosen_logstd) * rand_gaussian

        next_restart = 0
        done = False
        if (logrestart[0] > 0):
            next_restart = 1
            done = True

        self.z = next_z
        self.restart = next_restart
        self.rnn_state = next_state

        reward = 1  # always return a reward of one if still alive.

        if self.frame_count >= self.max_frame:
            done = True

        return self._current_state(), reward, done, {}

    def _get_image(self, upsize=False):
        # decode the latent vector
        img = self.vae.decode(self.z.reshape(1, 64)) * 255.
        img = np.round(img).astype(np.uint8)
        img = img.reshape(64, 64, 3)
        if upsize:
            img = Image.fromarray(img).resize((640,640))
        return img

    def _render(self, mode='human', close=False):
        if not self.render_mode:
            return

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'rgb_array':
            img = self._get_image(upsize=True)
            return img

        elif mode == 'human':
            img = self._get_image(upsize=True)
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)




def make_env(env_name, rep_act_prob=True):
    game_version = 'v0' if rep_act_prob else 'v4'
    full_game_name = '{}-{}'.format(env_name, game_version)
    env = BreakoutWrapper(env_name, full_game_name)
    return env


def make_model(load_model=True, rnn_path='tf_rnn/rnn.json', vae_path='tf_vae/vae.json'):
  # can be extended in the future.
  model = Model(load_model=load_model, rnn_path=rnn_path, vae_path=vae_path)
  return model

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

class Model():
    def __init__(self) :
        self.env_name = 'Breakout'
        self._make_env()

        self.noise_level = 0.
        self.input_size = (64+512)# z + h = 64+512
        self.output_size = 5 # done, actions ?
        self.shapes = [(self.input_size, self.output_size)]

        self.sample_output = False
        self.activation = [softmax]

        self.weight = []
        self.param_count = self.input_size * self.output_size

        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))

        self.render_mode = False

    def _make_env(self):
        self.env = make_env(self.env_name)
        np.random.seed(123)
        self.env.seed(123)
        self.num_actions = self.env.action_space.n

    def get_action(self, actual_state):
        actual_state = np.array(actual_state).flatten() # np.concatenate([self.z, self.rnn_state.h.flatten()], axis=0)
        action = softmax(np.matmul(actual_state,self.weight))
        action = np.argmax(action)
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1
        # todo state update ? self.state = rnn_next_state(self.rnn, z, action_one_hot, self.state)
        return action_one_hot, action

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            s_w = np.product(w_shape)
            s = s_w
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)
        # also load the vae and rnn
        self.env.vae.load_json('tf_vaedream/vae.json')
        self.env.rnn.load_json('tf_rnndream/rnn.json')

    def get_random_model_params(self, stdev=0.1):
        # return np.random.randn(self.param_count)*stdev
        return np.random.standard_cauchy(self.param_count) * stdev  # spice things up!

    def init_random_model_params(self, stdev=0.1):
        params = self.get_random_model_params(stdev=stdev)
        self.set_model_params(params)
        vae_params = self.env.vae.get_random_model_params(stdev=stdev)
        self.env.vae.set_model_params(vae_params)
        rnn_params = self.env.rnn.get_random_model_params(stdev=stdev)
        self.env.rnn.set_model_params(rnn_params)

def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):
    reward_list = []
    t_list = []
    total_reward = 0
    max_episode_length = 2100

    if train_mode and max_len > 0:
        max_episode_length = max_len

    if (seed >= 0):
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)
    for t in range(max_episode_length):

        if render_mode:
            model.env.render("human")
            if RENDER_DELAY:
                time.sleep(0.01)

        action = model.get_action(obs)
        prev_obs = obs
        obs, reward, done, info = model.env.step(action)
        total_reward += reward

        if done:
            break

    if render_mode:
        print("reward", total_reward, "timesteps", t)
        model.env.close()

    reward_list.append(total_reward)
    t_list.append(t)


    return reward_list, t_list

def main():
    reward, steps_taken = simulate(model,
                                   train_mode=False, render_mode=render_mode, num_episode=1)
    print("terminal reward", reward, "average steps taken", np.mean(steps_taken) + 1)

if __name__ == "__main__":
  main()