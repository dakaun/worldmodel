import numpy as np
import random
import json
import sys
import config
import time
import argparse
from WorldModelsExperiments.breakout.rnn.rnn import hps_sample, RNNModel, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
from WorldModelsExperiments.breakout.vae.vae import ConvVAE
from WorldModelsExperiments.breakout.env import make_env
from copy import deepcopy
from PIL import Image
from pyglet.window import key
import pyglet
import multiprocessing

RENDER_DELAY = True

INPUT_SHAPE = (64,64)

def _process_frame(frame): # converts into (64,64,3)
  img = Image.fromarray(frame)
  img = img.resize(INPUT_SHAPE)  # resize
  obs = np.array(img)
  obs = obs / 255.
  return obs

def make_model(env_name, load_model=True, rnn_path='tf_rnn/rnn.json', vae_path='tf_vae/vae.json',kl_tolerance=0.5, z_size=32):
  # can be extended in the future.
  model = Model(env_name, load_model=load_model, rnn_path=rnn_path, vae_path=vae_path, kl_tolerance=kl_tolerance,z_size=z_size)
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def passthru(x):
  return x

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x,lo),hi)

class Model():
  ''' simple feedforward model '''
  def __init__(self, env_name, load_model=True, rnn_path='tf_rnn/rnn.json', vae_path='tf_vae/vae.json', kl_tolerance=0.5, z_size=32):
    self.env_name = env_name
    self._make_env()

    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True, kl_tolerance=kl_tolerance, z_size=z_size)

    #hps_sample_dynamic = hps_sample._replace(num_actions=self.num_actions)

    self.rnn = RNNModel(hps_sample, env_name, gpu_mode=False, reuse=True, z_size=z_size)

    if load_model:
      self.vae.load_json(vae_path)  # vae_path)
      self.rnn.load_json(rnn_path)

    self.state = rnn_init_state(self.rnn)
    self.rnn_mode = True

    self.input_size = rnn_output_size(self.rnn)  # concatenate z,h - output of rnn (288)=z+ self.state.h[0]
    self.z_size = z_size

    if 'Breakout' in env_name:
      self.num_actions = self.env.action_space.n
      self.weight = np.random.randn(self.input_size, self.env.action_space.n)
      self.bias = np.random.randn(self.env.action_space.n)
      self.param_count = (self.input_size)*self.env.action_space.n+self.env.action_space.n
    elif 'CarRacing' in env_name:
      self.num_actions = self.env.action_space.shape[0]
      self.weight = np.random.randn(self.input_size, 3)
      self.bias = np.random.randn(3)
      self.param_count = (self.input_size) * 3 + 3

    self.render_mode = False

  # def _make_env(self):
  #   self.env = gym.make(self.env_name)
  #   np.random.seed(123)
  #   self.env.seed(123)
  #   self.num_actions = self.env.action_space.n

  def _make_env(self):
    self.env = make_env(self.env_name)
    np.random.seed(123)
    self.env.seed(123)
    #self.num_actions = self.env.action_space.n
    #self.render_mode = render_mode
    #self.env = make_env(self.env_name, seed=seed, render_mode=render_mode, load_model=load_model)

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    #result = np.copy(obs).astype(np.float)/255.
    result = np.expand_dims(obs, axis=0)#reshape(1, 64, 64, 3)
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0] #
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, mu, logvar

  def get_action(self, z):
    h = rnn_output(self.state, z)  # np.concatenate([z, state.h[0]])
    if 'Breakout' in self.env_name:
      # could probabilistically sample from softmax, but greedy
      action = softmax(np.matmul(h, self.weight) + self.bias)
      action = np.argmax(action)
      #print("Action sampled from VAE:", action)
      action_one_hot = np.zeros(self.num_actions)
      action_one_hot[action] = 1
      #print("Action hot:", action_one_hot)
      self.state = rnn_next_state(self.rnn, z, action_one_hot, self.state, self.env_name)
      return action, action_one_hot
    elif 'CarRacing' in self.env_name:
      action = np.tanh(np.dot(h, self.weight)+ self.bias)
      action[1] = (action[1]+1.0)/2.0
      action[2] = clip(action[2])
      self.state= rnn_next_state(self.rnn, z, action, self.state, self.env_name)
      return action, ''

  def set_model_params(self, model_params):
    if 'Breakout' in self.env_name:
      self.bias = np.array(model_params[:4])
      self.weight = np.array(model_params[4:]).reshape(self.input_size, 4)
    elif 'CarRacing' in self.env_name:
      self.bias = np.array(model_params[:3])
      self.weight = np.array(model_params[3:]).reshape(self.input_size,3)

  def load_model(self, filename):
    with open(filename) as f:
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up!

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    vae_params = self.vae.get_random_model_params(stdev=stdev)
    self.vae.set_model_params(vae_params)
    rnn_params = self.rnn.get_random_model_params(stdev=stdev)
    self.rnn.set_model_params(rnn_params)

  def __deepcopy__(self):
    return Model(deepcopy())
  # https://stackoverflow.com/questions/50352273/how-to-exclude-specific-references-from-deepcopy
  # https://stackoverflow.com/questions/23580974/copy-all-but-one-part-of-an-object-in-python
  # https://stackoverflow.com/questions/4794244/how-can-i-create-a-copy-of-an-object-in-python
  


def evaluate(model):
  # run 100 times and average score, according to the reles.
  model.env.seed(0)
  total_reward = 0.0
  N = 100
  for i in range(N):
    reward, t = simulate(model, train_mode=False, render_mode=False, num_episode=1)
    total_reward += reward[0]
  return (total_reward / float(N))

def key_press(symbol, mod):
  global human_sets_pause, key_to_action
  print('pressed key')
  if symbol == key.LEFT:
    human_sets_pause = not human_sets_pause
    key_to_action = 3
  elif symbol == key.RIGHT:
    human_sets_pause = not human_sets_pause
    key_to_action = 2

def run_in_env(obs, model, treward, done=False):
  while not done:
    model.env.viewer.imshow(obs)
    obs = _process_frame(obs)
    z, mu, logvar = model.encode_obs(obs)
    action,_ = model.get_action(z)
    obs, reward, done, info = model.env.step(action)
    treward += reward
  return treward

def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):
  global human_sets_pause, key_to_action
  reward_list = []
  t_list = []
  max_episode_length = 2100
  human_sets_pause = False

  action_list_episode = []
  observation_list_episode = []

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):
    action_list = []
    observation_list = []
    obs = model.env.reset()
    if obs is None:
      obs = deepcopy(model.env.reset())
    #obs = _process_frame(obs)

    t=0
    total_reward = 0.0
    done = False
    #prev_info = {"ale.lives": model.env.ale.lives()}

    while not done:
      if render_mode:
        model.env.render("human")
        model.env.unwrapped.viewer.window.on_key_press = key_press
        if RENDER_DELAY:
          time.sleep(0.01)
      else:
        model.env.render('rgb_array')
      obs = _process_frame(obs)
      z, mu, logvar = model.encode_obs(obs)
      action,_ = model.get_action(z) #action_one_hot, action
      obs, reward, done, info = model.env.step(action)

      # if prev_info['ale.lives']>info['ale.lives']:
      #   model.env.step(1)

      #todo carracing: penalize for turning to frequently ?

      #prev_info = info

      action_list.append(action)
      observation_list.append(obs)
      #obs = _process_frame(obs)
      total_reward += reward
      t += 1
      if t>1000:
        break


      if done:
        if render_mode:
          model.env.close()
        action_list_episode.append(action_list)
        observation_list_episode.append(observation_list)
        break
      # if t ==600:
      #   human_sets_pause = True
      #   key_to_action = 2
      while human_sets_pause:# and t ==25:
        print('while true')
        print('key to action: ', key_to_action)
        print('Achieved Reward before shift ', total_reward)
        time.sleep(2)
        for i in range(20):
          a = key_to_action
          img, _, _, _ = model.env.step(a) # todo not only step, also update rnn state ?
        print('render for several steps done, shift to ', key_to_action)
        time.sleep(2)
        reward = run_in_env(img, model, total_reward, done)
        print('Achieved Reward: ', reward)
        time.sleep(2)
        human_sets_pause = False
        model.env.close()

    if render_mode:
      print("reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)
  return reward_list, t_list, action_list_episode, np.array(observation_list_episode)

def main():

  global RENDER_DELAY
  global final_mode

  use_model = False

  parser = argparse.ArgumentParser(description='Run Breakout with all given models')
  # parser.add_argument('-n', '--ename', type=str, default='Breakout', help='Env Name')
  # parser.add_argument('-f', '--file', type=str, help='path to best json file') # file: log/carracing.cma.16.64.best.json
  # parser.add_argument('--vae', type=str, help='path to vae model')
  # parser.add_argument('--rnn', type=str, help='path to rnn model')
  # parser.add_argument('--render', type=bool, default=True, help='Boolean to show images')
  # args = parser.parse_args()
  # vae_path = args.vae
  # rnn_path = args.rnn

  render_mode = True  # args.render
  env_name='CarRacing'
  rnn_path = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v3/200522/tf_rnn/rnn.json'
  vae_path = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v3/200522/tf_vae/vae.json'

  file = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v3/200522/log/carracing.cma.16.64.best.json'

  if file: #args.file
    use_model = True
    filename = file
    print("filename", filename)

  model = make_model(env_name, rnn_path=rnn_path, vae_path=vae_path)
  print('model size', model.param_count)

  if (use_model):
    model.load_model(filename)
  else:
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  N_episode = 1
  reward_list = []
  human_sets_pause = False

  for i in range(N_episode):
    reward, steps_taken, action_list, obs_list = simulate(model,
      train_mode=False, render_mode=render_mode, num_episode=100)
    reward_list.append(reward[0])
    time.sleep(5)

  print("average_reward", np.mean(reward_list), "stdev", np.std(reward_list), "average steps taken", np.mean(steps_taken))

if __name__ == "__main__":
  main()
