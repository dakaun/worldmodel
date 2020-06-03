import tensorflow as tf
import sys
sys.path.append('../../gym')
from gym.spaces import Box
from gym.envs.classic_control import rendering
from gym.envs.box2d.car_racing import CarRacing
from gym.envs.atari.atari_env import AtariEnv
from PIL import Image
import numpy as np

INPUT_SHAPE=(64,64)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def _process_frame(frame): # converts into (64,64,3)
  img = Image.fromarray(frame)
  img = img.resize(INPUT_SHAPE)  # resize
  obs = np.array(img)
  obs = obs / 255.
  return obs

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

    def _step(self, action):
        obs, reward, done, _ = super(BreakoutWrapper, self)._step(action)
        return _process_frame(obs), reward, done, {}

    def _reset(self,):
        obs = super(BreakoutWrapper,self)._reset()
        return _process_frame(obs)


class CarRacingWrapper(CarRacing):
    def __init__(self, full_episode):
        super(CarRacingWrapper, self).__init__()
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(64,64,3))

    def _step(self, action):
        obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
        return _process_frame(obs), reward, done, {}

def make_env(env_name, rep_act_prob=True, full_episode=False):
    #env = gym.make(env_name)
    if 'Breakout' in env_name:
        game_version = 'v0' if rep_act_prob else 'v4'
        full_game_name = '{}-{}'.format(env_name, game_version)
        env = BreakoutWrapper(env_name, full_game_name)
        return env
    elif 'CarRacing' in env_name:
        env = CarRacingWrapper(full_episode=full_episode)
        return env

if __name__ == '__main__':
    env = make_env('CarRacing')
    print(env)