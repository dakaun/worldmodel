from gym.envs.atari.atari_env import AtariEnv
import tensorflow as tf
import json
import os
import numpy as np
from gym.spaces.box import Box
from gym.envs.classic_control import rendering


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

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

def make_env(env_name, rep_act_prob=True):
    #env = gym.make(env_name)
    game_version = 'v0' if rep_act_prob else 'v4'
    full_game_name = '{}-{}'.format(env_name, game_version)
    env = BreakoutWrapper(env_name, full_game_name)
    return env
