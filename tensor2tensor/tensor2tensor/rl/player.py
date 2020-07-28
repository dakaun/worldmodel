# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Play with a world model.

Controls:
  WSAD and SPACE to control the agent.
  R key to reset env.
  C key to toggle WAIT mode.
  N to perform NOOP action under WAIT mode.
  X to reset simulated env only, when running sim-real comparison.

Run this script with the same parameters as trainer_model_based.py. Note that
values of most of them have no effect on player, so running just

python -m tensor2tensor/rl/player.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base

might work for you.

More advanced example:

python -m tensor2tensor/rl/record_ppo.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base \
    --sim_and_real=False \
    --simulated_env=False \
    --loop_hparams=generative_model="next_frame" \
    --video_dir=my/video/dir \
    --zoom=6 \
    --fps=50 \
    --env=real \
    --epoch=-1

Check flags definitions under imports for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../../../gym')
import gym
from gym.utils import play
import numpy as np

sys.path.append('../../../tensor2tensor')
import tensor2tensor
import pygame
import time
#from tensor2tensor.rl.evaluator import make_agent_from_hparams
from tensor2tensor.rl import rl_utils
from tensor2tensor.models.research import rl
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils.hparam import HParams

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.rl import player_utils
from tensor2tensor.rl.envs.simulated_batch_env import PIL_Image
from tensor2tensor.rl.envs.simulated_batch_env import PIL_ImageDraw
from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv
from tensor2tensor.rl.rl_utils import absolute_hinge_difference
from tensor2tensor.rl.rl_utils import full_game_name
# Import flags from t2t_trainer and trainer_model_based
import tensor2tensor.rl.trainer_model_based_params  # pylint: disable=unused-import
from tensor2tensor.utils import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("video_dir", "gym-results/",
                    "Where to save played trajectories.")
flags.DEFINE_float("zoom", 3,
                   "Resize factor of displayed game.")
flags.DEFINE_float("fps", 30,
                   "Frames per second.")
flags.DEFINE_string("epoch", "last",
                    "Data from which epoch to use.")
flags.DEFINE_boolean("sim_and_real", True,
                     "Compare simulated and real environment.")
flags.DEFINE_boolean("simulated_env", True,
                     "Either to use 'simulated' or 'real' env.")
flags.DEFINE_boolean("dry_run", False,
                     "Dry run - without pygame interaction and display, just "
                     "some random actions on environment")
flags.DEFINE_string("model_ckpt", "",
                    "World model checkpoint path.")
flags.DEFINE_string("wm_dir", "/home/student/t2t_train/mb_sd_pong_pretrained3/142/world_model",
                    "Directory with world model checkpoints. Inferred from "
                    "output_dir if empty.")
flags.DEFINE_string("policy_dir", "",
                    "Directory with policy. Inferred from output_dir if empty.")
flags.DEFINE_string("episodes_data_dir", "",
                    "Path to data for simulated environment initialization. "
                    "Inferred from output_dir if empty.")
flags.DEFINE_boolean("game_from_filenames", False,
                     "If infer game name from data_dir filenames or from "
                     "hparams.")
flags.DEFINE_boolean('show_all_actions', False, 'Show all possible actions and their course')
dry_run = False
show_all_actions = True

@registry.register_hparams
def planner_small(): #todo adapt to tiny?
  return HParams(
      num_rollouts=64,
      planning_horizon=16,
      rollout_agent_type="policy",
      batch_size=64,
      env_type="simulated",
      uct_const=0.0,
      uniform_first_action=True,
  )

class PlayerEnv(gym.Env):
  """Base (abstract) environment for interactive human play with gym.utils.play.

  Additionally to normal actions passed to underlying environment(s) it
  allows to pass special actions by `step` method.

  Special actions:
    RETURN_DONE_ACTION: Returns done from `step` to force gym.utils.play to
      call reset.
    TOGGLE_WAIT_ACTION: Change between real-time-play and wait-for-pressed-key
      modes.
    WAIT_MODE_NOOP_ACTION: perform noop action (when wait-for-pressed-key mode
    is on)

  For keyboard keys related to actions above see `get_keys_to_action` method.

  Naming conventions:
    envs_step_tuples: Dictionary of tuples similar to these returned by
      gym.Env.step().
      {
        "env_name": (observation, reward, done, info),
        ...
      }
      Keys depend on subclass.
  """

  # Integers (as taken by step() method) related to special actions.
  RETURN_DONE_ACTION = 101
  TOGGLE_WAIT_ACTION = 102
  WAIT_MODE_NOOP_ACTION = 103

  HEADER_HEIGHT = 27

  def __init__(self, action_meanings):
    """Constructor for PlayerEnv.

    Args:
      action_meanings: list of strings indicating action names. Can be obtain by
        >>> env = gym.make("PongNoFrameskip-v4")  # insert your game name
        >>> env.unwrapped.get_action_meanings()
        See gym AtariEnv get_action_meanings() for more details.
    """
    self.action_meanings = action_meanings
    self._wait = True
    # If action_space will be needed, one could use e.g. gym.spaces.Dict.
    self.action_space = None
    self._last_step_tuples = None
    self.action_meanings = action_meanings
    self.name_to_action_num = {name: num for num, name in
                               enumerate(self.action_meanings)}

  def get_keys_to_action(self):
    """Get mapping from keyboard keys to actions.

    Required by gym.utils.play in environment or top level wrapper.

    Returns:
      {
        Unicode code point for keyboard key: action (formatted for step()),
        ...
      }
    """
    # Based on gym AtariEnv.get_keys_to_action()
    #print('PlayerEnv.get_keys_to_action:here')
    keyword_to_key = {
        "UP": ord("w"),
        "DOWN": ord("s"),
        "LEFT": ord("a"),
        "RIGHT": ord("d"),
        "FIRE": ord(" "),
    }

    keys_to_action = {}

    for action_id, action_meaning in enumerate(self.action_meanings):
      keys_tuple = tuple(sorted([
          key for keyword, key in keyword_to_key.items()
          if keyword in action_meaning]))
      assert keys_tuple not in keys_to_action
      keys_to_action[keys_tuple] = action_id

    # Special actions:
    keys_to_action[(ord("r"),)] = self.RETURN_DONE_ACTION
    keys_to_action[(ord("c"),)] = self.TOGGLE_WAIT_ACTION
    keys_to_action[(ord("n"),)] = self.WAIT_MODE_NOOP_ACTION

    #print('PlayerEnv.get_keys_to_action: ', keys_to_action)
    return keys_to_action

  def _player_actions(self):
    return {
        self.RETURN_DONE_ACTION: self._player_return_done_action,
        self.TOGGLE_WAIT_ACTION: self._player_toggle_wait_action,
    }

  def _player_toggle_wait_action(self):
    self._wait = not self._wait
    return self._last_step_tuples

  def step(self, action):
    """Pass action to underlying environment(s) or perform special action."""
    # Special codes
    if type(action)== np.ndarray:
        action = int(action)
    if action in self._player_actions(): # 101: SimAndRealEnvPlayer._player_return_done_action, 102: PlayerEnv._player_toggle_wait_action, 110: SimAndRealEnvPlayer.player_restart_simulated_env_action
      envs_step_tuples = self._player_actions()[action]()
    elif self._wait and action == self.name_to_action_num["NOOP"]:
      # Ignore no-op, do not pass to environment.
      envs_step_tuples = self._last_step_tuples
      if isinstance(self.sim_env, rl_utils.BatchStackWrapper):
          envs_step_tuples = dict(envs_step_tuples)
          envs_step_tuples['sim_env'] = list(envs_step_tuples['sim_env'])
          envs_step_tuples['sim_env'][0] = self.sim_env._history_buffer
    else:
      # Run action on environment(s).
      if action == self.WAIT_MODE_NOOP_ACTION:
        action = self.name_to_action_num["NOOP"]
      # Perform action on underlying environment(s).
      envs_step_tuples = self._step_envs(action)
      self._update_statistics(envs_step_tuples)

    if len(envs_step_tuples['sim_env'][0].shape)>3:
        envs_step_tuples_laststep= dict(envs_step_tuples)
        envs_step_tuples_laststep['sim_env'] = list(envs_step_tuples_laststep['sim_env'])
        envs_step_tuples_laststep['sim_env'][0] = envs_step_tuples_laststep['sim_env'][0][0][-1]
        self._last_step_tuples = envs_step_tuples_laststep
    else:
        self._last_step_tuples = envs_step_tuples
    #print('PlayerEnv.step(): ', envs_step_tuples)
    #print('PlayerEnv.step(): ', envs_step_tuples)
    ob, reward, done, info = self._player_step_tuple(envs_step_tuples)
    return ob, reward, done, info

  def _augment_observation(self, ob, reward, cumulative_reward):
    """"Expand observation array with additional information header (top rows).

    Args:
      ob: observation
      reward: reward to be included in header.
      cumulative_reward: total cumulated reward to be included in header.

    Returns:
      Expanded observation array.
    """
    img = PIL_Image().new("RGB",
                          (ob.shape[1], self.HEADER_HEIGHT,))
    draw = PIL_ImageDraw().Draw(img)
    draw.text(
        (1, 0), "c:{:3}, r:{:3}".format(int(cumulative_reward), int(reward)),
        fill=(255, 0, 0)
    )
    draw.text(
        (1, 15), "fc:{:3}".format(int(self._frame_counter)),
        fill=(255, 0, 0)
    )
    header = np.asarray(img)
    del img
    header.setflags(write=1) #ValueError: cannot set WRITEABLE flag to True of this array
    # Top row color indicates if WAIT MODE is on.
    if self._wait:
      pixel_fill = (0, 255, 0)
    else:
      pixel_fill = (255, 0, 0)
    header[0, :, :] = pixel_fill
    return np.concatenate([header, ob], axis=0)

  def reset(self):
    raise NotImplementedError

  def _step_envs(self, action):
    """Perform action on underlying environment(s)."""
    raise NotImplementedError

  def _update_statistics(self, envs_step_tuples):
    """Update underlying environment(s) total cumulative rewards."""
    raise NotImplementedError

  def _player_return_done_action(self):
    """Function.

    Returns:
       envs_step_tuples: such that `player_step_tuple(envs_step_tuples)`
        will return done.
    """
    raise NotImplementedError

  def _player_step_tuple(self, envs_step_tuples):
    """Infer return tuple for step() given underlying environment tuple(s)."""
    raise NotImplementedError


class SimAndRealEnvPlayer(PlayerEnv):
  """Run simulated and real env side-by-side for comparison.

  Displays three windows - one for real environment, second for simulated
  and third for their differences.

  Normal actions are passed to both environments.

  Special Actions:
    RESTART_SIMULATED_ENV_ACTION: restart simulated environment only, using
      current frames from real environment.
    See `PlayerEnv` for rest of special actions.

  Naming conventions:
    envs_step_tuples: dictionary with two keys.
    {
      "real_env": (observation, reward, done, info),
      "sim_env": (observation, reward, done, info)
    }
  """

  RESTART_SIMULATED_ENV_ACTION = 110

  def __init__(self, real_env, sim_env, action_meanings):
    """Init.

    Args:
      real_env: real environment such as `FlatBatchEnv<T2TGymEnv>`.
      sim_env: simulation of `real_env` to be compared with. E.g.
        `SimulatedGymEnv` must allow to update initial frames for next reset
        with `add_to_initial_stack` method.
      action_meanings: list of strings indicating action names. Can be obtain by
        >>> env = gym.make("PongNoFrameskip-v4")  # insert your game name
        >>> env.unwrapped.get_action_meanings()
        See gym AtariEnv get_action_meanings() for more details.
    """
    super(SimAndRealEnvPlayer, self).__init__(action_meanings)
    assert real_env.observation_space.shape == sim_env.observation_space.shape
    self.real_env = real_env
    self.sim_env = sim_env
    orig = self.real_env.observation_space
    # Observation consists three side-to-side images - simulated environment
    # observation, real environment observation and difference between these
    # two.
    shape = (orig.shape[0] + self.HEADER_HEIGHT, orig.shape[1] * 3,
             orig.shape[2])

    self.observation_space = gym.spaces.Box(low=orig.low.min(),
                                            high=orig.high.max(),
                                            shape=shape, dtype=orig.dtype)

  def _player_actions(self):
    actions = super(SimAndRealEnvPlayer, self)._player_actions()
    actions.update({
        self.RESTART_SIMULATED_ENV_ACTION:
            self.player_restart_simulated_env_action,
    })
    return actions

  def get_keys_to_action(self):
    #print('SimAndRealEnvPlayer.get_keys_to_action:here')
    keys_to_action = super(SimAndRealEnvPlayer, self).get_keys_to_action()
    keys_to_action[(ord("x"),)] = self.RESTART_SIMULATED_ENV_ACTION
    return keys_to_action

  def _player_step_tuple(self, envs_step_tuples):
    """Construct observation, return usual step tuple.

    Args:
      envs_step_tuples: tuples.

    Returns:
      Step tuple: ob, reward, done, info
        ob: concatenated images [simulated observation, real observation,
          difference], with additional informations in header.
        reward: real environment reward
        done: True iff. envs_step_tuples['real_env'][2] is True
        info: real environment info
    """
    ob_real, reward_real, _, _ = envs_step_tuples["real_env"]
    if len(envs_step_tuples["sim_env"][0].shape) ==5:
        ob_sim, reward_sim, _, _ = envs_step_tuples["sim_env"]
        ob_sim = ob_sim[0][-1]
    else:
        ob_sim, reward_sim, _, _ = envs_step_tuples["sim_env"]
    ob_err = absolute_hinge_difference(ob_sim, ob_real)

    ob_real_aug = self._augment_observation(ob_real, reward_real,
                                            self.cumulative_real_reward)
    ob_sim_aug = self._augment_observation(ob_sim, reward_sim,
                                           self.cumulative_sim_reward)
    ob_err_aug = self._augment_observation(
        ob_err, reward_sim - reward_real,
        self.cumulative_sim_reward - self.cumulative_real_reward
    )
    ob = np.concatenate([ob_sim_aug, ob_real_aug, ob_err_aug], axis=1)
    _, reward, done, info = envs_step_tuples["real_env"]
    if len(envs_step_tuples['sim_env'][0].shape) >3:
        return (ob,envs_step_tuples['sim_env'][0]), reward, done, info
    else:
        return ob, reward, done, info

  def reset(self):
    """Reset simulated and real environments."""
    self._frame_counter = 0
    ob_real = self.real_env.reset()
    # Initialize simulated environment with frames from real one.
    if isinstance(self.sim_env, rl_utils.BatchStackWrapper):
        self.sim_env.env.add_to_initial_stack(ob_real)
        for _ in range(3):
            ob_real, _, _, _ = self.real_env.step(self.name_to_action_num["NOOP"])
            self.sim_env.env.add_to_initial_stack(ob_real)
        obs4_sim = self.sim_env.reset() #shape (1, 4, 105, 80, 3)
        ob_sim = obs4_sim[0][0]
        assert np.all(ob_real == ob_sim)
        self._last_step_tuples = self._pack_step_tuples((ob_real, 0, False, {}),
                                                        (ob_sim, 0, False, {}))
        self.set_zero_cumulative_rewards()
        ob, _, _, _ = self._player_step_tuple(self._last_step_tuples)
        return ob, obs4_sim # shows three images (sim, real, diff + annotations)

    else:
        self.sim_env.add_to_initial_stack(ob_real)
        for _ in range(3):
            ob_real, _, _, _ = self.real_env.step(self.name_to_action_num["NOOP"])
            self.sim_env.add_to_initial_stack(ob_real)
        ob_sim = self.sim_env.reset()
        assert np.all(ob_real == ob_sim)
        self._last_step_tuples = self._pack_step_tuples((ob_real, 0, False, {}),
                                                        (ob_sim, 0, False, {}))
        self.set_zero_cumulative_rewards()
        ob, _, _, _ = self._player_step_tuple(self._last_step_tuples)
        return ob

  def _pack_step_tuples(self, real_env_step_tuple, sim_env_step_tuple):
    return dict(real_env=real_env_step_tuple,
                sim_env=sim_env_step_tuple)

  def set_zero_cumulative_rewards(self):
    self.cumulative_real_reward = 0
    self.cumulative_sim_reward = 0

  def _step_envs(self, action):
    """Perform step(action) on environments and update initial_frame_stack."""
    #print('SimAndRealEnvPlayer._step_envs) ', action)

    self._frame_counter += 1
    real_env_step_tuple = self.real_env.step(action)
    sim_env_step_tuple = self.sim_env.step(action)
    if isinstance(self.sim_env, rl_utils.BatchStackWrapper):
        self.sim_env.env.add_to_initial_stack(real_env_step_tuple[0])

    else:
        self.sim_env.add_to_initial_stack(real_env_step_tuple[0])
    return self._pack_step_tuples(real_env_step_tuple, sim_env_step_tuple)

  def _update_statistics(self, envs_step_tuples):
    self.cumulative_real_reward += envs_step_tuples["real_env"][1]
    self.cumulative_sim_reward += envs_step_tuples["sim_env"][1]

  def _player_return_done_action(self):
    ob = np.zeros(self.real_env.observation_space.shape, dtype=np.uint8)
    return self._pack_step_tuples((ob, 0, True, {}),
                                  (ob, 0, True, {}))

  def player_restart_simulated_env_action(self):
    self._frame_counter = 0
    ob = self.sim_env.reset()
    assert np.all(self._last_step_tuples["real_env"][0] == ob)
    self.set_zero_cumulative_rewards()
    return self._pack_step_tuples(
        self._last_step_tuples["real_env"], (ob, 0, False, {}))


class SingleEnvPlayer(PlayerEnv):
  """"Play on single (simulated or real) environment.

  See `PlayerEnv` for more details.

  Naming conventions:
    envs_step_tuples: dictionary with single key.
      {
        "env": (observation, reward, done, info),
      }
      Plural form used for consistency with `PlayerEnv`.
  """

  def __init__(self, env, action_meanings):
    super(SingleEnvPlayer, self).__init__(action_meanings)
    self.env = env
    # Set observation space
    orig = self.env.observation_space
    shape = tuple([orig.shape[0] + self.HEADER_HEIGHT] + list(orig.shape[1:]))
    self.observation_space = gym.spaces.Box(low=orig.low.min(),
                                            high=orig.high.max(),
                                            shape=shape, dtype=orig.dtype)

  def _player_step_tuple(self, envs_step_tuples):
    """Augment observation, return usual step tuple."""
    ob, reward, done, info = envs_step_tuples["env"]
    ob = self._augment_observation(ob, reward, self.cumulative_reward)
    return ob, reward, done, info

  def _pack_step_tuples(self, env_step_tuple):
    return dict(env=env_step_tuple)

  def reset(self):
    self._frame_counter = 0
    ob = self.env.reset()
    self._last_step_tuples = self._pack_step_tuples((ob, 0, False, {}))
    self.cumulative_reward = 0
    return self._augment_observation(ob, 0, self.cumulative_reward)

  def _step_envs(self, action):
    self._frame_counter += 1
    return self._pack_step_tuples(self.env.step(action))

  def _update_statistics(self, envs_step_tuples):
    _, reward, _, _ = envs_step_tuples["env"]
    self.cumulative_reward += reward

  def _player_return_done_action(self):
    ob = np.zeros(self.env.observation_space.shape, dtype=np.uint8)
    return self._pack_step_tuples((ob, 0, True, {}))

def make_agent(
    agent_type, env, policy_hparams, policy_dir, sampling_temp,
    sim_env_kwargs_fn=None, frame_stack_size=None, rollout_agent_type=None,
    batch_size=None, inner_batch_size=None, env_type=None, **planner_kwargs
):
  """Factory function for Agents."""
  if batch_size is None:
    batch_size = env.batch_size
  return {
      "random": lambda: rl_utils.RandomAgent(  # pylint: disable=g-long-lambda
          batch_size, env.observation_space, env.action_space
      ),
      "policy": lambda: rl_utils.PolicyAgent(  # pylint: disable=g-long-lambda
          batch_size, env.observation_space, env.action_space,
          policy_hparams, policy_dir, sampling_temp
      )#,
      #"planner": lambda: rl_utils.PlannerAgent(  # pylint: disable=g-long-lambda
      #    batch_size, make_agent(
      #        rollout_agent_type, env, policy_hparams, policy_dir,
      #        sampling_temp, batch_size=inner_batch_size
      #    ), make_env(env_type, env.env, sim_env_kwargs_fn()),
      #    lambda env: rl_utils.BatchStackWrapper(env, frame_stack_size),
      #    discount_factor=policy_hparams.gae_gamma, **planner_kwargs
      #),
  }[agent_type]()

def make_agent_from_hparams(
    agent_type, base_env, stacked_env, loop_hparams, policy_hparams,
    planner_hparams, model_dir, policy_dir, sampling_temp, video_writers=()
):
  """Creates an Agent from hparams."""
  def sim_env_kwargs_fn():
    return rl.make_simulated_env_kwargs(
        base_env, loop_hparams, batch_size=planner_hparams.batch_size,
        model_dir=model_dir
    )
  planner_kwargs = planner_hparams.values()
  planner_kwargs.pop("batch_size")
  planner_kwargs.pop("rollout_agent_type")
  planner_kwargs.pop("env_type")
  return make_agent(
      agent_type, stacked_env, policy_hparams, policy_dir, sampling_temp,
      sim_env_kwargs_fn, 4, #loop_hparams.frame_stack_size
      planner_hparams.rollout_agent_type,
      inner_batch_size=planner_hparams.batch_size,
      env_type=planner_hparams.env_type,
      video_writers=video_writers, **planner_kwargs
  )

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def resume_game(agent, env, screen, observations, simenv_pvar, simenv_var, realenv_var, pauseobs, frame_counter,
                laststeptuples, pause_reward, action):
    print(action)
    env.sim_env.env.env.batch_env._actions_t = simenv_pvar['actions']
    env.sim_env.env.env.batch_env._batch_env = simenv_pvar['batch_env']
    env.sim_env.env.env.batch_env._dones_t = simenv_pvar['dones']
    env.sim_env.env.env.batch_env._indices_t = simenv_pvar['indices']
    env.sim_env.env.env.batch_env._obs_t = simenv_pvar['obs']
    env.sim_env.env.env.batch_env._reset_t = simenv_pvar['reset']
    env.sim_env.env.env.batch_env._reset_op = simenv_pvar['reset']
    env.sim_env.env.env.batch_env._rewards_t = simenv_pvar['reward']
    env.sim_env.env.env.batch_env._sess = simenv_pvar['sess']

    env.real_env.batch_env._envs[0].env.env.restore_full_state(realenv_var['alestate'])
    env.real_env.batch_env.state[0] = realenv_var['state']

    env.sim_env._history_buffer = simenv_var['hbuffer']
    env.sim_env._initial_frames = simenv_var['iframes']

    env._frame_counter = frame_counter
    env._last_step_tuples = laststeptuples
    env.cumulative_real_reward = pause_reward['realr']
    env.cumulative_real_reward = pause_reward['simr']

    obs4, obsshow = pauseobs
    for i in range(5):
        observations.append(np.zeros(shape=(132,240,3), dtype=np.uint8))

    video_size = [obsshow.shape[1], obsshow.shape[0]]
    zoom = 3
    video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    obs, rew, env_done, info = env.step(action)
    obsshow, obs4 = obs
    observations.append(obsshow)

    tot_reward = 0
    for i in range(10):
        # observations: 4 stacked observations, shape: (1,4,105,80,3)
        actions = agent.act(obs4, {})
        print(actions)
        obs, rew, env_done, info = env.step(actions)
        tot_reward += rew
        obsshow, obs4 = obs
        # rendered = env.render(mode='rgb_array')
        #display_arr(screen, obsshow, transpose=True, video_size=video_size)
        observations.append(obsshow)
        time.sleep(0.5)
        #pygame.display.flip()
    env.close()
    #pygame.quit()
    return observations, tot_reward

def main(dry_run=False, show_all_actions=False):
  # gym.logger.set_level(gym.logger.DEBUG)
  hparams = registry.hparams(FLAGS.loop_hparams_set) # add planner_small
  hparams.parse(FLAGS.loop_hparams)
  # Not important for experiments past 2018
  if "wm_policy_param_sharing" not in hparams.values().keys():
    hparams.add_hparam("wm_policy_param_sharing", False)
  directories = player_utils.infer_paths(
      output_dir=FLAGS.output_dir,
      world_model=FLAGS.wm_dir,
      policy=FLAGS.policy_dir,
      data=FLAGS.episodes_data_dir)
  if FLAGS.game_from_filenames:
    hparams.set_hparam(
        "game", player_utils.infer_game_name_from_filenames(directories["data"])
    )
  action_meanings = gym.make(full_game_name(hparams.game)).\
      unwrapped.get_action_meanings()
  epoch = FLAGS.epoch if FLAGS.epoch == "last" else int(FLAGS.epoch)

  def make_real_env():
    env = player_utils.setup_and_load_epoch(
        hparams, data_dir=directories["data"],
        which_epoch_data=None)
    env = FlatBatchEnv(env)  # pylint: disable=redefined-variable-type
    return env

  def make_simulated_env(setable_initial_frames, which_epoch_data):
    env = player_utils.load_data_and_make_simulated_env(
        directories["data"], directories["world_model"],
        hparams, which_epoch_data=which_epoch_data,
        setable_initial_frames=setable_initial_frames)
    return env

  if FLAGS.sim_and_real:
    sim_env = make_simulated_env(
        which_epoch_data=None, setable_initial_frames=True)
    real_env = make_real_env()
    env = SimAndRealEnvPlayer(real_env, sim_env, action_meanings)
  else:
    if FLAGS.simulated_env:
      env = make_simulated_env(  # pylint: disable=redefined-variable-type
          which_epoch_data=epoch, setable_initial_frames=False)
    else:
      env = make_real_env()
    env = SingleEnvPlayer(env, action_meanings)  # pylint: disable=redefined-variable-type

  #env = player_utils.wrap_with_monitor(env, FLAGS.video_dir)
  #env.reset()
  #for i in range(10):
  #    obs, rew, env_done, info = env.step(i%6)
  #    rendered = env.render(mode='rgb_array')

  if FLAGS.dry_run or dry_run: #intervene with single action
    print('dry run')
    # build agent
    env.sim_env = rl_utils.BatchStackWrapper(env.sim_env, stack_size=4)
    eval_hparams = trainer_lib.create_hparams('ppo_original_params')
    planner_hparams = hparams_lib.create_hparams('planner_small')
    policy_dir= '~/t2t_train/mb_sd_pong_pretrained3/142/policy'
    agent = make_agent_from_hparams(agent_type='policy', base_env=env.real_env, stacked_env=env.sim_env, loop_hparams=FLAGS.loop_hparams,
                                    policy_hparams=eval_hparams, planner_hparams=planner_hparams, model_dir="", policy_dir=policy_dir, sampling_temp=0.5, video_writers=())


    env.unwrapped.get_keys_to_action()
    obsshow, obs4 = env.reset()

    video_size = [obsshow.shape[1],obsshow.shape[0]]
    zoom = 3
    video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
    screen = pygame.display.set_mode(video_size)

    clock = pygame.time.Clock()
    pygame.display.init()
    pong_human_sets_pause=False
    manual_action = None
    actions = None
    env_done = False
    total_reward = 0
    for _ in range(1):
      # teilweise von play.play kopiert
      observations = []
      while not env_done:
        if actions == None:
            actions = agent.act(obs4, {})
        else:
            actions = manual_action
        print(actions)
        obs, rew, env_done, info = env.step(actions)
        total_reward += rew
        obsshow, obs4 = obs
        display_arr(screen, obsshow, transpose=True, video_size=video_size)
        observations.append(obsshow)
        time.sleep(1)
        pygame.display.flip()
        clock.tick(FLAGS.fps)
        actions = None

        for event in pygame.event.get(): #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
            if event.type ==pygame.KEYDOWN:
                print(event.key)
                if event.key == 97: # DOWN : A
                    time.sleep(1)
                    #manual_action = np.array([3])
                    print('Keys down')
                    for i in range(2):
                        obs, rew, env_done, info = env.step(np.array([3]))
                        total_reward += rew
                        obsshow, obs4 = obs
                        display_arr(screen, obsshow, transpose=True, video_size=video_size)
                        pygame.display.flip()
                elif event.key == 100: # UP : D
                    time.sleep(1)
                    #manual_action = np.array([2])
                    print('Keys up')
                    for i in range(2):
                        obs, rew, env_done, info = env.step(np.array([2]))
                        total_reward += rew
                        obsshow, obs4 = obs
                        display_arr(screen, obsshow, transpose=True, video_size=video_size)
                        pygame.display.flip()
                elif event.key == 120: # RESET SIM_ENV : X
                    time.sleep(1)
                    obs, rew, env_done, info = env.step(110)
                    obsshow, obs4 = obs
                elif event.key == 114: # RESET ENV : R
                    time.sleep(1)
                    obsshow, obs4 = env.reset()
                    rew = 0
                    info = {}
                elif event.key == 110: # NOOP : N
                    time.sleep(1)
                    print('NOOP')
                    for i in range(2):
                        obs, rew, env_done, info = env.step(103)
                        total_reward += rew
                        obsshow, obs4 = obs
                        display_arr(screen, obsshow, transpose=True, video_size=video_size)
                        pygame.display.flip()
            elif event.type == pygame.QUIT:
                env_done = True

      env.step(PlayerEnv.RETURN_DONE_ACTION)  # reset
    observations = np.array(observations)
    lenframes = observations.shape[0]
    env.close()
    pygame.quit()
    return observations, lenframes, total_reward
  elif FLAGS.show_all_actions or show_all_actions: #press space and show all actions
      # build agent
      env.sim_env = rl_utils.BatchStackWrapper(env.sim_env, stack_size=4)
      eval_hparams = trainer_lib.create_hparams('ppo_original_params')
      planner_hparams = hparams_lib.create_hparams('planner_small')
      policy_dir = '~/t2t_train/mb_sd_pong_pretrained3/142/policy'
      agent = make_agent_from_hparams(agent_type='policy', base_env=env.real_env, stacked_env=env.sim_env,
                                      loop_hparams=FLAGS.loop_hparams,
                                      policy_hparams=eval_hparams, planner_hparams=planner_hparams, model_dir="",
                                      policy_dir=policy_dir, sampling_temp=0.5, video_writers=())

      env.unwrapped.get_keys_to_action()
      obsshow, obs4 = env.reset()

      video_size = [obsshow.shape[1], obsshow.shape[0]]
      zoom = 3
      video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
      screen = pygame.display.set_mode(video_size)

      pong_human_pause = False

      # teilweise von play.play kopiert
      observations = []
      env_done = False
      total_reward = 0
      while not env_done:
          # observations: 4 stacked observations, shape: (1,4,105,80,3)
          actions = agent.act(obs4, {})
          obs, rew, env_done, info = env.step(actions)
          total_reward += rew
          obsshow, obs4 = obs

          display_arr(screen, obsshow, transpose=True, video_size=video_size)
          observations.append(obsshow)
          pygame.display.flip()
          time.sleep(0.5)

          for event in pygame.event.get():  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
              if event.type == pygame.KEYDOWN:
                  if event.key == pygame.K_SPACE:  # unten 5
                      pong_human_pause = True
                      print('space angekommen')
                      # Save Game Status
                  elif event.key == 120: # RESET SIM_ENV : X
                      time.sleep(1)
                      obs, rew, env_done, info = env.step(110)
                      obsshow, obs4 = obs
          if pong_human_pause:
              simenv_pvar = {
                  'actions': env.sim_env.env.env.batch_env._actions_t,
                  'batch_env': env.sim_env.env.env.batch_env._batch_env,
                  'dones': env.sim_env.env.env.batch_env._dones_t,
                  'indices': env.sim_env.env.env.batch_env._indices_t,
                  'obs': env.sim_env.env.env.batch_env._obs_t,
                  'reset': env.sim_env.env.env.batch_env._reset_op,
                  'reward': env.sim_env.env.env.batch_env._rewards_t,
                  'sess': env.sim_env.env.env.batch_env._sess
              }
              simenv_var = {
                  'hbuffer': env.sim_env._history_buffer,
                  'iframes': env.sim_env._initial_frames
              }
              realenv_var={
                  'state': env.real_env.batch_env.state,
                  'alestate': env.real_env.batch_env._envs[0].env.env.clone_full_state()
              }
              pauseobs = (obs4, obsshow)
              frame_counter = env._frame_counter
              laststeptuples = env._last_step_tuples
              from copy import deepcopy
              pause_observations = deepcopy(observations) # try deepcopy
              pause_reward = {'realr': env.cumulative_real_reward,
                              'simr': env.cumulative_sim_reward}

              #resume game with different actions
              # normal
              print('resume game')
              obs_normal, trewardn = resume_game(agent, env, screen, deepcopy(observations), simenv_pvar, simenv_var,
                            realenv_var, pauseobs, frame_counter, laststeptuples, pause_reward, np.array([0]))
              trewardn += total_reward
              print('normalgamedone')
              # oben
              obs_up, trewardu = resume_game(agent, env, screen, deepcopy(observations), simenv_pvar, simenv_var,
                            realenv_var, pauseobs, frame_counter, laststeptuples, pause_reward, np.array([2]))
              trewardu += total_reward
              print('rightdone')
              # unten
              obs_down, trewardd = resume_game(agent, env, screen, deepcopy(observations), simenv_pvar, simenv_var,
                            realenv_var, pauseobs, frame_counter, laststeptuples, pause_reward, np.array([3]))
              trewardd += total_reward
              print('leftdone')
              break
      print('reseting')
      env.step(PlayerEnv.RETURN_DONE_ACTION)  # reset
      try:
          obs_normal = np.array(obs_normal)
          print(obs_normal.shape)
          obs_up = np.array(obs_up)
          obs_down = np.array(obs_down)
          print('concatenating')
          obs_total = np.concatenate((obs_up, obs_normal, obs_down), axis=2)
          print(obs_total.shape)
          print(obs_total.shape[0], obs_total.shape[1], obs_total.shape[2])
          lenframes = obs_total.shape[0]
      except:
          print('You didn\'t press the space button')
          #obs_total = np.ones(shape=(50, 264,1440, 3), dtype=np.uint8)
          #obs_total[:]=255
          obs_total = np.array(observations)
          height = obs_total.shape[1] *1.5
          width = obs_total.shape[2] *1.5
          lenframes = obs_total.shape[0]
      env.close()
      pygame.quit()
      return obs_total, lenframes, (trewardu, trewardn, trewardd)
  else:
      env = player_utils.wrap_with_monitor(env, FLAGS.video_dir)
      total_reward = play.play(env, zoom=FLAGS.zoom, fps=FLAGS.fps)
      env.close()
      return total_reward
  env.close()
  print('env closed')


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
