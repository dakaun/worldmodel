'''
saves ~ 200 episodes generated from a random policy
'''
import numpy as np
import random
import os
import gym
from WorldModelsExperiments.breakout.model import make_model
from PIL import Image
import statistics

MAX_FRAMES = 2100
MAX_TRIALS = 100000
SEQ_LENGTH = 400  # todo adjust: as long as possible
INPUT_SHAPE = (64,64)

render_mode = False # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

def _process_frame(frame): # converts into (64,64,3)
  img = Image.fromarray(frame)
  img = img.resize(INPUT_SHAPE)  # resize
  obs = np.array(img)
  obs = obs /255.
  return obs

model = make_model(load_model=False)

total_frames = 0
for trial in range(MAX_TRIALS):  #
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []
    recording_reward = []
    appended_reward = 0

    np.random.seed(random_generated_int)
    model.env.seed(random_generated_int)

    # random policy
    #repeat = np.random.randint(1, 11)

    obs = model.env.reset() # the latent code
    obs = _process_frame(obs)

    if obs is None:
      obs = np.zeros(model.input_size)

    for frame in range(MAX_FRAMES):
      if render_mode:
        model.env.render("human")
      z, mu, logvar = model.encode_obs(obs)
      #action_one_hot, action = model.get_action(z) # use more diverse random policy:

      action = model.env.action_space.sample()
      action_one_hot = np.zeros(model.num_actions)
      action_one_hot[action] = 1

      #if frame % repeat == 0:
      #  action = np.random.rand() * 2.0 - 1.0
      #  repeat = np.random.randint(1, 11)

      recording_obs.append(obs)
      recording_action.append(action_one_hot)

      obs, reward, done, info = model.env.step(action)
      obs = _process_frame(obs)

      recording_reward.append(reward)
      appended_reward += reward

      if done:
        break

    total_frames += frame
    #print("dead at", frame, "total recorded frames for this trial", total_frames, " with final reward of ", appended_reward)
    if trial % 100 == 0 and trial > 0:
      print(f'At trial {trial} with an average reward of {statistics.mean(val for val in recording_reward)} ')
    if (len(recording_obs) > SEQ_LENGTH):
      recording_obs = np.array(recording_obs[0:SEQ_LENGTH], dtype=np.float16)
      recording_action = np.array(recording_action[0:SEQ_LENGTH], dtype=np.float16)
      recording_reward = np.array(recording_reward[0:SEQ_LENGTH], dtype=np.float16)
      np.savez_compressed(filename, obs=recording_obs, action=recording_action, reward=recording_reward)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()
print('extraction done, observations saved')
