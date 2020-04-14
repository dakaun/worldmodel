import numpy as np
from gym.envs.atari.atari_env import AtariEnv
from gym.spaces.box import Box
from gym.utils import seeding
import os
import json
from PIL import Image

SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 3

with open(os.path.join('/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v2/200228/tf_initial_z', 'initial_z.json'), 'r') as f:
    [initial_mu, initial_logvar] = json.load(f)

initial_mu_logvar = [list(elem) for elem in zip(initial_mu, initial_logvar)]

def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    print('error with sampling ensemble')
    return -1

class BreakoutDream(AtariEnv):
    metadata = {
        'render.modes':['human', 'rgb_array']
    }
    def __init__(self, agent, frame_limit):
        self.observation_space = Box(low=0, high=255., shape=(64,64,3))
        self.agent = agent
        self.vae = agent.vae
        self.rnn = agent.rnn
        self.z_size = self.rnn.hps.encoded_img_width
        self.viewer = None
        self.frame_count = None
        self.frame_limit = frame_limit
        self.z = None
        self.temperature = 0.7

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sample_next_z(self, action):
        r_model = self.rnn
        temp = self.temperature

        sess = r_model.sess
        hps = r_model.hps

        prev_x = np.zeros((1, 1, self.z_size))
        prev_x[0][0] = self.z

        input_x = np.concatenate((prev_x, action.reshape(1, 1, 4)), axis=2)
        feed = {
            r_model.input_x : input_x,
            r_model.initial_state: self.agent.state
        }
        [logmix, mean, logstd, self.agent.state] = sess.run(
            [r_model.out_logmix, r_model.out_mean, r_model.out_logstd, r_model.final_state], feed)

        # adjust temperatures
        logmix2 = np.copy(logmix) / temp
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(self.z_size, 1)

        mixture_idx = np.zeros(self.z_size)
        chosen_mean = np.zeros(self.z_size)
        chosen_logstd = np.zeros(self.z_size)
        for j in range(self.z_size):
            idx = get_pi_idx(self.np_random.rand(), logmix2[j])
            mixture_idx[j] = idx
            chosen_mean[j] = mean[j][idx]
            chosen_logstd[j] = logstd[j][idx]

        rand_gaussian = self.np_random.randn(self.z_size)*np.sqrt(temp)
        next_x = chosen_mean + np.exp(chosen_logstd)*rand_gaussian

        next_z = next_x.reshape(self.z_size)

        return next_z

    def _reset(self):
        idx = self.np_random.randint(0,len(initial_mu_logvar))
        init_mu, init_logvar = initial_mu_logvar[idx]
        init_mu = np.array(init_mu)/10000.
        init_logvar = np.array(init_logvar)/10000.

        self.z = init_mu + np.exp(init_logvar/2.0) * self.np_random.randn(*init_logvar.shape)

        self.frame_count = 0
        return self.z

    def _step(self, action_one_hot):
        self.frame_count += 1
        next_z = self._sample_next_z(action_one_hot)
        reward = 0
        done = False
        if self.frame_count > self.frame_limit:
            done = True
        self.z = next_z
        return next_z, reward, done, {}

    def _render(self, mode='human', close = False):
        img = self.vae.decode(self.z.reshape(1, self.z_size))*255.
        img = np.round(img).astype(np.uint8)
        img = img.reshape(64,64,3)

        img = Image.fromarray(img)
        img_resize = img.resize(size=(int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))))
        img_resize = np.array(img_resize)

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            return img_resize
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img_resize)




def make_env(env_name, agent, frame_limit, seed=-1, render_mode = False):
    env = BreakoutDream(agent, frame_limit)
    if seed<0:
        seed = np.random.randint(2**31-1)
    env.seed(seed)
    return env