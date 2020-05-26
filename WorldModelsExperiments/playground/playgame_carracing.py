from WorldModelsExperiments.breakout.model import Model, make_model, _process_frame
from pyglet.window import key
import time
import numpy as np
import imageio
from copy import deepcopy

path = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v3/200522'
rnn_path = path + '/tf_rnn/rnn.json'
vae_path = path + '/tf_vae/vae.json'
controller_path = path + '/log/carracing.cma.16.64.best.json'
env_name = 'CarRacing'
model = make_model(env_name, rnn_path=rnn_path, vae_path=vae_path)
model.load_model(controller_path)
print('models loaded')

def key_press(symbol, mod):
    global human_sets_pause
    if symbol == key.SPACE:
        print('key pressed')
        human_sets_pause = not human_sets_pause

def play_game(model, num_episode=1, render_mode=True):
    global human_sets_pause
    human_sets_pause=False
    reward_list = []
    obs_sequence = np.zeros(shape=(10000, 96, 96, 3), dtype=np.uint8)
    # tsne_data = pd.DataFrame()

    for episode in range(num_episode):
        total_reward = 0
        obs = model.env.reset()
        done = False
        seq_counter = 0

        while not done:
            model.env.render('human')
            model.env.unwrapped.viewer.window.on_key_press = key_press

            obs = _process_frame(obs)
            z, mu, logvar = model.encode_obs(obs)
            action, _ = model.get_action(z)
            obs, reward, done, info = model.env.step(action)

            #data = np.concatenate([z, model.state.h[0]]).reshape(1, 288)
            #tsne_data = tsne_data.append(pd.DataFrame(data), ignore_index=True)
            obs_sequence[seq_counter, :, :, :] = obs
            total_reward += reward
            seq_counter += 1
            #time.sleep(0.2)

            if human_sets_pause:
                time.sleep(1)
                print('render for several steps done, shift with current reward: ', total_reward)
                time.sleep(2)
                human_sets_pause = False
                #pause_state_env = deepcopy(model.env)
                print(action)

                model.env.viewer.close()
                model.env.viewer = None
                print('close env')
                break

        if done:
            print('game episode ', str(episode), ' is done with total reward: ', total_reward)
            if render_mode:
                model.env.viewer.close()
                model.env.viewer = None
                # model.env.close()
                print('close env')
            return obs_sequence, seq_counter
        time.sleep(2)
    return obs_sequence, seq_counter


if __name__ == '__main__':
    obs, counter = play_game(model)
    filename = 'test.mp4'
    with imageio.get_writer(filename, mode='I', macro_block_size=None) as video:
        for image in range(counter):
            video.append_data(obs[image])
    # filename, mode='I', macro_block_size=None, format='FFMPEG'
    #filename, mode='I', macro_block_size=None, format='FFMPEG', fps=1

    print('done')