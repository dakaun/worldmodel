import dash
import dash_html_components as html
import base64
from PIL import Image
from WorldModelsExperiments.breakout.model import Model, make_model, _process_frame
import pandas as pd
import numpy as np
from pyglet.window import key
import time
import imageio
import cv2
from gym.envs.classic_control import rendering

path = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v3/200423'
rnn_path = path + '/tf_rnn/rnn.json'
vae_path = path + '/tf_vae/vae.json'
controller_path = path + '/log/breakout.cma.16.64.best.json'

model = make_model(rnn_path=rnn_path, vae_path=vae_path)
model.load_model(controller_path)
print('models loaded')

def key_press(symbol, mod):
    global human_sets_pause
    if symbol == key.SPACE:
        print('key pressed')
        human_sets_pause = not human_sets_pause

def play_game(model, num_episode=1, render_mode=True):
    global human_sets_pause
    human_sets_pause = False
    reward_list = []
    obs_sequence = np.zeros(shape=(10000, 210, 160, 3), dtype=np.uint8)
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
            _, action = model.get_action(z)
            obs, reward, done, info = model.env.step(action)

            # data = np.concatenate([z, model.state.h[0]]).reshape(1, 288)
            # tsne_data = tsne_data.append(pd.DataFrame(data), ignore_index=True)
            obs_sequence[seq_counter, :, :, :] = obs
            total_reward += reward
            seq_counter += 1
            time.sleep(0.2)

            if human_sets_pause:
                time.sleep(1)
                print('render for several steps done, shift with current reward: ', total_reward)
                time.sleep(2)
                human_sets_pause = False
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
    return obs_sequence, seq_counter, obs, model.state, total_reward, model.env.clone_full_state(), z

def generate_video(sequence, filename, counter):
    print('generating video..')
    print(filename)
    print(counter)
    with imageio.get_writer(filename, mode='I', macro_block_size=None) as video:
        for image in range(counter):
            video.append_data(sequence[image])
    print('done generating video')

def resume_game(pause_status, action):
    obs_normal = np.zeros(shape=(3000, 210, 160, 3), dtype=np.uint8)
    obs_normal[:pause_status['counter']] = pause_status['sequence']
    seq_counter = pause_status['counter'] + 7
    total_reward = pause_status['totalreward']
    obs = pause_status['obs']
    model.state = pause_status['modelstate']
    model.env.restore_full_state(pause_status['gamestate'])
    done = False

    if action==2 | action==3:
        for i in range(1):
            obs, _,_,_= model.env.step(action)
            #todo update model state
    obs_normal[seq_counter,:,:,:]=obs
    seq_counter+=1
    while not done and seq_counter < 3000:
        obs = _process_frame(obs)
        z, mu, logvar = model.encode_obs(obs)
        _, action = model.get_action(z)
        obs, reward, done, info = model.env.step(action)
        model.env.render('rgb_array')

        obs_normal[seq_counter, :, :, :] = obs
        total_reward += reward
        seq_counter += 1

    print('Episode is done with total reward: ', total_reward)
    #model.env.viewer.close()
    #model.env.viewer = None

    return obs_normal,seq_counter

def main():
    print('start playing game')
    initial_obs_sequence, seq_counter, obs, state, treward, gamestate, z = play_game(model)
    pause_status = {
        'sequence': initial_obs_sequence[:seq_counter,:,:,:],
        'counter':seq_counter,
        'obs': obs,
        'modelstate':state,
        'totalreward': treward,
        'gamestate':gamestate,
        'z':z
    }
    #normal
    resume_obs_sequence_normal, seq_countern = resume_game(pause_status,0)
    print('normalgamedone')
    #right
    resume_obs_sequence_right, seq_counterr = resume_game(pause_status, 2)
    print('rightdone')
    #left
    resume_obs_sequence_left, seq_counterl = resume_game(pause_status, 3)
    print('leftdone')
    all_images = np.concatenate((resume_obs_sequence_left,resume_obs_sequence_normal, resume_obs_sequence_right),axis=2)
    print(all_images.shape)

    fin_counter = max(seq_counterl,seq_countern,seq_counterr)
    model.env.viewer = rendering.SimpleImageViewer()
    for img in range(fin_counter):
        model.env.viewer.imshow(all_images[img])
        time.sleep(0.1)

if __name__ == '__main__':
    main()

