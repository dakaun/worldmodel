import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input,Output
import base64
from PIL import Image
from WorldModelsExperiments.breakout.model import make_model, _process_frame
from WorldModelsExperiments.breakout.rnn.rnn import rnn_next_state
import pandas as pd
import numpy as np
from pyglet.window import key
import time
import cv2
import gym


def key_press(symbol, mod):
    global human_sets_pause
    if symbol == key.SPACE:
        print('key pressed')
        human_sets_pause = not human_sets_pause

def play_game(model, num_episode=1, render_mode=True):
    global human_sets_pause
    human_sets_pause = False
    reward_list = []
    if 'Breakout' in model.env_name:
        obs_sequence = np.zeros(shape=(10000, 210, 160, 3), dtype=np.uint8)
    elif 'CarRacing' in model.env_name:
        obs_sequence = np.zeros(shape=(10000, 288, 288, 3), dtype=np.uint8)
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

            # data = np.concatenate([z, model.state.h[0]]).reshape(1, 288)
            # tsne_data = tsne_data.append(pd.DataFrame(data), ignore_index=True)
            obs_sequence[seq_counter, :, :, :] = obs
            total_reward += reward
            seq_counter += 1
            time.sleep(0.1)

            if human_sets_pause:
                time.sleep(1)
                print('render for several steps done, shift with current reward: ', total_reward)
                if 'Breakout' in model.env_name:
                    model_state = model.env.clone_full_state()
                else:
                    model_state = None
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
    return model, obs_sequence, seq_counter, obs, total_reward, model_state, z

def resume_game(model, pause_status, action):
    if 'Breakout' in model.env_name:
        obs_normal = np.zeros(shape=(10000, 210, 160, 3), dtype=np.uint8)
    elif 'CarRacing' in model.env_name:
        obs_normal = np.zeros(shape=(10000, 288, 288, 3), dtype=np.uint8)
    obs_normal[:pause_status['counter']] = pause_status['sequence']
    seq_counter = pause_status['counter'] + 10 # todo add white pages instead of black - more obvious
    total_reward = pause_status['totalreward']
    obs = pause_status['obs']
    model.state = pause_status['modelstate']

    if model.env_name=='Breakout':
        model.env.restore_full_state(pause_status['gamestate'])
    elif model.env_name=='CarRacing':
        model.env.car = pause_status['car']
        model.env.prev_reward = pause_status['prev_reward']
        model.env.reward = pause_status['reward']
        model.env.road = pause_status['road']
        model.env.road_poly = pause_status['road_poly']
        model.env.score_label = pause_status['score_label']
        model.env.start_alpha = pause_status['start_alpha']
        model.env.state = pause_status['env_state']
        model.env.t = pause_status['t']
        model.env.tile_visited_count = pause_status['tile_visited_count']
        model.env.track = pause_status['track']

    done = False

    if model.env_name== 'Breakout':
        if action==2 | action==3:
            for i in range(1):
                obs, _, _, _= model.env.step(action)
                model.env.render('rgb_array')
                obs_into_z= _process_frame(obs)
                z = model.vae.encode(obs_into_z.reshape(1,64,64,3))
                action_one_hot = np.zeros(model.num_actions)
                action_one_hot[action]=1
                model.state = rnn_next_state(model.rnn, z, action_one_hot, model.state, model.env_name)
    elif model.env_name=='CarRacing':
        for i in range(1):
            obs, _, _, _=model.env.step(action)
            model.env.render('rgb_array')
            obs_into_z = _process_frame(obs)
            z = model.vae.encode(obs_into_z.reshape(1, 64, 64, 3))
            model.state = rnn_next_state(model.rnn, z, action, model.state, model.env_name)

    obs_normal[seq_counter,:,:,:]=obs
    seq_counter+=1
    resume_counter=0
    while not done and resume_counter < 80:
        model.env.render('rgb_array')
        obs = _process_frame(obs)
        z, mu, logvar = model.encode_obs(obs)
        action,_ = model.get_action(z)
        obs, reward, done, info = model.env.step(action)
        #model.env.render('rgb_array')

        obs_normal[seq_counter, :, :, :] = obs
        total_reward += reward
        seq_counter += 1
        resume_counter +=1

    print('Episode is done with total reward: ', total_reward)
    #model.env.viewer.close()
    #model.env.viewer = None

    return obs_normal,seq_counter

app = dash.Dash(__name__)
server = app.server
app.config['suppress_callback_exceptions'] = True
#app.config.supress_callback_exceptions = True

colors = {
    'background-color': 'LightGray'
}
breakout = html.Div(id='header1',
                      style={
                          'textAlign': 'center',
                          'background-color': 'LightGray'
                      },
                      children=[
                          html.H1(children='Breakout Word Model'),
                          html.H3(children='Dashboard to display the world model of breakout.'),
                          html.Div(id='subbody', children=[
                              html.H5(['Press the Button to run Breakout.']),
                              html.Button('Start Breakout',
                                      id='start_gameb',
                                      n_clicks=0,
                                      style={
                                          'textAlign': 'center'

                                      }),
                          ]),
                          html.Video(id='initial_game_videob',
                                     controls=True,
                                     style={
                                         'textAlign': 'center'
                                     },
                                     height=357,
                                     width=816
                                     )
                      ])
carracing = html.Div(id='header1',
                      style={
                          'textAlign': 'center',
                          'background-color': 'LightGray'
                      },
                      children=[
                          html.H1(children='CarRacing World Model'),
                          html.H3(children='Dashboard to display word model of CarRacing.'),
                          html.Div(id='subbody', children=[
                              html.H5(['Press Button to run CarRacing.']),
                              html.Button('Start CarRacing',
                                          id='start_gamec',
                                          n_clicks=0)
                          ]),
                          html.Video(id='initial_game_videoc',
                                     controls=True,
                                     style={
                                         'textAlign': 'center'
                                     },
                                     height=357,
                                     width=816
                                     )
                     ])

overview = html.Div(id='header1',
                    style={
                          'textAlign': 'center',
                          'background-color': 'LightGray'
                      },
                    children=[
                        html.H1('Explanations of different World Model States'),
                        html.H3(children='Choose either CarRacing or Breakout to display Page')
                    ])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# todo below video display further information about each result
# todo possible to refer back to frame number from video second - then ability to press button/input second of video to get more information about that gamestatus

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname=='/':
        return overview
    elif pathname=='/breakout':
        return breakout
    elif pathname=='/carracing':
        return carracing

@app.callback(Output('initial_game_videoc', 'src'),
              [Input('url', 'pathname'),
               Input('start_gamec', 'n_clicks')])
def start_carracinggame(page, buttonclick):
    path = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v3/200522'
    rnn_path = path + '/tf_rnn/rnn.json'
    vae_path = path + '/tf_vae/vae.json'
    controller_path = path + '/log/carracing.cma.16.64.best.json'
    env_name = 'CarRacing'
    model = make_model(env_name=env_name, rnn_path=rnn_path, vae_path=vae_path)
    model.load_model(controller_path)
    print('Carracing models loaded')

    if ('carracing' in page) and buttonclick:
        print('start playing game')
        model, initial_obs_sequence, seq_counter, obs, treward, gamestate, z = play_game(model)
        pause_status = {
            'sequence': initial_obs_sequence[:seq_counter, :, :, :],
            'counter': seq_counter,
            'obs': obs,
            'modelstate': model.state,
            'totalreward': treward,
            'gamestate': gamestate,
            'car': model.env.car,
            'prev_reward': model.env.prev_reward,
            'reward': model.env.reward,
            'road': model.env.road,
            'road_poly': model.env.road_poly,
            'score_label': model.env.score_label,
            'start_alpha': model.env.start_alpha,
            'env_state': model.env.state,
            't': model.env.t,
            'tile_visited_count': model.env.tile_visited_count,
            'track': model.env.track

        }
        # normal
        resume_obs_sequence_normal, seq_countern = resume_game(model, pause_status, np.array([0.,0.,0.]))
        print('normalgamedone')
        # right
        resume_obs_sequence_right, seq_counterr = resume_game(model, pause_status, np.array([0.5,0.,0.]))
        print('rightdone')
        # left
        resume_obs_sequence_left, seq_counterl = resume_game(model, pause_status, np.array([-0.5,0.,0.]))
        print('leftdone')
        all_images = np.concatenate((resume_obs_sequence_left, resume_obs_sequence_normal, resume_obs_sequence_right),
                                    axis=2)
        fin_counter = max(seq_counterl, seq_countern, seq_counterr)
        init_obs_seq_filename = 'obs_video_carracing.webm'
        height = all_images[0].shape[0]
        width = all_images[0].shape[1]
        sequence = all_images[:, :, :, [2, 1, 0]]

        video = cv2.VideoWriter(init_obs_seq_filename, cv2.VideoWriter_fourcc(*'vp80'), 10, frameSize=(width, height))
        for image in range(fin_counter):
            video.write(sequence[image])
        video.release()
        print('done generating video')

        videom = open(init_obs_seq_filename, 'rb').read()
        encoded_video = base64.b64encode(videom).decode()
        print('send video to dashboard')
        return 'data:video/webm;base64,{}'.format(encoded_video)

@app.callback(Output('initial_game_videob', 'src'),
              [Input('url','pathname'),
               Input('start_gameb', 'n_clicks')])
def start_breakoutgame(page, buttonclick):
    path = '/home/student/Dropbox/MA/worldmodel/worldmodel-breakout-server-version-v3/200420/retrain/'
    rnn_path = path + '/tf_rnn/rnn.json'
    vae_path = path + '/tf_vae/vae.json'
    controller_path = path + '/log/breakout.cma.16.64.best.json'
    env_name = 'Breakout'
    model = make_model(env_name=env_name, rnn_path=rnn_path, vae_path=vae_path)
    model.load_model(controller_path)
    print('Breakout models loaded')

    if ('breakout' in page) and buttonclick:
        print('start playing game')
        model, initial_obs_sequence, seq_counter, obs, treward, gamestate, z = play_game(model)
        pause_status = {
            'sequence': initial_obs_sequence[:seq_counter, :, :, :],
            'counter': seq_counter,
            'obs': obs,
            'modelstate': model.state,
            'totalreward': treward,
            'gamestate': gamestate
        }
        # normal
        resume_obs_sequence_normal, seq_countern = resume_game(model, pause_status, 0)
        print('normalgamedone')
        # right
        resume_obs_sequence_right, seq_counterr = resume_game(model, pause_status, 2)
        print('rightdone')
        # left
        resume_obs_sequence_left, seq_counterl = resume_game(model, pause_status, 3)
        print('leftdone')
        all_images = np.concatenate((resume_obs_sequence_left, resume_obs_sequence_normal, resume_obs_sequence_right),
                                    axis=2)
        print(all_images.shape)

        fin_counter = max(seq_counterl, seq_countern, seq_counterr)
        init_obs_seq_filename = 'obs_video_breakout.webm'

        height = all_images[0].shape[0]
        width = all_images[0].shape[1]
        sequence = all_images[:, :, :, [2, 1, 0]]
        print(height)
        print(width)

        video = cv2.VideoWriter(init_obs_seq_filename, cv2.VideoWriter_fourcc(*'vp80'), 10, frameSize=(width, height))
        for image in range(fin_counter):
            video.write(sequence[image])
        video.release()
        print('done generating video')

        videom = open(init_obs_seq_filename, 'rb').read()
        encoded_video = base64.b64encode(videom).decode()
        print('send video to dashboard')
        return 'data:video/webm;base64,{}'.format(encoded_video)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=1872)
