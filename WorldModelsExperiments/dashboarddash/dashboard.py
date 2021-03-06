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
import os

import sys
sys.path.append('../../tensor2tensor')
import tensor2tensor
from tensor2tensor.rl import player
import runpy

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
                          html.H1(children='Breakout World Model'),
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
                          html.H3(children='Dashboard to display Word Model of CarRacing.'),
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

worldmodel_kaiser = html.Div(id='header1',
                             style={
                    'textAlign': 'center',
                    'background-color': 'LightGray'
                },
                             children=[
                    html.H1(children='Explaining Reinforcement Learning through its World Model'),
                    html.H2(children='Interact with the World Model of Kaiser et al. (2019)'),
                    html.Div(id='description', children=[
                        dcc.Markdown('''
                    This user interface presents different modules to interact with the world model to generate explanations for the agent's policy.
                    The overall aim is to gain an understanding of the agent's policy and develop trust.

                    **Modules**:  

                    **&#9312** Fully play inside the world model and decide each action of the agent - no agent is trained.  
                    **&#9313** Both, the world model as well as agent are trained. The user can intervene single actions and observe the reactions.    
                    **&#9314** Again, both components are trained. The user can pause the game to display all possible actions from there.
                    '''),
                        html.Img(src='assets/game_descrip.png',height=300),
                        html.Div(id='', children=[
                            html.P(['Choose a Game:']),
                            dcc.Dropdown(id='dropdown_game',
                                         options=[
                                             {'label': 'Pong', 'value': 'pong'},
                                             {'label': 'Breakout', 'value': 'breakout'}
                                         ],
                                         value='pong', clearable=False)
                            ], className='slider_div'),
                        html.Div(id='speed_slider_div', children=[
                            html.P(['Speed of the game:']),
                            dcc.Slider(id='slider_speed',
                                       min=0, max=1, step=0.1, value=0.3,
                                       marks={0: 'Fast', 1: 'Slow'}),
                            html.Div(id='speed_slider_div_placeholder', style={'display':'none'})
                        ],
                                 className='slider_div')
                    ],
                             className='ponggame_cluster_descrip'),
                    html.Div(id='playing_pong', children=[
                        html.H3(children='Module: Focus on World Model'),
                        html.Div(children=[
                            html.Div(children=[
                                html.P(['Press button to play:']),
                                html.Button('Start to play',
                                            id='start_play_gamep',
                                            n_clicks=0)
                                ],
                                className='button-cluster'
                                ),
                            dcc.Markdown('''
                            Keys to play:
                            
                            **A**: Down, **D**: Up  
                            **N**: Perform NOOP,  
                            **R**: Key to reset env  
                            **X**: Reset simulated Env,  
                            **C**: Key to change between real-time-play and wait-for-pressed-key
                            ''',
                                className='key-descr-cluster'
                                )
                            ],
                            className='descrip-cluster'
                        ),
                        html.Video(id='playing_gamep',
                                   controls=True,
                                   height=396,
                                   width=720,
                                   className='video-cluster'),
                        html.Div(id='playing_gamep_descrip',
                                 className='video-descrip')
                    ],
                             className='ponggame_cluster'),
                    html.Div(id='pong_run_in_worldmodel', children=[
                        html.H3(children='Module: Focus on Agent'),
                        html.Div(children=[
                            html.Div(children=[
                                html.P(['Press button to run the game and intervene with single actions:']),
                                html.Button('Start to play',
                                            id='start_gamep_singlea',
                                            n_clicks=0)
                                ],
                                className='button-cluster'),
                            dcc.Markdown('''
                            Keys to intervene:
                            
                            **A**: Down, **D**: Up
                            **N**: Perform NOOP,  
                            **R**: Key to reset env  
                            **X**: Reset simulated Env,
                            ''',
                                className='key-descr-cluster'),
                            ],
                            className='descrip-cluster'
                        ),
                        html.Video(id='initial_game_videop',
                                   controls=True,
                                   height=396,
                                   width=720,
                                   className='video-cluster'),
                        html.Div(id='initial_game_videop_descrip',
                                 className='video-descrip')
                    ],
                        className='ponggame_cluster'),
                    html.Div(id='pong_run_in_worldmodel_showallactions', children=[
                        html.H3(
                            children='Module: Display Plot Threads'),
                        html.Div(children=[
                            html.Div(children=[
                                html.P(['Press button to run the game and pause to play all possible actions:']),
                                html.Button('Start to play',
                                            id='start_gamep_alla',
                                            n_clicks=0),
                                html.P(['Select length for plot threads']),
                                dcc.Slider(id='slider_plotlength',
                                           min=5, max=20, step=1, value=10,
                                           marks={
                                               5: '5', 10: '10', 15: '15', 20: '20'
                                           })
                                ],
                                className='button-cluster'),
                            dcc.Markdown('''
                            Keys to intervene:
                            
                            **Space**: Pause
                            ''',
                                         className='key-descr-cluster'),
                            ],
                            className='descrip-cluster'
                            # hier kommt Markdown hin für Description
                        ),
                        html.Video(id='game_videop_allactions',
                                   controls=True,
                                   height=264,
                                   width=1440,
                                   className='video-cluster'),
                        html.Div(id='game_videop_allactions_descrip',
                                 className='video-descrip')
                    ],
                            className='ponggame_cluster')
                ])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    '''
    method to switch between pages.
    worldmodel_kaiser: world model von kaiser et al. mit pong und breakout
    breakout: world model von ha et al. mit breakout
    carracing: world model von ha et al. mit carracing
    :param pathname: pagename
    :return: page
    '''
    if pathname=='/':
        return worldmodel_kaiser
    elif pathname=='/breakout':
        return breakout
    elif pathname=='/carracing':
        return carracing

@app.callback(Output('speed_slider_div_placeholder', 'children'),
              [Input('slider_speed', 'value')])
def input_speed_games(speed):
    '''
    adapt speed game of worldmodel_kaiser games
    :param speed: get speed
    :return: speed
    '''
    return speed

@app.callback([Output('playing_gamep', 'src'),
               Output('playing_gamep', 'height'),
               Output('playing_gamep', 'width'),
               Output('playing_gamep_descrip', 'children')],
              [Input('dropdown_game', 'value'),
               Input('start_play_gamep', 'n_clicks'),
               Input('speed_slider_div_placeholder', 'children')])
def pong_playing(value, buttonclick, speed_game):
    '''
    Module 1: Focus on world model - take over the role of the agent and define actions
    :param value: name of the game (either pong or breakout)
    :param buttonclick: start of the game by clicking the start button
    :param speed_game: game of the speed
    :return: video of the game including size of the game
    '''
    if buttonclick:
        total_reward = player.main(game_name=value, speed_game=speed_game)
        try:
            filename =[]
            filelist = os.listdir('gym-results')
            filelist.sort()
            for file in filelist:
                if file.endswith('1.mp4'): filename.append(file)
            videom = open('gym-results/' + filename[0], 'rb').read()
            encoded_video = base64.b64encode(videom).decode()
            src= 'data:video/mp4;base64,{}'.format(encoded_video)
            if total_reward == None:
                total_reward=0
            children = 'Game well played with a total reward of ', str(
                total_reward), '. The Video of your game is displayed here.'
        except:
            filename = "pong_playing.mp4"
            videom = open('assets/' + filename, 'rb').read()
            encoded_video = base64.b64encode(videom).decode()
            src = 'data:video/mp4;base64,{}'.format(encoded_video)
            children = 'Game well played with a total reward of ', str(
                total_reward), '. But the Video couldn\'t be saved.'
        height = 264
        width = 480
        return src, height, width, children
    else:
        filename = "initial_module1.mp4"
        videom = open('assets/'+ filename, 'rb').read()
        encoded_video = base64.b64encode(videom).decode()
        src = 'data:video/mp4;base64,{}'.format(encoded_video)
        height = 264
        width = 480
        children = ""
        return src, height, width, children


@app.callback([Output('initial_game_videop', 'src'),
               Output('initial_game_videop', 'height'),
               Output('initial_game_videop', 'width'),
               Output('initial_game_videop_descrip', 'children')],
              [Input('dropdown_game', 'value'),
               Input('start_gamep_singlea', 'n_clicks'),
               Input('speed_slider_div_placeholder', 'children')])
def pong_singleactions(value, buttonclick, speed_game):
    '''
    Module 2: Focus on agent - intervene the trajectory with single actions
    :param value: name of the game (either pong or breakout)
    :param buttonclick: start of the game by clicking the start button
    :param speed_game: game of the speed
    :return: video of the game including size of the game
    '''
    print('Speed: ', speed_game)
    if buttonclick:
        observations, fin_counter, total_reward = player.main(game_name=value, speed_game=speed_game, dry_run=True)
        filename='obs_video_pong_sa.webm'
        height = observations[0].shape[0]
        width = observations[0].shape[1]
        observations = observations[:, :, :, [2, 1, 0]]
        video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'vp80'), 10, frameSize=(width, height))
        for image in range(fin_counter):
            video.write(observations[image])
        video.release()
        videom =open(filename, 'rb').read()
        encoded_video= base64.b64encode(videom).decode()
        src = 'data:video/mp4;base64,{}'.format(encoded_video)
        children = "The Agent achieved a total reward of ", str(total_reward), ". The Video of your game is displayed here."
        height *= 2
        width *= 2
        return src, height, width, children
    else:
        filename = 'initial_module2.webm'
        videom = open('assets/' + filename, 'rb').read()
        encoded_video = base64.b64encode(videom).decode()
        src = 'data:video/mp4;base64,{}'.format(encoded_video)
        children = ""
        height = 264
        width = 480
        return src, height, width, children

@app.callback([Output('game_videop_allactions', 'src'),
               Output('game_videop_allactions', 'height'),
               Output('game_videop_allactions', 'width'),
               Output('game_videop_allactions_descrip', 'children')],
              [Input('dropdown_game', 'value'),
               Input('start_gamep_alla', 'n_clicks'),
               Input('slider_plotlength', 'value'),
               Input('speed_slider_div_placeholder', 'children')])
def pong_allactions(value, buttonclick, slider_length, speed_game):
    '''
    Module 3: Display all plot threads - pause within the game to display all possible actions and the following plot threads
    :param value: name of the game (either pong or breakout)
    :param buttonclick: start of the game by clicking the start button
    :param slider_length: length of the plot threads
    :param speed_game: game of the speed
    :return: video of the game including size of the game
    '''
    print('Speed: ', speed_game)
    if buttonclick and slider_length:
        observations, fin_counter, (trewardu, trewardn, trewardd), children= player.main(game_name=value, speed_game=speed_game, slider_length=slider_length, show_all_actions=True)
        filename='obs_video_pong_aa.webm'
        height = observations[0].shape[0]
        width = observations[0].shape[1]
        observations = observations[:, :, :, [2, 1, 0]]
        video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'vp80'), 10, frameSize=(width, height))
        for image in range(fin_counter):
            video.write(observations[image])
        video.release()
        videom =open(filename, 'rb').read()
        encoded_video= base64.b64encode(videom).decode()
        src = 'data:video/mp4;base64,{}'.format(encoded_video)
        if not children:
            children = "Each plot thread achieved the following reward in the game: " \
                       "Action up: ", trewardu, \
                       " Action noop: ", trewardn, \
                       " Action down: ", trewardd
        height *= 1.5
        width *= 1.5
        return src, height, width, children
    else:
        filename = 'initial_module3.webm'
        videom = open('assets/' + filename, 'rb').read()
        encoded_video = base64.b64encode(videom).decode()
        src = 'data:video/mp4;base64,{}'.format(encoded_video)
        children = ""
        height = 198
        width = 1080
        return src, height, width, children

@app.callback([Output('initial_game_videoc', 'src'),
               Output('initial_game_videoc', 'height'),
               Output('initial_game_videoc', 'width')],
              [Input('url', 'pathname'),
               Input('start_gamec', 'n_clicks')])
def carracing_allactions(page, buttonclick):
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
        src = 'data:video/webm;base64,{}'.format(encoded_video)
        height *= 1.5
        width *=1.5
        return src, height, width

@app.callback([Output('initial_game_videob', 'src'),
               Output('initial_game_videob', 'height'),
               Output('initial_game_videob', 'width')],
              [Input('url','pathname'),
               Input('start_gameb', 'n_clicks')])
def breakout_allactions(page, buttonclick):
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
        src = 'data:video/webm;base64,{}'.format(encoded_video)
        height *= 1.5
        width *= 1.5
        return src, height, width

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=1873)
