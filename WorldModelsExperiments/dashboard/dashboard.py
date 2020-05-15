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
    return obs_sequence, seq_counter


def generate_video(sequence, filename, counter):
    print('generating video..')
    print(filename)
    print(counter)
    with imageio.get_writer(filename, mode='I', macro_block_size=None) as video:
        for image in range(counter):
            video.append_data(sequence[image])
    print('done generating video')


app = dash.Dash(__name__)
server = app.server

colors = {
    'background-color': 'LightGray'
}

app.layout = html.Div(id='header1',
                      style={
                          'textAlign': 'center',
                          'background-color': 'LightGray'
                      },
                      children=[
                          html.H1(children='Breakout Word Model',
                                  style={
                                      'textAlign': 'center'
                                  }),
                          html.H3(children='Dashboard to display the world model of breakout.',
                                  style={
                                      'textAlign': 'center'
                                  }),
                          html.Div(id='subbodytest', children=[
                              html.H5(['Press the Button to run Breakout.']),
                              html.Button('Start Breakout',
                                      id='start_game',
                                      n_clicks=0,
                                      style={
                                          'textAlign': 'center'
                                      }),
                          ]),
                          html.Video(id='initial_game_video',
                                     controls=True,
                                     style={
                                         'textAlign': 'center'
                                     })

                          # display_video()
                          # html.Video(id='videoexample',
                          #         controls=True,
                          #         src='data:video/mp4;base64,{}'.format(encoded_video.decode()),
                          #         style={
                          #             'textAlign': 'center'
                          #         })
                      ])


@app.callback(dash.dependencies.Output('initial_game_video', 'src'),
              [dash.dependencies.Input('start_game', 'n_clicks')])
def start_game(buttonclick):
    if buttonclick:
        print('start playing game')
        obs_sequence, seq_counter = play_game(model)
        print(obs_sequence.shape)
        print('ending game')
        init_obs_seq_filename = 'obs.mp4'
        print('generating video..')
        # np.savez_compressed('test.npz', obs=obs_sequence)
        print(seq_counter)
        # with imageio.get_writer(init_obs_seq_filename, mode='I', macro_block_size=None) as video:
        #    for image in range(10):
        #        video.append_data(obs_sequence[image])

        print('done generating video')
        # generate_video(obs_sequence, init_obs_seq_filename, seq_counter)
        print('loading video')
        # filename = ''
        video = open('/home/student/Dropbox/MA/assets/obs_sequence_all.mp4', 'rb').read()
        encoded_video = base64.b64encode(video)
        print('send video to dashboard')
        return 'data:video/mp4;base64,{}'.format(encoded_video.decode())


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=1885)
