import dash
import dash_html_components as html
import dash_core_components as dcc
import base64
import flask
import os
import gym
from PIL import Image

# video_path = '/home/student/Dropbox/MA/assets/obs_sequence_all.mp4'

app = dash.Dash(__name__)
server = app.server

colors = {
    'text': '#111111'
}

# video = open(video_path, 'rb').read()
# encoded_video = base64.b64encode(video)

app.layout = html.Div(style={
    'textAlign':'center'
    },
    children=[
        html.H1(children='Breakout Word Model',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }),
        html.Div(children='Dashboard to display the world model of breakout.',
                 style={
                     'textAlign': 'center',
                     'color': colors['text']
                 }),
        html.Button('Press to start Breakout',id='start_game', n_clicks=0),
        html.Img(id='game_img')

        #display_envimg(),
        # html.Video(id='videoexample',
        #         controls=True,
        #         src='data:video/mp4;base64,{}'.format(encoded_video.decode()),
        #         style={
        #             'textAlign': 'center'
        #         })
    ])
@app.callback(
    dash.dependencies.Output('game_img', 'src'),
    [dash.dependencies.Input('start_game', 'n_clicks')]
)
def run_game(buttonclick):
    env = gym.make('Breakout-v0')
    if buttonclick:
        obs = env.reset()
        img = Image.fromarray(obs)
        return img


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=1880)
