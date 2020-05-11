import dash
import dash_html_components as html
import dash_core_components as dcc
import base64

video_path = '/home/dakaun/PycharmProjects/world_model/WorldModelsExperiments/explain/obs_sequence.mp4'

app = dash.Dash(__name__)
server = app.server

video = open(video_path,'rb').read()
b64 = base64.b64encode(video)

app.layout = html.Div(children=[
  html.Video(
    controls= True,
    scr="https://www.w3schools.com/html/mov_bbb.mp4",
    autoPlay= True)
])



if __name__ == '__main__':
  app.run_server(debug=True, host='0.0.0.0', port=1880)

'''
video
def embed_mp4(filename,video):",
    "  \"\"\"Embeds an mp4 file in the notebook.\"\"\"\n",
    "  video = open(filename,'rb').read()\n",
    "  b64 = base64.b64encode(video)\n",
    "  tag = ,
    "  <video width=\"640\" height=\"480\" controls>\n",
    "    <source src=\"data:video/mp4;base64,{0}\" type=\"video/mp4\">\n",
    "  Your browser does not support the video tag.\n",
    "  </video>.format(b64.decode())",
    "\n",
    "  return IPython.display.HTML(tag)"
    (1)
    app.layout = html.Div(children=[
  html.H1(children='Breakout Worldmodel'),
  html.Video(src='data:video/mp4;base64,{} type=\'video/mp4\''.format(b64.decode()))
  )]
    nicht funktioniert - kein Video angeszeigt
    (2)
    app.layout = html.Div(children=[
  html.Video(scr="https://www.w3schools.com/html/mov_bbb.mp4")
])
  (3)
  
'''