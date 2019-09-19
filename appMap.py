from flask import Flask, render_template
import folium
import pandas as pd
import numpy as np
import requests

app = Flask(__name__)

@app.route('/')
def home():
    dc = {
        'normal':'gray',
        'bug':'darkgreen',
        'grass':'green',
        'flying':'lightgray',
        'dark':'black',
        'dragon':'beige',
        'electric':'lightgreen',
        'fire':'red',
        'water':'blue',
        'poison':'purple',
        'ghost':'darkpurple',
        'steel':'cadetblue',
        'ground':'lightred',
        'ice':'white',
        'fairy':'pink',
        'psychic':'dark',
        'rock':'orange',
        'fighting':'darkred'
    }
    df = pd.read_csv('pokemon-spawns.csv', index_col=False)
    map = folium.Map(
        location=[df.iloc[0]['lat'],df.iloc[0]['lng']],
        zoom_start=12
    )
    for i in range(len(df.iloc[100:150])):
        x = str(df.iloc[i]['name']).lower()
        url = f'https://pokeapi.co/api/v2/pokemon/{x}/'
        data = requests.get(url)
        if data.status_code == 200:
            statistic = data.json()
            y = statistic['sprites']['front_default']
            z = statistic['types'][0]['type']['name']
            folium.Marker(
                [(df.iloc[i]['lat']),(df.iloc[i]['lng'])],
                popup=df.iloc[i]['name'],
                tooltip=f'<img width="100px" height="100px" src={y}>',
                icon=folium.Icon(color=str(dc[z]))
            ).add_to(map)
    map.save('templates/map.html')
    return render_template('home.html')

@app.route('/map')
def map():
    return render_template('map.html')

if __name__ == '__main__':
    app.run(debug=True)