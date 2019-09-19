import pandas as pd
import numpy as np
import requests
# import json

df = pd.read_csv('pokemon-spawns.csv', index_col=False)
df = df.set_index('s2_id')
# df1 = pd.DataFrame(df,columns=['lat','lng'])
# print(df.iloc[0:100]['name'])

# pokemon=str(input('masukan nama pokemon: '))
# pokelow=pokemon.lower()
x = str(df.iloc[0]['name']).lower()
url = f'https://pokeapi.co/api/v2/pokemon/{x}/'
data = requests.get(url)

statistic = data.json()

print(statistic['types'][0]['type']['name'])
# print(statistic['sprites']['front_default'])

# {
# 'white',
# 'beige', 
# 'darkblue', 
# 'purple',
# 'lightgreen', 
# 'darkred', 
# 'lightblue', 
# 'blue', 
# 'cadetblue', 
# 'gray', 
# 'green', 
# 'orange', 
# 'darkgreen', 
# 'black', 'lightred', 'pink', 'lightgray', 'darkpurple', 'red'}