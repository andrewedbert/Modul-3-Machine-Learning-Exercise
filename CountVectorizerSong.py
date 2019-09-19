import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('Top_1_000_Songs_To_Hear_Before_You_Die.csv')
# print(df)
# print(df.isnull().sum())

df = df[['THEME', 'TITLE', 'ARTIST']]

def kombinasi(i):
    return str(i['THEME'])+'$'+str(i['ARTIST'])

df['x'] = df.apply(kombinasi, axis=1)
# print(df)
model = CountVectorizer(
    tokenizer = lambda i: i.split('$')
)

kategori = model.fit_transform(df['x'])
# print(model.get_feature_names())

cosScore = cosine_similarity(kategori)

suka = 'Someone Great'
indexSuka = df[df['TITLE'] == suka].index.values[0]
# print(indexSuka)

song = list(enumerate(cosScore[indexSuka]))
# print(song)

sortSong = sorted(song, key=lambda i: i[1], reverse=True)
# print(sortSong)

for i in sortSong[:10]:
    print(df.iloc[i[0]]['TITLE'], df.iloc[i[0]]['ARTIST'],i[1])