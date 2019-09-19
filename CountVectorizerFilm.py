import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('movie.csv')
# print(df.isnull().sum())

# feature: title & genres
df = df[['title','genres']]
# print(df.iloc[0])

model = CountVectorizer(
    tokenizer = lambda i: i.split('|')
)
gMatrix = model.fit_transform(df['genres'])

# print(model.get_feature_names())
# print(gMatrix.toarray()[0])

cosScore = cosine_similarity(gMatrix)
# print(cosScore)

# test
suka = 'Toy Story (1995)'
indexSuka = df[df['title'] == suka].index.values[0]
# print(indexSuka)

film = list(enumerate(cosScore[indexSuka]))
# print(film)

sortFilm = sorted(film, key=lambda i: i[1], reverse=True)
# print(sortFilm)

# 10 film yg cos score highest
# print(sortFilm[:10])

for i in sortFilm[:10]:
    print(df.iloc[i[0]]['title'], i[1])