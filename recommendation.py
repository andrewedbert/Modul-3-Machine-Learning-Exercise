# 1. High ranked based
# 2. Content based filtering = cos function
# 3. Collaborative filtering

# cos similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cov = CountVectorizer()
data = [
    'Jakarta Bali Jakarta Surabaya Jakarta',
    'Bali Surabaya Bali Jakarta Surabaya Bali'
]
dataMx = cov.fit_transform(data)
print(cov.get_feature_names())
print(dataMx.toarray())

skorKesamaan =  cosine_similarity(dataMx)
print(skorKesamaan)