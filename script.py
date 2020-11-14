"""
@author: Fanchon Herman
"""

import os
import pandas as pd
from datetime import datetime
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import count_nonzero
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from source import (f_omega, get_matrix_Y, get_matrix_U_V, altmin, explode)
sns.set()


###########################
# data import
###########################

dir_ = os.path.dirname(__file__)

df_ratings = pd.read_csv(os.path.join(dir_, "data/u.data"), sep='\t',
                         names=['userId', 'movieId', 'rating', 'timestamp'])

df_movies = pd.read_csv(os.path.join(dir_, "data/movies.csv"))
print(df_movies.head())

df_links = pd.read_csv(os.path.join(dir_, "data/links.csv"))
print(df_links.head())

df_tags = pd.read_csv(os.path.join(dir_, "data/tags.csv"))
print(df_tags.head())

########################
# some analysies
########################

df_ratings.isnull().any()
df_movies.isnull().any()
df_links.isnull().any()
df_links.dropna(subset=['tmdbId'], inplace=True)
df_tags.isnull().any()

df_ratings.shape
df_movies.shape
df_links.shape
df_tags.shape

# drama type movies
drama_movies = df_movies['genres'].str.contains('Drama')
df_movies[drama_movies].head()

# movie ID with tag 'funny'
tag_funny = df_tags['tag'].str.contains('funny')
df_tags[tag_funny].head()

# concatenate two dataframes "df_movies" and "df_ratings"
df_movies_ratings = df_movies.merge(df_ratings, on='movieId', how='inner')
df_movies_ratings.head(3)
del df_movies_ratings['timestamp']
df_movies_ratings.columns

# movie genres
df_movies.genres = df_movies.genres.str.split('|')
df_movies = explode(df_movies, ['genres'])
df_movies.head()

# movie genre distributions
plt.figure(figsize=(10, 8))
rat = collections.Counter(df_movies['genres'])
plt.bar(rat.keys(), rat.values(), width=0.7)
plt.xticks(ticks=np.arange(20), labels=('Adventure', 'Animation',
           'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action',
		   'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War',
		   'Musical', 'Documentary', 'IMAX', 'Western', 'Film-Noir',
		   '(no genres listed)'), rotation=90)
plt.xlabel('genres')
plt.ylabel('nombre de films')
plt.savefig('genres.pdf')

#######################
# Time structure
#######################

df_ratings['date'] = ""
for i in range(len(df_ratings['timestamp'])):
    df_ratings['date'].values[i] = datetime.fromtimestamp(
        df_ratings['timestamp'].values[i])
time_improved = pd.to_datetime(df_ratings['date'], format='%Y/%m/%d %H:%M:%s')
df_ratings['year'] = time_improved.dt.strftime('%Y')
df_ratings['month'] = time_improved.dt.strftime('%m')
month_count = df_ratings[['year', 'month',
                         'rating']].groupby(['year', 'month']).count()
month_count = month_count.reset_index()

##############################
# Data visualization
##############################

print(df_ratings.head())
print(df_ratings.shape)

#############################
# Ratings distributions
############################

# distribution

rat = collections.Counter(df_ratings['rating'])
plt.bar(rat.keys(), rat.values(), width=0.7, color=['orange', 'violet', 'c',
        'lightcoral', 'm'])
plt.xticks(ticks=[1, 2, 3, 4, 5],
           labels=('1', '2', '3', '4',  '5'))
plt.xlabel('note')
plt.ylabel('nombre')
plt.savefig('distrib_notes.pdf')

# proportion

df_ratings.groupby(['rating']).size() / df_ratings['rating'].count()

##################################
# Number of ratings posted per month
##################################

plt.figure()
month_count['rating'].plot(style='ro-')
plt.ylabel('Nombre de notes')
plt.xticks(np.arange(7+1), ('Sep', 'Oct', 'Nov', 'Dec', 'Janv 98', 'Feb',
           'Mar', 'Apr'))
plt.title('Nombre de notes par mois')
plt.savefig('notes_mois.pdf')

########################################################
# creation of the Y matrix containing all ratings
# columns: movies , rows: users
########################################################

Y_matrix = get_matrix_Y(df_ratings)
print(Y_matrix)


##############################
# sparsity of the Y matrix
##############################

n_users = Y_matrix.shape[0]
n_movies = Y_matrix.shape[1]
sparsity = 1.0 - count_nonzero(Y_matrix) / Y_matrix.size
print(sparsity*100)

plt.figure()
plt.spy(Y_matrix, precision=0.1)
plt.ylabel('user')
plt.xlabel('movie')
plt.savefig('sparse.pdf')

###########
# SVD
###########

U, V = get_matrix_U_V(df_ratings)
Vt = np.transpose(V)
ortho = np.dot(np.transpose(U), U)  # orthogonality check

# function F_omega
algo = f_omega(U, V)
print(algo)

# function altmin (doesn't work)
T = 5
U_0 = np.eye(U.shape[0], 1)
# 	altmin(U_0, T)

# train test
data = df_ratings.copy()
del data['timestamp']
del data['date']
del data['year']
del data['month']


train_set, test_set = train_test_split(data, test_size=.2)
# algorithm = TruncatedSVD()
# algorithm.fit(train_set)
# predictions = algorithm.test(test_set)
