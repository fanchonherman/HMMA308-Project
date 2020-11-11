# -*- coding: utf-8 -*-
"""
@author: Fanchon Herman
"""

#############################
# import des packages
#############################
import os
import pandas as pd
import collections
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np
from numpy import count_nonzero
from scipy.sparse.linalg import svds

sns.set()

###########################
# import des données
###########################

dir_ = os.path.dirname(__file__)
df_ratings = pd.read_csv(os.path.join(dir_, "data/u.data"), sep='\t',
                         names=['userId', 'movieId', 'rating', 'timestamp'])

#######################
# Structure du temps
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
# Visualisation des données
##############################

print(df_ratings.head())
print(df_ratings.shape)

#############################
# Distributions des notes
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
# Nombre de notes mises par mois
##################################

plt.figure()
month_count['rating'].plot(style='ro-')
plt.ylabel('Nombre de notes')
plt.title('Nombre de notes par mois')
plt.savefig('notes_mois.pdf')


########################################################
# creation de la matrice Y contenant toute les notes
# colonnes : films , lignes : utilisateurs
########################################################

print(df_ratings.shape[0])
nb_users = df_ratings['userId'].unique().shape[0]
print(nb_users)
nb_movies = df_ratings['movieId'].unique().shape[0]
print(nb_movies)
Y_matrix = np.zeros((nb_users, nb_movies))
print(Y_matrix.shape)
for row in df_ratings.itertuples(index=False):
    Y_matrix[row.userId - 1, row.movieId - 1] = row.rating

print(Y_matrix)

##############################
# sparcité de la matrice Y
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

#df_ratings.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
#test = df_ratings.drop(['timestamp', 'date'], axis=1)
#test = test.values
#df_ratings['date'] = df_ratings['date'].apply(str)
#df_ratings['date'] = list(map(lambda x: pd.to_datetime(x), df_ratings['date']))
#test = test.astype("float")
#U, sigma, Vp = np.linalg.svd(test, full_matrices=False)
#U, sigma, Vp = np.linalg.svd(Y_matrix, full_matrices=False)
#print('U=', U, 'sigma=', sigma, 'Vp=', Vp)

U, sigma, Vt = svds(Y_matrix)
print('U=', U, 'sigma=', sigma, 'Vt=', Vt)
ortho = np.dot(np.transpose(U), U)  # verification de orthogonalite


def F_omega(Y, U, Vt):
    somme = 0
    produit = np.dot(U, Vt)
    for i in np.arange(Y_matrix.shape[0]):
        for j in np.arange(Y_matrix.shape[1]):
            somme = somme + (Y[i, j] - produit[i, j])**2
    return(somme)


F_omega(Y_matrix, U, Vt)


#def Altmin(omega, U_0, T):
#    for t in np.arange(1, T+1):
#		Vt = np.argmin(F_omega(Y, , V))
		


	