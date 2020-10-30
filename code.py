################################
# Data Analysis on 100K dataset
################################


######################
# Packages needded
#####################

import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import collections

#################################
# Download datasets
#################################

dir_ = os.path.dirname(__file__)

df_ratings = pd.read_csv(os.path.join(dir_, "data/ratings.csv"))
print(df_ratings.head())

df_movies = pd.read_csv(os.path.join(dir_, "data/movies.csv"))
print(df_movies.head())

df_links = pd.read_csv(os.path.join(dir_, "data/links.csv"))
print(df_links.head())

df_tags = pd.read_csv(os.path.join(dir_, "data/tags.csv"))
print(df_tags.head())

df_ratings['timestamp'] = df_ratings['timestamp'].apply(datetime.fromtimestamp)
df_ratings['year'] = df_ratings['timestamp'].dt.year
df_ratings['month'] = df_ratings['timestamp'].dt.month
df_ratings = df_ratings.sort_values('timestamp')

df_tags['timestamp'] = df_tags['timestamp'].apply(datetime.fromtimestamp)
df_tags['year'] = df_tags['timestamp'].dt.year
df_tags['month'] = df_tags['timestamp'].dt.month
df_tags = df_tags.sort_values('timestamp')


# Ratings distributions

# =============================================================================
# df_ratings['rating'].hist(freq=True, grid=True, xlabelsize=12, ylabelsize=12)
# plt.xlabel("Life Expectancy", fontsize=15)
# plt.ylabel("Frequency",fontsize=15)
# plt.xlim([22.0,90.0])
# =============================================================================

rat = collections.Counter(df_ratings['rating'])
plt.bar(rat.keys(), rat.values(), width=0.4, color=['b', 'g', 'r', 'c', 'm',
		'y', 'k', 'lightcoral', 'violet', 'orange'])
plt.xticks(ticks=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
		   labels=('0.5', '1', '1.5', '2', '2.5', '3', '3.5','4', '4.5', '5'))
plt.xlabel('rating')
plt.ylabel('count')

# SVD

df_ratings.dropna(inplace=True)
test = df_ratings.drop('timestamp', axis=1)
test = test.values # convert a pandas as a numpy array
test = test.astype("float")
U, sigma, Vp = np.linalg.svd(test, full_matrices=False)




