"""
@author: Fanchon Herman
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


dir_ = os.path.dirname(__file__)
df_ratings = pd.read_csv(os.path.join(dir_, "data/u.data"), sep='\t',
                         names=['userId', 'movieId', 'rating', 'timestamp'])


def get_matrix_Y(df):
    nb_users = df['userId'].unique().shape[0]
    nb_movies = df['movieId'].unique().shape[0]
    Y_matrix = np.zeros((nb_users, nb_movies))
    for row in df.itertuples(index=False):
        Y_matrix[row.userId - 1, row.movieId - 1] = row.rating
    return(Y_matrix)


def get_matrix_U_V(df):
    Y_matrix = get_matrix_Y(df)
    U, sigma, Vt = svds(Y_matrix)
    return(U, np.transpose(Vt))


def f_omega(U, V):
    Vt = np.transpose(V)
    somme = 0
    produit = np.dot(U, Vt)
    Y_matrix = get_matrix_Y(df_ratings)
    for i in np.arange(Y_matrix.shape[0]):
        for j in np.arange(Y_matrix.shape[1]):
            somme = somme + (Y_matrix[i, j] - produit[i, j])**2
    return(somme)


def altmin(U_0, T):
    U, V = get_matrix_U_V(df_ratings)
    for t in np.arange(1, T + 1):
        V[:, t] = np.argmin(f_omega(U[:, t - 1], V))
        U[:, t] = np.argmin(f_omega(U, V[:, t]))
    return(U[:, T], V[:, T])

# https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe\
# -string-entry-to-separate-rows


def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None and len(lst_cols) > 0 and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = pd.DataFrame({col: np.repeat(df[col].values, lens) for col in idx_cols}, index=idx).assign(**{col: np.concatenate(df.loc[lens>0, col].values) for col in lst_cols})

    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens == 0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res
