from surprise import Reader, Dataset, SVD, KNNBasic
from surprise.model_selection import cross_validate
import pandas as pd

reader = Reader()
ratings = pd.read_csv('ratings_small.csv')

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#data.split(n_folds=5)
#'MSD', 'Cosine', 'Pearson'
#'user_based', 'item_based'
sim_options = {'name': 'MSD', 'item_based' : True}
#algo = KNNBasic(sim_options=sim_options)
algo = KNNBasic(k=10, sim_options=sim_options)
#svd = SVD(biased=False)
#output = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
output = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#print(output)