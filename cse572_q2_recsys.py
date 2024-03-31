import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import SVD, KNNBasic
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt

class RecSys:
	def __init__(self, file_name):
		self._loadData(file_name)
	
	def _loadData(self, filename):
		df = pd.read_csv(filename)
		df = df.drop(["timestamp"], axis=1)
		reader = Reader(rating_scale=(1,5))
		self.data = Dataset.load_from_df(df, reader)

	def runRecSys(self):
		# # part c,d
		pmf = self.PMF()
		print("Probabilistic Matrix Factorization", "MAE:", np.mean(pmf['test_mae']), "RMSE:", np.mean(pmf['test_rmse']))
		ubcf = self.UBCF()
		print("User based Collaborative Filtering", "MAE:", np.mean(ubcf['test_mae']), "RMSE:", np.mean(ubcf['test_rmse']))
		ibcf = self.IBCF()
		print("Item based Collaborative Filtering", "MAE:", np.mean(ibcf['test_mae']), "RMSE:", np.mean(ibcf['test_rmse']))

		# part e
		sim_metrics = ['msd','cosine','pearson']
		loss_ubcf_mae = []
		loss_ubcf_rmse = []
		loss_ibcf_mae = []
		loss_ibcf_rmse = []
		for sim_met in sim_metrics:
			ubcf = self.UBCF(sim_met)
			loss_ubcf_mae.append(np.mean(ubcf['test_mae']))
			loss_ubcf_rmse.append(np.mean(ubcf['test_rmse']))
			ibcf = self.IBCF(sim_met)
			loss_ibcf_mae.append(np.mean(ibcf['test_mae']))
			loss_ibcf_rmse.append(np.mean(ibcf['test_rmse']))

		plt.plot(loss_ubcf_mae, label="UBCF with MAE")
		plt.plot(loss_ubcf_rmse, label="UBCF with RMSE")
		plt.plot(loss_ibcf_mae, label="IBCF with MAE")
		plt.plot(loss_ibcf_rmse, label="IBCF with RMSE")
		plt.xticks([0, 1, 2], sim_metrics)
		plt.legend()
		plt.show()

		# part f
		loss_ubcf_mae = []
		loss_ubcf_rmse = []
		loss_ibcf_mae = []
		loss_ibcf_rmse = []
		xticks = []
		for i in range(1, 101, 10):
			xticks.append(i)
			ubcf = self.UBCF('msd', k=i)
			loss_ubcf_mae.append(np.mean(ubcf['test_mae']))
			loss_ubcf_rmse.append(np.mean(ubcf['test_rmse']))
			ibcf = self.IBCF('msd', k=i)
			loss_ibcf_mae.append(np.mean(ibcf['test_mae']))
			loss_ibcf_rmse.append(np.mean(ibcf['test_rmse']))

		plt.plot(loss_ubcf_mae, label="UBCF with MAE")
		plt.plot(loss_ubcf_rmse, label="UBCF with RMSE")
		plt.plot(loss_ibcf_mae, label="IBCF with MAE")
		plt.plot(loss_ibcf_rmse, label="IBCF with RMSE")
		print(loss_ubcf_mae)
		print(loss_ubcf_rmse)
		print(loss_ibcf_mae)
		print(loss_ibcf_rmse)
		plt.xticks(np.arange(len(xticks)), np.arange(1, 10*len(xticks), 10))
		plt.legend()
		plt.show()


	def UBCF(self, sim_metric='msd', k=40):
		# KNNBasic for user based collaborative filter
		# ref: https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic
		loss = cross_validate(KNNBasic(k, sim_options={'name': sim_metric}), self.data, measures=["MAE", "RMSE"], \
								cv=5)
		return loss

	def IBCF(self, sim_metric='msd', k=40):
		# KNNBasic for item based collaborative filter with setting user_based=false
		# ref: https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#similarity-measures-configuration
		loss = cross_validate(KNNBasic(k, sim_options={'name': sim_metric, 'user_based': False}), self.data, measures=["MAE", "RMSE"], \
								cv=5)
		return loss

	def PMF(self):
		# SVD with biased=false is PMF
		# ref: https://surprise.readthedocs.io/en/stable/matrix_factorization.html
		loss = cross_validate(SVD(biased=False, n_epochs=20), self.data, measures=["MAE", "RMSE"], \
								cv=5)
		return loss



rs = RecSys("recsys_data/ratings_small.csv")
rs.runRecSys()
