import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

class KMeans:
	def __init__(self, cluster_no, data, labels):
		self.k = cluster_no
		self.data = data
		self.labels = labels.reshape((1, labels.size))
		self.pred_labels = np.zeros((1, self.data.shape[0]), dtype=int)
		self.error_list = []
		self.acc_list = []

	def _assignToCentroid(self, centroids, similarity):
		dist = np.zeros((self.k, self.data.shape[0]))
		for j in range(len(centroids)):
			centroid = centroids[j]
			if(similarity == "euclidean"):
				dist[j] = self.euclidean(centroid)
			if(similarity == "cosine"):
				# cosine_dist = 1 - cosine_similarity
				dist[j] = 1 - self.cosine(centroid)
			if(similarity == "jarcard"):
				# jarcard_dist = 1 - jarcard_similarity
				dist[j] = 1 - self.jarcard(centroid)
		self.pred_labels = np.argmin(dist, axis=0).reshape((1, self.data.shape[0]))

	def _computeNewCentroids(self, tmp_data):
		return np.mean(tmp_data, axis = 0)

	def _majorityVoteLabel(self):
		for i in range(self.k):
			idx = self.pred_labels == i
			if(np.sum(idx) == 0):
				continue
			self.pred_labels[idx] = sp.stats.mode(self.labels[idx])[0]

	def runKMeans(self, similarity):
		# choose random centroids
		centroid = []
		prev_centroid = []
		centroid_idx = np.random.randint(data.shape[0], size=self.k)
		for i in range(self.k):
			centroid.append(data[centroid_idx[i]])
		
		n_iter = 100
		prev_sse = np.inf
		for j in range(n_iter):
			# assign each data point to a centroid
			self._assignToCentroid(centroid, similarity)
			prev_centroid = centroid.copy()
			centroid = []
			# compute new centroids
			for i in range(self.k):
				# print(self.pred_labels)
				idx = self.pred_labels[0] == i
				if(np.sum(idx) == 0):
					continue
				tmp_data = self.data[idx]
				# print(tmp_data)
				centroid.append(self._computeNewCentroids(tmp_data))
			
			self._majorityVoteLabel()
			# print(j, self.computeAcc())

			# # check if new sse is greater than previous sse
			# new_sse = self.computeSSE()
			# if(new_sse > prev_sse):
			# 	break
			# prev_sse = new_sse

			# # check if change in centroid position
			# tot_dist_change = 0
			# for i in range(self.k):
			# 	tot_dist_change += np.linalg.norm(prev_centroid[i] - centroid[i])

			# if(tot_dist_change == 0):
			# 	break

			
			self.error_list.append(self.computeSSE())
			self.acc_list.append(self.computeAcc())

	def computeSSE(self):
		return np.sum((self.pred_labels - self.labels)**2)
	
	def computeAcc(self):
		return (np.sum(self.pred_labels == self.labels)/self.pred_labels.size)*100

	def euclidean(self, centroid):
		return np.linalg.norm(self.data - centroid, axis = 1)

	def cosine(self, centroid):
		return np.dot(self.data, centroid)/(np.linalg.norm(self.data, axis = 1) * np.linalg.norm(centroid))

	def jarcard(self, centroid):
		return np.sum(np.minimum(self.data, centroid), axis = 1)/np.sum(np.maximum(self.data, centroid), axis = 1)

# read data
df_data = pd.read_csv("knn_data/data.csv", header=None)
data = df_data.to_numpy()

# read labels
df_labels = pd.read_csv("knn_data/label.csv", header=None)
labels = df_labels.to_numpy()

knn = KMeans(10, data, labels)
knn.runKMeans("euclidean")
print("euclidean")
print("Final SSE:", knn.computeSSE())
e_sse, e_acc = knn.error_list, knn.acc_list

knn = KMeans(10, data, labels)
knn.runKMeans("cosine")
print("cosine")
print("Final SSE:", knn.computeSSE())
c_sse, c_acc = knn.error_list, knn.acc_list

knn = KMeans(10, data, labels)
knn.runKMeans("jarcard")
print("jarcard")
print("Final SSE:", knn.computeSSE())
j_sse, j_acc = knn.error_list, knn.acc_list

plt.figure(1)
plt.plot(e_sse, label="euclidean")
plt.plot(c_sse, label="cosine")
plt.plot(j_sse, label="jarcard")
plt.legend()
plt.figure(2)
plt.plot(e_acc, label="euclidean")
plt.plot(c_acc, label="cosine")
plt.plot(j_acc, label="jarcard")
plt.legend()
plt.show()

