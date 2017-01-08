from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def PCA():
	data = pd.read_csv(filepath_or_buffer='iris.data',  header=None,  sep=',')

	data.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
	data.dropna(how="all", inplace=True)

	data.tail()
	feature_matrix = data.ix[:,0:4].values
	labels = data.ix[:, 4].values


	mean_seplen = np.mean(feature_matrix[:, 0])
	mean_sepwid = np.mean(feature_matrix[:, 1])
	mean_petlen = np.mean(feature_matrix[:, 2])
	mean_petwid = np.mean(feature_matrix[:, 3])
	mean_vector = np.array([[mean_seplen], [mean_sepwid], [mean_petlen], [mean_petwid]])



	covMatrix = np.cov([feature_matrix[:,0], feature_matrix[:,1],feature_matrix[:,2], feature_matrix[:, 3]])

	print covMatrix


	eigVal, eigVec = np.linalg.eig(covMatrix)
	eigPairs = [(np.abs(eigVal[i]), eigVec[:,i]) for i in range(len(eigVal))]
	eigPairs.sort(key=lambda x: x[0], reverse=True)
	for i in eigPairs:
		print (i[0])



 	matrix = np.hstack((eigPairs[0][1].reshape(4,1), eigPairs[1][1].reshape(4,1)))
 	transformed = matrix.T.dot(feature_matrix.T)
 	print transformed.T.shape
	plt.scatter(transformed[0, 0:50], transformed[1, 0:50], color='blue')
	plt.scatter(transformed[0, 50:150], transformed[1, 50:150], color='red')
	plt.show()


def main():
	PCA()



if __name__ == "__main__":
	main()
