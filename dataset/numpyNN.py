import random
import numpy as np

def linearData(n_sample=400):
	theta = np.random.rand() * 2 * np.pi
	w_star = np.array([[np.cos(theta), np.sin(theta)]])
	margin = 0.1
	noise = 0.1
	#  create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (X @ w_star.T) > 0
	label = label.astype(float)
	# create margin
	idx = (label * (X @ w_star.T)) < margin
	X = X + margin * ((idx * label) @ w_star)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def XORData(n_sample=400):
	margin = 0.1
	noise = 0.1
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (X[:, 0] * X[:, 1]) > 0
	label = label.astype(float).reshape((-1, 1))
	# create margin
	pos_flag = X >= 0
	X = X + 0.5 * margin * pos_flag
	X = X - 0.5 * margin * (~pos_flag)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def circleData(n_sample=400):
	noise = 0.05
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	dist = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
	label = dist <= 0.5
	label = label.astype(float).reshape((-1, 1))
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def sinusoidData(n_sample=400):
	noise = 0.05
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (np.sin(np.sum(X, axis=- 1) * 2 * np.pi) > 0)
	label = label.astype(float).reshape((-1, 1))
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def swissrollData(n_sample=400):
	noise = 0.05
	nHalf = int(n_sample / 2)
	# create data
	t = np.random.rand(nHalf, 1)
	x1 = t * np.cos(2 * np.pi * t * 2)
	y1 = t * np.sin(2 * np.pi * t * 2)
	t = np.random.rand(n_sample - nHalf, 1)
	x2 = (-t) * np.cos(2 * np.pi * t * 2)
	y2 = (-t) * np.sin(2 * np.pi * t * 2)
	xy1 = np.concatenate([x1, y1], axis=1)
	xy2 = np.concatenate([x2, y2], axis=1)
	X = np.concatenate([xy1, xy2], axis=0)
	label = np.concatenate([np.ones((nHalf, 1)), np.zeros((n_sample - nHalf, 1))], axis=0)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def sample_data(data_name='circle', nTrain=200, nTest=200, random_seed=0,):
	"""
	Data generation function
	:param data_name: linear-separable, XOR, circle, sinusoid, swiss-roll
	:return:
	"""
	random.seed(random_seed)
	np.random.seed(random_seed)
	n_sample = nTrain + nTest
	if data_name == 'linear-separable':
		X, label = linearData(n_sample)
	elif data_name == 'XOR':
		X, label = XORData(n_sample)
	elif data_name == 'circle':
		X, label = circleData(n_sample)
	elif data_name == 'sinusoid':
		X, label = sinusoidData(n_sample)
	elif data_name == 'swiss-roll':
		X, label = swissrollData(n_sample)
	else:
		raise NotImplementedError


	indices = np.random.permutation(n_sample)
	train_idx, test_idx = indices[:nTrain], indices[nTrain:]
	x_train = X[train_idx]
	y_train = label[train_idx]
	x_test = X[test_idx]
	y_test = label[test_idx]
	return x_train, y_train, x_test, y_test
