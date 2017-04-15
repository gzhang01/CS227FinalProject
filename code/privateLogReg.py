import numpy as np
import math
import random

# Determines the weights that minimizes logistic loss
# function when assigning data to labels with DP
def privateLogReg(data, labels, eta, reg, t, eps, delta, c):
	# Build initial weight matrix (0.5 in every entry)
	assert len(data) != 0
	n = data.shape[0]
	w = np.matrix([0.5 for _ in xrange(data.shape[1] + 1)])
	
	# Add constant to data
	data = np.hstack((data, np.ones((n, 1))))

	# Sigma parameter
	sigma2 = math.log(1.0 * n / delta) * math.log(1.0 / delta) / (eps ** 2) / c
	print sigma2

	# Gradient Descent
	for _ in xrange(t):
		grad = np.zeros(w.shape)
		loss = 0
		
		# Find loss
		for i in xrange(n):
			exp = math.e ** (-1.0 * labels.item(i, 0) * w.dot(data[i, :].T).item(0, 0))
			loss += math.log(1 + exp)
		loss = 1.0 * loss / n + 2 * reg * (w.T * w).item(0, 0)

		# Calculate gradient
		index = random.randint(0, n - 1)
		exp = math.e ** (-1.0 * labels.item(index, 0) * w.dot(data[index, :].T).item(0, 0))
		grad += exp / (1 + exp) * (-1.0 * labels[index] * data[index, :])
		noise = [0 for _ in xrange(grad.shape[1])]
		for i in xrange(len(noise)):
			noise[i] += np.random.normal(scale=(sigma2 ** (0.5)))
		grad += np.matrix(noise)

		
		# Update weight
		w = w - eta * grad
		print w, loss

	return w






