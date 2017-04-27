import numpy as np
import math

# Determines the weights that minimizes logistic loss
# function when assigning data to labels
def logisticRegression(data, labels, eta, reg, t, w=None, mb=None):
	# Build initial weight matrix (0.5 in every entry)
	assert len(data) != 0
	n = data.shape[0]
	w = np.matrix([0.5 for _ in xrange(data.shape[1] + 1)]) if w == None else w
	
	# Add constant to data
	data = np.hstack((data, np.ones((n, 1))))

	# Gradient Descent
	old_loss = 10000000
	mb = mb if mb != None else n
	for _ in xrange(t):
		grad = np.zeros(w.shape)
		loss = 0
		
		# Sum
		# Will find gradient over all points, since we're finding total loss anyway
		for _ in xrange(mb):
			i = np.random.randint(0, n)
			exp = math.e ** (-1.0 * labels.item(i, 0) * w.dot(data[i, :].T).item(0, 0))
			grad += exp / (1 + exp) * (-1.0 * labels[i] * data[i, :])
			loss += math.log(1 + exp)
		
		# Calculate gradient and loss
		grad = 1.0 * grad / mb + 2 * reg * w
		loss = 1.0 * loss / mb
		
		# Update weight
		w = w - eta * grad
		# print w, loss

		# Test for convergence
		if abs(old_loss - loss) < 0.00001:
			break

		# Reset old loss
		old_loss = loss

	return w, loss
