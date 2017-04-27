import numpy as np
import math

# Determines the weights that minimizes logistic loss
# function when assigning data to labels using objective 
# perturbation
# NOTE: only valid for 2 dimensions currently!
def objectivePerturbation(data, labels, eta, reg, t, eps, delta, w=None, mb=None):
	# Build initial weight matrix (0.5 in every entry)
	assert len(data) != 0
	n = data.shape[0]
	w = np.matrix([0.5 for _ in xrange(data.shape[1] + 1)]) if w == None else w
	d = w.shape[1]
	
	# Add constant to data
	data = np.hstack((data, np.ones((n, 1))))

	# Gradient Descent
	mb = mb if mb != None else n
	for _ in xrange(t):
		grad = np.zeros(w.shape)
		loss = 0
		
		# Sum
		# Will find gradient over all points, since we're finding total loss anyway
		for i in xrange(mb):
			i = np.random.randint(0, n)
			exp = math.e ** (-1.0 * labels.item(i, 0) * w.dot(data[i, :].T).item(0, 0))
			grad += exp / (1 + exp) * (-1.0 * labels[i] * data[i, :])
			loss += math.log(1 + exp)

		# Calculate noise vector b
		bDir = [np.random.uniform(0, 2 * math.pi) for _ in xrange(d - 1)]
		bNorm = np.random.gamma(data.shape[1], 2.0 / eps)
		b = []
		for i in xrange(len(bDir)):
			b.append(bNorm * math.sin(bDir[i]))
			bNorm *= math.cos(bDir[i])
			if i == len(bDir) - 1:
				b.append(bNorm)
		b = np.matrix(b)

		# Calculate gradient and loss
		grad = 1.0 * grad / mb + 2 * reg * w + b / mb
		loss = 1.0 * loss / mb
		
		# Update weight
		w = w - eta * grad
		# print w, loss

	return w, loss

