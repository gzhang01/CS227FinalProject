import numpy as np
import random
import sys

# Generates datapoints for train and test set
# Parameters:
# 	n - number of datapoints desired
#	d - dimension of datapoints
#	w - weight vector
# 		* NOTE: |w| must equal d + 1 since |w| should allow
#				for a scalar term 
# Returns: (data, category)
#	data - n by d numpy matrix of datapoints
#	category - n by 1 numpy matrix of categories (-1 or 1)
#		* NOTE: category = 1 if above hyperplace; -1 if below
def generate(n, d, w):
	# Variables to hold generated data
	data = []
	category = []

	# Numpy array of weights
	weights = np.array(w)

	# Generating n data points
	for _ in xrange(n):
		# Get datapoint
		datum = []
		for _ in xrange(d):
			datum.append(random.uniform(0, 1))
		
		# Assign category
		tmp = list(datum)
		tmp.append(1)
		cat = 1 if np.array(tmp).dot(weights) > 0 else -1

		# Add to dataset
		data.append(datum)
		category.append([cat])

	# Return data as numpy matrices
	return np.matrix(data), np.matrix(category)


# Diagnosing function. Run with "python generate.py"
def diagnose():
	n = 25
	data, cat = generate(n, 2, [-1, 1, 0])
	for i in xrange(n):
		print data[i], cat[i]

if __name__ == "__main__":
	if len(sys.argv) == 2:
		diagnose()

	# Generate data
	elif len(sys.argv) == 5:
		# Get inputs from command line
		n = int(sys.argv[1])
		d = int(sys.argv[2])
		w = map(int, sys.argv[3].replace("[", "").replace("]", "").split(","))
		outfile = sys.argv[4]

		# Generate data
		data, cat = generate(n, d, w)

		# Write to outfile (NOTE: overwrites file!)
		with open("data/" + outfile, "w") as f:
			for i in xrange(n):
				for j in xrange(d):
					f.write("{0}, ".format(data[i, j]))
				f.write("{0}\n".format(cat[i, 0]))

	# Generate usage message
	else:
		print "Usage: python generate.py [n] [d] [w] [outfile]"
		print "Usage: python generate.py diagnose"
