import numpy as np
import matplotlib.pyplot as plt
from generate import generate
from privateLogReg import privateLogReg
import math


# Generate train data
# n = 100
# w_real = [2, 1, -1.7]
# noisy = False
# data, labels = generate(n, 2, w_real, noisy)

# # Write conditions
# with open("../data/cAnalysis.txt", "a") as f:
# 	f.write("n: {0}, w: {1}, noisy: {2}\n".format(n, w_real, noisy))
# 	f.write("c, loss, numCorrect, total, accuracy\n")

# Testing for values of c^2
c2 = [1.0/400, 1.0/200, 1.0/100, 1.0/50, 1.0/25, 1.0/10, 1.0/2, 1, 2, 4, 8, 16, 32]
losses = []
numCorrect = []

# # Starting weight
# w = np.matrix([0, 1, -0.5])

# Generate test data
nTest = 5000
# testData, testLabels = generate(nTest, 2, w_real, noisy)
# testData = np.hstack((testData, np.ones((nTest, 1))))

# # Run tests
# for i in xrange(len(c2)):
# 	print "Testing: {0}".format(c2[i])
# 	w, loss = privateLogReg(data, labels, eta=0.2, reg=0.0005, t=n * n, eps=1, delta=0.1, c=c2[i], w=w, mb=20)
# 	losses.append(loss)
# 	numCorrect.append(0)

# 	for j in xrange(len(testData)):
# 		if testLabels.item(j, 0) * w.dot(testData[j, :].T).item(0, 0) >= 0:
# 			numCorrect[i] += 1

# 	with open("../data/cAnalysis.txt", "a") as f:
# 		f.write("{0}, {1}, {2}, {3}, {4}\n".format(c2[i], losses[i], numCorrect[i], nTest, 1.0 * numCorrect[i] / nTest))

with open("../data/cAnalysis.txt", "r") as f:
	lines = f.readlines()

lines = lines[18:31]

for line in lines:
	a = line[:-1].split(",")
	losses.append(float(a[1]))
	numCorrect.append(int(a[2]))

c2 = map(math.log, c2)

plt.plot(c2, losses)
plt.suptitle('Loss vs. log c^2', size=16)
plt.xlabel('log c^2')
plt.ylabel('Loss')
plt.savefig('../data/cAnalysisLoss.png')
plt.show()

plt.plot(c2, map(lambda x: 1.0 * x / nTest, numCorrect))
plt.suptitle('Accuracy vs. log c^2', size=16)
plt.xlabel('log c^2')
plt.ylabel('Accuracy')
plt.savefig('../data/cAnalysisAccuracy.png')
plt.show()



