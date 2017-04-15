import numpy as np
from generate import generate
import matplotlib.pyplot as plt
from nonPrivateLogReg import logisticRegression
from privateLogReg import privateLogReg

# Extract data from train set
# data = []
# labels = []
# with open("data/train.txt", "r") as f:
# 	for line in f:
# 		tmp = line.strip().split(", ")
# 		data.append([float(tmp[i]) for i in xrange(len(tmp) - 1)])
# 		labels.append([int(tmp[-1])])
# data = np.matrix(data)
# labels = np.matrix(labels)


# Generate data
n = 50
w_real = [1, -1, 0]
data, labels = generate(n, 2, w_real)
# for i in xrange(data.shape[0]):
# 	print data[i, :], labels[i, :]

# Run regression
w1 = logisticRegression(data, labels, eta=0.5, reg=0.001, t=n ** 2)
w2 = privateLogReg(data, labels, eta=0.5, reg=0.001, t=n ** 2, eps=1, delta=0.1, c=100)
print 1 / w1.item(0, 0) * w1
print w_real


# Plot
cat1 = []
cat2 = []
for i in xrange(data.shape[0]):
	if labels.item(i, 0) == 1:
		cat1.append((data.item(i, 0), data.item(i, 1)))
	else:
		cat2.append((data.item(i, 0), data.item(i, 1)))
xs = np.arange(0, 1.01, 0.05)
liney = [(-w_real[0] * x - w_real[2]) / w_real[1] for x in xs]
sgdy = [(-w1.item(0, 0) * x - w1.item(0, 2)) / w1.item(0, 1) for x in xs]
nonPrivY = [(-w2.item(0, 0) * x - w2.item(0, 2)) / w2.item(0, 1) for x in xs]


plt.plot([pt[0] for pt in cat1], [pt[1] for pt in cat1], "go")
plt.plot([pt[0] for pt in cat2], [pt[1] for pt in cat2], "ro")
plt.plot(xs, liney, "blue")
plt.plot(xs, sgdy, "black")
plt.plot(xs, nonPrivY, "m")
plt.axis((0,1,0,1))
plt.show()

