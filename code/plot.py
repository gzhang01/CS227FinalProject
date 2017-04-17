import numpy as np
from generate import generate
import matplotlib.pyplot as plt
from nonPrivateLogReg import logisticRegression
from privateLogReg import privateLogReg
from objectPerturb import objectivePerturbation
import matplotlib.animation as animation

# Plot graphs
def plotGraph(cat1, cat2, xs, liney, w1, w2, w3, show=True, save=True, outfile="tmp.png"):
	# If neither showing nor saving, don't need to do any work
	if not show and not save:
		return

	# Plot setup
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	ax1.set_title('Generator')
	ax2.set_title('Non-Private SGD')
	ax3.set_title('Private SGD')
	ax4.set_title('Objective Perturbation')
	ax1.axis((0,1,0,1))
	ax2.axis((0,1,0,1))
	ax3.axis((0,1,0,1))
	ax4.axis((0,1,0,1))
	ax1.plot([pt[0] for pt in cat1], [pt[1] for pt in cat1], "go")
	ax1.plot([pt[0] for pt in cat2], [pt[1] for pt in cat2], "ro")
	ax2.plot([pt[0] for pt in cat1], [pt[1] for pt in cat1], "go")
	ax2.plot([pt[0] for pt in cat2], [pt[1] for pt in cat2], "ro")
	ax3.plot([pt[0] for pt in cat1], [pt[1] for pt in cat1], "go")
	ax3.plot([pt[0] for pt in cat2], [pt[1] for pt in cat2], "ro")
	ax4.plot([pt[0] for pt in cat1], [pt[1] for pt in cat1], "go")
	ax4.plot([pt[0] for pt in cat2], [pt[1] for pt in cat2], "ro")
	ax1.plot(xs, liney, "blue")
	line2, = ax2.plot(xs, [(-w1[0].item(0, 0) * x - w1[0].item(0, 2)) / w1[0].item(0, 1) for x in xs], "black")
	line3, = ax3.plot(xs, [(-w2[0].item(0, 0) * x - w2[0].item(0, 2)) / w2[0].item(0, 1) for x in xs], "m")
	line4, = ax4.plot(xs, [(-w3[0].item(0, 0) * x - w3[0].item(0, 2)) / w3[0].item(0, 1) for x in xs], "c")

	# Plotting animation
	def animate(i):
		# Compiling data to plot
		line2.set_ydata([(-w1[i].item(0, 0) * x - w1[i].item(0, 2)) / w1[i].item(0, 1) for x in xs])
		line3.set_ydata([(-w2[i].item(0, 0) * x - w2[i].item(0, 2)) / w2[i].item(0, 1) for x in xs])
		line4.set_ydata([(-w3[i].item(0, 0) * x - w3[i].item(0, 2)) / w3[i].item(0, 1) for x in xs])
		return line2, line3, line4,
	
	ani = animation.FuncAnimation(f, animate, np.arange(1, len(w1)), interval=30, blit=False)

	if save:
		plt.savefig("data/{0}".format(outfile))

	if show:
		plt.show()
	else:
		plt.close()


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

# Starting weights
w1 = np.matrix([0, 1, -0.5])
w2 = np.matrix([0, 1, -0.5])
w3 = np.matrix([0, 1, -0.5])

# Weights
sgdWeights = [w1]
nonPrivWeights = [w2]
objPretWeights = [w3]

# Granularity
t = 5

# Loss values
xLoss = []
sgdLoss = []
nonPrivLoss = []
objPretLoss = []

# Compiling data to plot
cat1 = []
cat2 = []
for j in xrange(data.shape[0]):
	if labels.item(j, 0) == 1:
		cat1.append((data.item(j, 0), data.item(j, 1)))
	else:
		cat2.append((data.item(j, 0), data.item(j, 1)))
xs = np.arange(0, 1.01, 0.05)
liney = [(-w_real[0] * x - w_real[2]) / w_real[1] for x in xs]

for i in xrange(n ** 2 / t):
	# Run regression
	w1, loss1 = logisticRegression(data, labels, eta=0.5, reg=0.0005, t=t, w=w1)
	w2, loss2 = privateLogReg(data, labels, eta=0.5, reg=0.0005, t=t, eps=1, delta=0.1, c=1.0/400, w=w2)
	w3, loss3 = objectivePerturbation(data, labels, eta=0.5, reg=0.0005, t=t, eps=1, delta=0.1, w=w3)
	# print 1 / w1.item(0, 0) * w1, loss1
	# print 1 / w2.item(0, 0) * w2, loss2
	# print 1 / w3.item(0, 0) * w3, loss3
	# print w_real

	# Add loss value
	xLoss.append((i + 1) * t)
	sgdLoss.append(loss1)
	nonPrivLoss.append(loss2)
	objPretLoss.append(loss3)

	# Add weights
	sgdWeights.append(w1)
	nonPrivWeights.append(w2)
	objPretWeights.append(w3)

# Plot last graph
plotGraph(cat1, cat2, xs, liney, sgdWeights, nonPrivWeights, objPretWeights, show=True, save=False)

plt.plot(xLoss, sgdLoss, "black")
plt.plot(xLoss, nonPrivLoss, "m")
plt.plot(xLoss, objPretLoss, "c")
plt.show()

