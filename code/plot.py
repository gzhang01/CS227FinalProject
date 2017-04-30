import numpy as np
from generate import generate
import matplotlib.pyplot as plt
from nonPrivateLogReg import logisticRegression
from privateLogReg import privateLogReg
from objectPerturb import objectivePerturbation
import matplotlib.animation as animation
import sys

# Plot graphs
def plotGraph(cat1, cat2, xs, liney, w1, w2, w3, show=True, save=True, outfile="tmp.mp4", interval=30):
	# If neither showing nor saving, don't need to do any work
	if not show and not save:
		return

	# Plot setup
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	plt.suptitle("Logistic Regression Results", size=16)
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
		if i % 10 == 0:
			print i
		# Compiling data to plot
		line2.set_ydata([(-w1[i].item(0, 0) * x - w1[i].item(0, 2)) / w1[i].item(0, 1) for x in xs])
		line3.set_ydata([(-w2[i].item(0, 0) * x - w2[i].item(0, 2)) / w2[i].item(0, 1) for x in xs])
		line4.set_ydata([(-w3[i].item(0, 0) * x - w3[i].item(0, 2)) / w3[i].item(0, 1) for x in xs])
		return line2, line3, line4,
	
	if len(w1) > 1:
		ani = animation.FuncAnimation(f, animate, np.arange(1, len(w1)), interval=interval, blit=False)
		if save:
			ani.save("{0}".format(outfile))
	elif save:
		plt.savefig("{0}".format(outfile))

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


def main(n, w_real, options):
	# Check for noisy threshold
	noisy = True if "-n" in options else False

	# Generate data
	data, labels = generate(n, 2, w_real, noisy)

	# Starting weights
	w1 = np.matrix([0, 1, -0.5])
	w2 = np.matrix([0, 1, -0.5])
	w3 = np.matrix([0, 1, -0.5])

	# Weights
	sgdWeights = [w1 for _ in xrange(20)]
	nonPrivWeights = [w2 for _ in xrange(20)]
	objPretWeights = [w3 for _ in xrange(20)]

	# Granularity
	t = 5
	lossT = 5
	c = 1.0/400

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
		w1, loss1 = logisticRegression(data, labels, eta=0.2, reg=0.0005, t=t, w=w1, mb=20)
		w2, loss2 = privateLogReg(data, labels, eta=0.2, reg=0.0005, t=t, eps=1, delta=0.1, c=c, w=w2, mb=20)
		w3, loss3 = objectivePerturbation(data, labels, eta=0.2, reg=0.0005, t=t, eps=1, delta=0.1, w=w3, mb=20)

		# Add loss value
		if i % lossT == 0:
			print i
			xLoss.append((i + 1) * t)
			sgdLoss.append(loss1)
			nonPrivLoss.append(loss2)
			objPretLoss.append(loss3)

		# Add weights
		sgdWeights.append(w1)
		nonPrivWeights.append(w2)
		objPretWeights.append(w3)

	# Plot animation
	save = True if "-s" in options else False
	noiseString = "N" if noisy else ""
	outfile = "../data/[{0}]{1}c{2}{3}Vid.mp4".format(",".join([str(x) for x in w_real]), n, c, noiseString) if save else "../data/tmp.mp4"
	time = 15
	interval = 30
	cutoff = time * 1000 / interval
	plotGraph(cat1, cat2, xs, liney, sgdWeights[:cutoff], nonPrivWeights[:cutoff], objPretWeights[:cutoff], show=True, save=save, outfile=outfile)

	# Plot loss functions
	plt.plot(xLoss, sgdLoss, "black")
	plt.plot(xLoss, nonPrivLoss, "m")
	plt.plot(xLoss, objPretLoss, "c")
	plt.suptitle('Loss Over Iterations', size=16)
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	if save:
		plt.savefig("{0}Loss.png".format(outfile.replace("Vid.mp4", "")))
	plt.show()

	# Plot final result
	plotGraph(cat1, cat2, xs, liney, w1, w2, w3, show=True, save=save, outfile=outfile.replace("Vid.mp4", "final.png"))

	# Testing Set
	testN = 5000
	testData, testLabels = generate(testN, 2, w_real, noisy)
	testData = np.hstack((testData, np.ones((testN, 1))))

	# [Nonprivate, private, objective]
	finalWeights = [w1, w2, w3]
	numCorrect = [0, 0, 0]
	for i in xrange(len(testData)):
		for j in xrange(len(finalWeights)):
			if testLabels.item(i, 0) * finalWeights[j].dot(testData[i, :].T).item(0, 0) >= 0:
				numCorrect[j] += 1

	if save:
		testOutfile = outfile.replace("Vid.mp4", "Test.txt")
		with open(testOutfile, "w") as f:
			f.write("Nonprivate: 	{0} Correct 	{1} Total 	{2}%\n".format(numCorrect[0], testN, 1.0 * numCorrect[0] / testN))
			f.write("Private: 		{0} Correct 	{1} Total 	{2}%\n".format(numCorrect[1], testN, 1.0 * numCorrect[1] / testN))
			f.write("Objective: 	{0} Correct 	{1} Total 	{2}%\n".format(numCorrect[2], testN, 1.0 * numCorrect[2] / testN))
	print "Nonprivate: 	{0} Correct 	{1} Total 	{2}%".format(numCorrect[0], testN, 1.0 * numCorrect[0] / testN)
	print "Private: 	{0} Correct 	{1} Total 	{2}%".format(numCorrect[1], testN, 1.0 * numCorrect[1] / testN)
	print "Objective: 	{0} Correct 	{1} Total 	{2}%".format(numCorrect[2], testN, 1.0 * numCorrect[2] / testN)

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "Usage: python plot.py [n] [w] [options]"
		exit(1)

	# Parse command line input
	n = int(sys.argv[1])
	w_real = map(float, sys.argv[2].replace("[", "").replace("]", "").split(","))
	options = sys.argv[3:]
	main(n, w_real, options)



