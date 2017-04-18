# Differentially Private Stochastic Gradient Descent
George Zhang, CS227 Final Project (Cynthia Dwork), Harvard University Spring 2017

## Purpose
The goal of this project is to implement and analyze the utility of differentially private algorithms for stochastic gradient descent. 

## Usage
There are two files you can run: `generate.py` and `plot.py`. The SGD functions are in their respective Python files.

### `generate.py`
This file is used to generate random points on a d-dimensional space where each coordinate is on the interval [0, 1]. To run `generate.py` use: `python generate.py [n] [d] [w] [outfile]`
 * `n` - number of data points to generate
 * `d` - dimensions of data points
 * `w` - weight vector (NOTE: |w| must equal d + 1 to accommodate constant term)
   * example: `[-1,1,0]`
 * `outfile` - name of file to write data
 
This file can also be run using `python generate.py diagnose` which runs a diagnostic function. Any testing can be done in this function.

### `plot.py`
This file runs logistic regression using non-private SGD, private SGD, and objective perturbation. It then creates an animation showing the iterations run as well as a loss over iterations graph. To run `plot.py` use: `python plot.py [n] [w] [options]`
 * `n` - number of data points to generate (passed into `generate.py`)
 * `w` - weight vector (passed into `generate.py`)
   * NOTE: since `plot.py` only plots in two dimensions, |w| must equal 3
 * Options:
   * `-s`: flag to save the animation and loss graph under `data/[w][n]Vid.mp4` and `data/[w][n]Loss.png`

## To Do
### Main Goals
* Implement data generation script (Done)
* Implement non-private SGD (Done)
* Implement private SGD (Done)
* Data collection (Done)
* Presentation prepared
* Writeup complete

### Secondary Goals
* Visualization of algorithms (Done)
  * Plot final points / lines (Done)
  * Animate visualization (Done)
* Analysis with regards to theoretical utility
* Multinomial logistic regression?
