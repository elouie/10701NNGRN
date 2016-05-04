# Gene Regulatory Prediction

## Features


## Setup
For this project, we use SciPy, NumPy, and Theano for mathematical functions. Please see their guides to install:

* [NumPy](http://deeplearning.net/software/theano/tutorial/numpy.html)
* [SciPy](http://www.scipy.org/install.html)
* [Theano](http://www.deeplearning.net/software/theano/install.html#install)

### For MacOS:

Steps, some order matters, since there are dependencies during build steps:

* Install brew
* Install graphviz c libraries
* Install Theano graphics library
* Install SciPy, NumPy, and Theano

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install graphviz
brew install homebrew/science/hdf5
pip install -U pip setuptools
pip install scipy numpy theano pydot h5py
```

## Running Tests
We use the [unittest framework](https://docs.python.org/2/library/unittest.html) provided by Python. Instructions will be updated as test are added.

## Algorithms
Our algorithms have a simple interface plus arguments to modify functionality. The basic interface is:

```
M = algorithm.train(A)
    A: n*t*k matrix containing n parameters (molecules), t time steps, and k samples.
    M: A trained model for the algorithm providing various parameters necessary for prediction..
P = algorithm.predict(x, t)
    x: n*m initialization matrix containing n parameters (molecules) for m initialization vectors.
    t: Number of time steps the algorithm should predict for.
    P: Predicted n*t*m matrix containing the n*t prediction for each of m initialization vectors.
```

Evaluation of predictive models is completed in the larger framework.

### Multilayer Perceptron
The basic algorithm is to run a multilayer perceptron over all the supplied m training samples and treat each nx1 time step vector as an input vector and the following n*1 time step vector as the output.

## Running the Framework
We have created a convenience framework to load data and run through the algorithms. The steps to run the framework will be provided here: 

## Informative Readings
TBC
