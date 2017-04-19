# README #

### What is this repository for? ###

* Testing what  I learn in class.
* v0.1
* Implemented features:
	* Simple 2D perceptron
	* Simple feedforward network

### How do I get set up? ###

* Needs EIGENv3.3.3 library: http://eigen.tuxfamily.org

### Nomenclature ###

#### Neural Networks ####

Layers is defined by an index starting at 0 which represents the input layer.

Weights, bias etc all follow this nomenclature even if a member does not exist (for example the weights matrix for the input layer does not exist).

Nonexistent elements are represented by a 1x1 '0' matrix

**example:**

A <2,5,1> feedforward network would have:

- 2 input neurons
- 5 hidden neurons in the hidden layer
- 1 output neuron

Weights: Vector of size 3

- weights[0] = 1x1 zero matrix
- weights[1] = 5x2 matrix
- weights[2] = 1x5 matrix

Bias: Vector of size 3

- bias[0] = 1x1 zero vector
- bias[1] = 5x1 vector
- bias[2] = 1x1 vector

Out: Vector of size 3

- out[0] = 2x1 vector
- out[1] = 5x1 vector
- out[2] = 1x1 vector

Deltas: Vector of size 3

- deltas[0] = 1x1 zero vector
- deltas[1] = 5x1 vector
- deltas[2] = 1x1 vector
