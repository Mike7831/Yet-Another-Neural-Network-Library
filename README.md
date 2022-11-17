# Yet Another Neural Network Library C++ (YANNL)

## Overview

Main goal of this C++ library is to mimic the behavior of `MLPRegressor` and `MLPClassifier` (Multi-Layer Perceptron) of the Python Scikit-Learn library in C++. It contains all the underlying classes for representing a Multi-Layer Perceptron.  
It also includes a very simple XML reader and MNIST file reader. [MNIST](http://yann.lecun.com/exdb/mnist/) is a database of handwritten digits that the main file uses to demonstrate the possibility of the library.

It uses only the C++ standard library. Compiled with C++14 compiler. No dependencies to other libraries.

## Characteristics

Layer types included:
* Dense
    * Hidden
    * Output classification (with a Cross-Entropy Error function)
    * Output regression (with Mean-Squared Error function)
* Dropout

Activation functions included:
* Identity
* Logistic
* Tanh
* ReLU
* ISRLU
Other activation functions can be added easily in the `ActivationFunction.h` file.


## Folder structure

* `data` Dataset used in `main` for demonstrating the possibility of the YANN-Library
    * MNIST handwritten digit database files
    * Iris flower dataset
* `lib`
    * `include` **Header-only library** that you can easily copy/paste into a project
        * `mnist-reader` Utility library for reading the [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/)
        * `neural-net` The Yet Another Neural Network Library including `MLPRegressor` and `MLPClassifier`. See in subsequent section the structure of this folder.
        * `xml-reader` Very simple XML reader. YANN-Library can output a neural network structure to a file and read/load it back. At first I thought about using an XML format for such serialization, but finally ended up with a flat file structure. `neural-net` can thus be used without this XML reader
    * `src` Nothing as the library is currently a **header-only** library
* `main`
    * `include` & `src` Header and source files containing prediction examples using the YANN-Library
        * `IrisClassification` Example of usage of `MLPClassifier` to predict the type of iris flower.
        * `MnistPrediction` Example of usage of the `neural-net` classes to predict the MNIST handwritten digits.
        * `XORPrediction` Example of usage of `MLPRegressor` to predict the output of an XOR gate.
    * `main.cpp` The `main' which launches the 3 previous examples: iris classification, MNIST prediction, XOR prediction.
* `test`
    * `expected` Output of unit/regression tests compare to the result files contained in this folder
    * `include/UnitTest.h` Battery of unit/regression tests for the `mnist-reader`, `neural-net` and `xml-reader`. Many methods within the `UnitTest` class have the same name as the files in the `expected` folder: the output of such methods are compared to those files.
    * `src` Nothing as the unit/regression tests are all contained in the header files
    * `test.cpp` The `main' which launches the whole test battery.

## Neural network library structure

```mermaid
classDiagram
MLP <|.. MLPRegressor
MLP <|.. MLPClassifier
MLP *-- "1..1" NeuralNetwork
NeuralNetwork *-- "0..*" NeuronLayer
NeuralNetwork *-- "1..1" Utils_SeedGenerator
Neuron *-- "1..1" ActivationFunction
NeuronLayer <|.. DenseLayer
NeuronLayer <|.. DropoutLayer
DenseLayer <|-- HiddenLayer
DenseLayer <|-- OutputClassificationLayer
DenseLayer <|-- OutputRegressionLayer
DenseLayer *-- "0..*" Neuron

class MLP {
    #unique_ptr<NeuralNetwork> net
    +fit(vect<vect<double> X_train, vect<double> y_train)
}
class MLPClassifier {
    +MLPClassifier(...)
    +predict(vect<double> X_test) size_t
}
class MLPRegressor {
    +MLPRegressor(...)
    +predict(vect<double> X_test) double
    +predict(vect<vect<double> X_test) vect_double
}
class NeuralNetwork {
    -vect<shared_ptr<NeuronLayer>> layers
    -double learningRate
    -double momentum
    +addHiddenLayer(vect<vect<double>> weights, ...)
    +addHiddenLayer(size_t inputSize, ...)
    +addOutputClassificationLayer(vect<vect<double>> weights, ...)
    +addOutputClassificationLayer(size_t inputSize, ...)
    +addOutputRegressionLayer(vect<vect<double>> weights, ...)
    +addOutputRegressionLayer(size_t inputSize, ...)
    +addDropoutLayer(double dropoutRate)
    +propagateForward(vect<double> input) output
    +propagateBackward(vect<double> expectedOutput)
    +saveToFile(string filepath) bool
    +loadFromFile(string filepath)$ NeuralNetwork
}
class DenseLayer {
    -vect<Neuron> neurons
    -double learningRate
    -double momentum
}
class DropoutLayer {
    -vect<bool> active_neurons
    -double dropoutRate
}
class Neuron {
    -vect<double> weights
    -shared_ptr<ActivationFunction> AFunc
    -double learningRate
    -double momentum
}
class ActivationFunction {
    +calc(double x) y
    +calcDerivate(double x) y
    +name() string
}
class Utils_SeedGenerator {
    +seed() : uint
}
```

## Installation

You will find at the root of the project:
* Solution file for Visual Studio 2022
* Workspace file for Code::Blocks

If you would like to launch the main examples (iris classification, MNIST prediction, XOR prediction) set the `main` project as active/startup project (right click > Activate project in C::B, right click > Set as startup project in MS VS).

If you would like to launch the battery of unit/regression tests, set the `test` project as active/startup project.

It is also easy to copy/paste the files into other projects as the library relies only on the STL and the library files are header-only.

## Usage

`main.cpp` within the `main` folder calls different examples which give examples on how to use the YANNL-Library

Globally `MLPRegressor` and `MLPClassifier` can be used approximately in the same way as the Python version.

It is also possible to build manually a neural network (`NeuralNetwork` class) and then add the layers (`HiddenLayer`, `OutputClassificationLayer`, `OutputRegressionLayer`, `DropoutLayer`). The input layer is not considered as a layer in the layer count. Its size is provided as input when building the `NeuralNetwork`. It should not be necessary to build manually layers and neurons as they are supposed to be built via the constructors and methods within `NeuralNetwork`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## What next

The `MLPRegressor` and `MLPClassifier` only use dense layers (hidden layers and output classification/regression layers).  
Dropout layers are not used. Could be an improvement.  
Further improvement would be to implement convolutional neural networks.

## Author

Mickael Deloison

## Sources

[A Step by Step Back propagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)  
[MLPRegressor specification](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)  
[MLPClassifier specification](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)  
[Dropout layer](https://keras.io/api/layers/regularization_layers/dropout/)  
[Gradient Descent With Momentum from Scratch](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/)  
