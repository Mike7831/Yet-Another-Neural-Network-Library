//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#ifndef YANNL_UNIT_TEST_H
#define YANNL_UNIT_TEST_H

#include "MLP.h"
#include "MnistReader.h"
#include "SimpleXMLReader.h"
#include <cassert> // assert for testing purpose

using namespace YANNL;

class YANNL_UnitTests
{
    const char* kTestDir = "../test/expected/";
    const char* kOutputDir = "../output/";
    const char* kDataDir = "../data/";
    const char* kTestFileExt = ".txt";
    
public:
    void execExceptionTests()
    {
        std::cout << ">> Testing exception returns... ";
        exceptions();
        std::cout << "done. \n";
    }

    void execNeuralNetworkTests()
    {
        try
        {
            std::cout << ">> Testing forward propagation and mean squared error calculation " <<
                "with a network of predefined weights... ";
            forwardPropAndMSErrorDefinedWeights();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a network of predefined weights... ";
            backPropDefinedWeights();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a dropout layer (rate = 0.4) " <<
                "after the input layer... ";
            backPropDropoutInput0_4();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a dropout layer (rate = 1.0) " <<
                "after the input layer... ";
            backPropDropoutInput1_0();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a dropout layer (rate = 0.4) " <<
                "after the hidden layer... ";
            backPropDropoutHidden0_4();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a dropout layer (rate = 1.0) " <<
                "after the hidden layer... ";
            backPropDropoutHidden1_0();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a network of predefined weights for 10000 epochs... ";
            backPropDefinedWeightsFor10000epochs();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a momentum... ";
            backPropMomentum();
            std::cout << "done. \n";

            std::cout << ">> Testing save and load of a neural network with predefined weights... ";
            saveAndLoadNetworkDefinedWeights();
            std::cout << "done. \n";

            std::cout << ">> Testing save, update and load of a neural network with predefined weights... ";
            saveLoadUpdateSaveLoadDefinedWeights();
            std::cout << "done. \n";

            std::cout << ">> Testing save and load of a neural network with random weights... ";
            saveAndLoadNetworkRandomWeights();
            std::cout << "done. \n";

            std::cout << ">> Testing forward propagation and cross entropy error calculation " <<
                "with a classification layer of 2 neurons... ";
            forwardPropAndCEErrorClassificationOutput2N();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a network with a classification "
                << "layer of 2 neurons... ";
            backPropClassificationOutput2N();
            std::cout << "done. \n";

            std::cout << ">> Testing forward propagation and cross entropy error calculation " <<
                "with a classification layer of 3 neurons... ";
            forwardPropAndCEErrorClassificationOutput3N();
            std::cout << "done. \n";

            std::cout << ">> Testing back propagation with a network with a classification "
                << "layer of 3 neurons... ";
            backPropClassificationOutput3N();
            std::cout << "done. \n";

            std::cout << ">> Testing save and load of a neural network with an output "
                "classification layer of 3 neurons... ";
            saveAndLoadNetworkClassificationOutput3N();
            std::cout << "done. \n";
        }
        catch (std::exception& e)
        {
            std::cout << "\n"
                << "Exception! " << e.what() << "\n";
            abort();
        }
    }

    void execMnistTests()
    {
        std::cout << ">> Reading MNIST test image file and check normalization method... ";
        mnistTestImageRead();
        std::cout << "done. \n";

        std::cout << ">> Reading MNIST test label file and check display... ";
        mnistTestLabelRead();
        std::cout << "done. \n";

        std::cout << ">> Exception when reading a MNIST file which does not exist... ";
        mnistTestReadException();
        std::cout << "done. \n";
    }

    void execOtherTests()
    {
        std::cout << ">> Testing a neural network simulating an XOR gate with random "
            "weights but fixed seed... ";
        xorRandomWeightsFixedSeed();
        std::cout << "done. \n";
    }

    void execXMLTests()
    {
        std::cout << ">> Reading XML file, saving it and comparing it... ";
        xmlReadAndSave();
        std::cout << "done. \n";
    }

    void execMLPTests()
    {
        std::cout << ">> Testing MLPRegressor with constant learning rate and no early stopping... ";
        mlpRegressorConstLearningRateNoEarlyStopping();
        std::cout << "done. \n";

        std::cout << ">> Testing MLPRegressor with constant learning rate and early stopping... ";
        mlpRegressorConstLearningRateEarlyStopping();
        std::cout << "done. \n";
        
        std::cout << ">> Testing MLPRegressor with inverse scaling learning rate... ";
        mlpRegressorInvScalingLearningRate();
        std::cout << "done. \n";
        
        std::cout << ">> Testing MLPRegressor with adaptive learning rate... ";
        mlpRegressorAdaptiveLearningRate();
        std::cout << "done. \n";
        
        std::cout << ">> Testing MLPClassifier with constant learning rate and no early stopping... ";
        mlpClassifierConstLearningRateNoEarlyStopping();
        std::cout << "done. \n";
    }

    void execBatchTrainingTests()
    {
        std::cout << ">> Testing weight updates after 3 forward and backward passes... ";
        backPropRegressionBatch3N();
        std::cout << "done. \n";

        std::cout << ">> Testing weight updates after saving and loading the network in the "
            << "middle of a batch training... ";
        saveAndLoadNetworkAfterBatchTraining();
        std::cout << "done. \n";
    }

private:
    void exceptions()
    {
        std::ostream os(nullptr);

        try
        {
            os << "Add hidden layer after an output layer" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputClassificationLayer(2);
            net->addHiddenLayer(2, ActivationFunctions::Tanh);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add hidden layer with predefined weights after an output layer" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputClassificationLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } });
            net->addHiddenLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Tanh);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add hidden layer with inconsistent weights" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ {0.4, 0.45}, {0.5, 0.55, 0.1} }, ActivationFunctions::Tanh);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add hidden layer with inconsistent weights" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ {0.4, 0.45}, {0.5} }, ActivationFunctions::Tanh);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add output classification layer after an output layer" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputRegressionLayer(2, ActivationFunctions::Tanh);
            net->addOutputClassificationLayer(2);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add output classification layer with predefined weights after an output layer" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputRegressionLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Tanh);
            net->addOutputClassificationLayer(2);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add output classification layer with inconsistent weights" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55, 0.1} });
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add output regression layer after an output layer" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputRegressionLayer(2, ActivationFunctions::Tanh);
            net->addOutputRegressionLayer(2, ActivationFunctions::Tanh);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add output regression layer with predefined weights after an output layer" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputRegressionLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Tanh);
            net->addOutputRegressionLayer(2, ActivationFunctions::Tanh);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Add output regression layer with inconsistent weights" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55, 0.1} }, ActivationFunctions::Tanh);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Propagating forward with no layers" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            os << "Output: " << net->propagateForward({ 0.05, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Propagating forward with no output layers" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            os << "Output: " << net->propagateForward({ 0.05, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Propagating forward with no output layers" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net->addHiddenLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
            os << "Output: " << net->propagateForward({ 0.05, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Propagating forward with inconsistent input size" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net->addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
            os << "Output: " << net->propagateForward({ 0.05, 0.1, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Propagating forward with inconsistent input size" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net->addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
            os << "Output: " << net->propagateForward({ 0.05 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Get probable class with no output layer" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            os << "Probable class: " << net->probableClass() << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Calculate error with no layers" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            os << "Error: " << net->calcError({ 0.05, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Calculate error with no output layers" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            os << "Error: " << net->calcError({ 0.05, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Calculate error with inconsistent output size (regression)" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net->addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
            os << "Error: " << net->calcError({ 0.05, 0.1, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Calculate error with inconsistent output size (classification)" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net->addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55} }, 0.6);
            os << "Error: " << net->calcError({ 0.05, 0.1, 0.1 }) << "\n";
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Propagate backward with no layers" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->propagateBackward({ 0.01, 0.99 });
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Propagate backward with no output layers" << "\n";
            std::unique_ptr<NeuralNetwork> net(std::make_unique<NeuralNetwork>(2, 0.5));
            net->addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net->propagateBackward({ 0.01, 0.99 });
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }

        try
        {
            os << "Inaccessible file when loading network" << "\n";
            NeuralNetwork net = NeuralNetwork::loadFromFile("dummyfile.txt");
            net.inspect(os);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }
    }

    void forwardPropAndMSErrorDefinedWeights()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDefinedWeights()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net.propagateForward({ 0.05, 0.1 });
        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutInput0_4()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5, 0, true, 18);
        net.addDropoutLayer(0.4);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutInput1_0()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addDropoutLayer(1.0);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutHidden0_4()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5, 0, true, 18);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addDropoutLayer(0.4);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutHidden1_0()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5, 0, true, 18);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addDropoutLayer(1.0);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDefinedWeightsFor10000epochs()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        for (size_t n = 0; n < 10000; n++)
        {
            net.propagateForward({ 0.05, 0.1 });
            net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });

            if (n == 0)
            {
                os << "Error after 1 case: " << net.calcError({ 0.01, 0.99 }) << "\n";
            }
        }

        os << "Error after 10000 cases: " << net.calcError({ 0.01, 0.99 }) << "\n";

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropMomentum()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5, 0.4);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net.inspect(os);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "Error: " << net.calcError({ 0.01, 0.99 }) << "\n";
        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "Error: " << net.calcError({ 0.01, 0.99 }) << "\n";
        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void saveAndLoadNetworkDefinedWeights()
    {
        NeuralNetwork net1(2, 0.5, 0, true, 20);
        net1.addDropoutLayer(0.4);
        net1.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net1.addDropoutLayer(0.4);
        net1.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });

        net1.saveToFile(std::string(kOutputDir) + "net1.txt");

        NeuralNetwork net2 = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");

        {
            std::stringstream is1 = readExpectedResultFile(std::string(kOutputDir) + "net1.txt");
            std::stringstream is2 = readExpectedResultFile(std::string(kOutputDir) + "net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }


        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net2.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net1.saveToFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");
        NeuralNetwork net1b = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
        NeuralNetwork net2b = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net2.txt");
        net1b.saveToFile(std::string(kOutputDir) + "net1.txt");
        net2b.saveToFile(std::string(kOutputDir) + "net2.txt");

        {
            std::stringstream is1 = readExpectedResultFile(std::string(kOutputDir) + "net1.txt");
            std::stringstream is2 = readExpectedResultFile(std::string(kOutputDir) + "net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }
    }

    void saveLoadUpdateSaveLoadDefinedWeights()
    {
        NeuralNetwork net1(2, 0.5, 0, true, 40);
        net1.addDropoutLayer(0.4);
        net1.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net1.saveToFile(std::string(kOutputDir) + "net1.txt");

        NeuralNetwork net2 = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");

        net1.addDropoutLayer(0.4);
        net1.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net2.addDropoutLayer(0.4);
        net2.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net1.propagateForward({ 0.05, 0.1 }); net2.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99 }); net2.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net1.propagateForward({ 0.05, 0.1 }); net2.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99 }); net2.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });

        net1.saveToFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");
        std::stringstream is1 = readExpectedResultFile(std::string(kOutputDir) + "net1.txt");
        std::stringstream is2 = readExpectedResultFile(std::string(kOutputDir) + "net2.txt");

        compareLineByLine(__func__, is1.str(), is2.str());
    }

    void saveAndLoadNetworkRandomWeights()
    {
        NeuralNetwork net1(2, 0.5, 0, true, 20);
        net1.addDropoutLayer(0.4);
        net1.addHiddenLayer(5, ActivationFunctions::Logistic, 0.35);
        net1.addDropoutLayer(0.4);
        net1.addOutputRegressionLayer(3, ActivationFunctions::Logistic, 0.6);

        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.85 });
        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.85 });

        net1.saveToFile(std::string(kOutputDir) + "net1.txt");

        NeuralNetwork net2 = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");

        {
            std::stringstream is1 = readExpectedResultFile(std::string(kOutputDir) + "net1.txt");
            std::stringstream is2 = readExpectedResultFile(std::string(kOutputDir) + "net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }


        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.85 });
        net2.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.85 });
        net1.saveToFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");
        NeuralNetwork net1b = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
        NeuralNetwork net2b = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net2.txt");
        net1b.saveToFile(std::string(kOutputDir) + "net1.txt");
        net2b.saveToFile(std::string(kOutputDir) + "net2.txt");

        {
            std::stringstream is1 = readExpectedResultFile(std::string(kOutputDir) + "net1.txt");
            std::stringstream is2 = readExpectedResultFile(std::string(kOutputDir) + "net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }
    }

    void forwardPropAndCEErrorClassificationOutput2N()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55} }, 0.6);
        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "CEE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropClassificationOutput2N()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55} }, 0.6);

        net.propagateForward({ 0.05, 0.1 });
        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void forwardPropAndCEErrorClassificationOutput3N()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55}, {0.8, 0.4} }, 0.6);
        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "CEE: " << net.calcError({ 0.01, 0.99, 0.82 }) << "\n";

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropClassificationOutput3N()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55}, {0.8, 0.4} }, 0.6);

        net.propagateForward({ 0.05, 0.1 });
        net.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.82 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void saveAndLoadNetworkClassificationOutput3N()
    {
        NeuralNetwork net1(2, 0.5);
        net1.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net1.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55}, {0.8, 0.4} }, 0.6);

        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.82 });
        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.82 });

        net1.saveToFile(std::string(kOutputDir) + "net1.txt");

        NeuralNetwork net2 = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");

        {
            std::stringstream is1 = readExpectedResultFile(std::string(kOutputDir) + "net1.txt");
            std::stringstream is2 = readExpectedResultFile(std::string(kOutputDir) + "net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }


        net1.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.82 });
        net2.propagateBackwardAndUpdateWeights({ 0.01, 0.99, 0.82 });
        net1.saveToFile(std::string(kOutputDir) + "net1.txt");
        net2.saveToFile(std::string(kOutputDir) + "net2.txt");
        NeuralNetwork net1b = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
        NeuralNetwork net2b = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net2.txt");
        net1b.saveToFile(std::string(kOutputDir) + "net1.txt");
        net2b.saveToFile(std::string(kOutputDir) + "net2.txt");

        {
            std::stringstream is1 = readExpectedResultFile(std::string(kOutputDir) + "net1.txt");
            std::stringstream is2 = readExpectedResultFile(std::string(kOutputDir) + "net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }
    }

    void xorRandomWeightsFixedSeed()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5, 0.9, true, 10); // Random weights but with a fixed seed
        net.addHiddenLayer(5, ActivationFunctions::Logistic);
        net.addOutputRegressionLayer(1, ActivationFunctions::Logistic);

        std::vector<std::pair<std::vector<double>, double>> trainingSets{
            { {0, 0}, 0},
            { {0, 1}, 1},
            { {1, 0}, 1},
            { {1, 1}, 0} };

        for (size_t n = 0; n < 10000; n++)
        {
            for (const std::pair<std::vector<double>, double>& trainingSet : trainingSets)
            {
                net.propagateForward(trainingSet.first);
                net.propagateBackwardAndUpdateWeights(trainingSet.second);
            }
        }

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            os << "Input: " << testSet.first
                << "  Output: " << net.propagateForward(testSet.first)
                << "  Expected: " << testSet.second << "\n";
        }

        compareLineByLine(__func__, os.str(), is.str());
    }

    void mnistTestImageRead()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        MnistReader::ImageContainer images;
        MnistReader::MnistFileAttrs attrs;

        attrs = MnistReader::readMnist(std::string(kDataDir) + "t10k-images.idx3-ubyte", images);
        MnistReader::displayMnist(images, os, 0, 10, attrs);

        compareLineByLine(__func__, os.str(), is.str());

        MnistReader::NormalizedImageContainer normImages = MnistReader::normalize(images);

        assert(*std::min_element(images[0].cbegin(), images[0].cend()) == 0);
        assert(*std::max_element(images[0].cbegin(), images[0].cend()) == 255);
        assert(*std::min_element(normImages[0].cbegin(), normImages[0].cend()) == 0.0);
        assert(*std::max_element(normImages[0].cbegin(), normImages[0].cend()) == 1.0);
    }

    void mnistTestLabelRead()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        MnistReader::LabelContainer labels;
        int32_t count;

        count = MnistReader::readMnist(std::string(kDataDir) + "t10k-labels.idx1-ubyte", labels);
        MnistReader::displayMnist(labels, os, 9990, 10002, count);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void mnistTestReadException()
    {
        std::ostream os(nullptr);

        try
        {
            os << "Read MNIST file which does not exist" << "\n";
            MnistReader::ImageContainer images;
            MnistReader::MnistFileAttrs attrs;
            attrs = MnistReader::readMnist("dummy.idx3-ubyte", images);
            assert(false);
        }
        catch (std::exception& e) { os << "Exception! " << e.what() << "\n"; }
    }

    void xmlReadAndSave()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);
        std::ifstream ifs(std::ifstream(kTestDir + std::string(__func__) + ".xml"));

        std::unique_ptr<XMLNode> xml = readXMLStream(ifs);
        xml->inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void mlpRegressorConstLearningRateNoEarlyStopping()
    {
        NeuralNetwork net(2, 0.5, 0.9, true, 10); // Random weights but with a fixed seed
        net.addHiddenLayer(5, ActivationFunctions::Logistic);
        net.addOutputRegressionLayer(1, ActivationFunctions::Logistic);

        std::vector<std::pair<std::vector<double>, double>> trainingSets{
            { {0, 0}, 0},
            { {0, 1}, 1},
            { {1, 0}, 1},
            { {1, 1}, 0} };

        for (size_t n = 0; n < 10000; n++)
        {
            for (const std::pair<std::vector<double>, double>& trainingSet : trainingSets)
            {
                net.propagateForward(trainingSet.first);
                net.propagateBackwardAndUpdateWeights(trainingSet.second);
            }
        }

        MLPRegressor mlp({ 5 },             // hidden_layer_sizes
            ActivationFunctions::Logistic,  // activation
            Solvers::SGD,                   // solver
            LearningRate::Constant,         // learning_rate
            0.5,                            // learning_rate_init
            0.5,                            // power_t
            10000,                          // max_iter
            true,                           // use_random_state
            10,                             // random_state
            1.0E-4,                         // tol
            false,                          // verbose
            0.9,                            // momentum
            false,                          // early_stopping
            10                              // n_iter_no_change
        );
        mlp.fit({ {0, 0},  {0, 1},  {1, 0},  {1, 1} }, { 0, 1, 1, 0 });

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            std::vector<double> output = net.propagateForward(testSet.first);
            assert(output.front() == mlp.predict(testSet.first));
        }
    }

    void mlpRegressorConstLearningRateEarlyStopping()
    {
        NeuralNetwork net(2, 0.5, 0.9, true, 10); // Random weights but with a fixed seed
        net.addHiddenLayer(5, ActivationFunctions::Logistic);
        net.addOutputRegressionLayer(1, ActivationFunctions::Logistic);

        std::vector<std::pair<std::vector<double>, double>> trainingSets{
            { {0, 0}, 0},
            { {0, 1}, 1},
            { {1, 0}, 1},
            { {1, 1}, 0} };

        MLPRegressor mlp({ 5 },             // hidden_layer_sizes
            ActivationFunctions::Logistic,  // activation
            Solvers::SGD,                   // solver
            LearningRate::Constant,         // learning_rate
            0.5,                            // learning_rate_init
            0.5,                            // power_t
            10000,                          // max_iter
            true,                           // use_random_state
            10,                             // random_state
            1.0E-5,                         // tol
            false,                          // verbose
            0.9,                            // momentum
            true,                           // early_stopping
            10                              // n_iter_no_change
        );
        mlp.fit({ {0, 0},  {0, 1},  {1, 0},  {1, 1} }, { 0, 1, 1, 0 });

        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            os << mlp.predict(testSet.first) << "\n";
        }

        compareLineByLine(__func__, os.str(), is.str());
    }

    void mlpRegressorInvScalingLearningRate()
    {
        NeuralNetwork net(2, 0.5, 0.9, true, 10); // Random weights but with a fixed seed
        net.addHiddenLayer(5, ActivationFunctions::Logistic);
        net.addOutputRegressionLayer(1, ActivationFunctions::Logistic);

        std::vector<std::pair<std::vector<double>, double>> trainingSets{
            { {0, 0}, 0},
            { {0, 1}, 1},
            { {1, 0}, 1},
            { {1, 1}, 0} };

        MLPRegressor mlp({ 5 },             // hidden_layer_sizes
            ActivationFunctions::Logistic,  // activation
            Solvers::SGD,                   // solver
            LearningRate::InvScaling,       // learning_rate
            0.5,                            // learning_rate_init
            0.5,                            // power_t
            10000,                          // max_iter
            true,                           // use_random_state
            10,                             // random_state
            1.0E-4,                         // tol
            false,                          // verbose
            0.9,                            // momentum
            false,                          // early_stopping
            10                              // n_iter_no_change
        );
        mlp.fit({ {0, 0},  {0, 1},  {1, 0},  {1, 1} }, { 0, 1, 1, 0 });

        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            os << mlp.predict(testSet.first) << "\n";
        }

        compareLineByLine(__func__, os.str(), is.str());
    }

    void mlpRegressorAdaptiveLearningRate()
    {
        NeuralNetwork net(2, 0.5, 0.9, true, 10); // Random weights but with a fixed seed
        net.addHiddenLayer(5, ActivationFunctions::Logistic);
        net.addOutputRegressionLayer(1, ActivationFunctions::Logistic);

        std::vector<std::pair<std::vector<double>, double>> trainingSets{
            { {0, 0}, 0},
            { {0, 1}, 1},
            { {1, 0}, 1},
            { {1, 1}, 0} };

        MLPRegressor mlp({ 5 },             // hidden_layer_sizes
            ActivationFunctions::Logistic,  // activation
            Solvers::SGD,                   // solver
            LearningRate::Adaptive,         // learning_rate
            0.5,                            // learning_rate_init
            0.5,                            // power_t
            10000,                          // max_iter
            true,                           // use_random_state
            10,                             // random_state
            1.0E-5,                         // tol
            false,                          // verbose
            0.9,                            // momentum
            false,                          // early_stopping
            10                              // n_iter_no_change
        );
        mlp.fit({ {0, 0},  {0, 1},  {1, 0},  {1, 1} }, { 0, 1, 1, 0 });

        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            os << mlp.predict(testSet.first) << "\n";
        }

        compareLineByLine(__func__, os.str(), is.str());
    }

    void mlpClassifierConstLearningRateNoEarlyStopping()
    {
        NeuralNetwork net(2, 0.1, 0.0, true, 10); // Random weights but with a fixed seed
        net.addHiddenLayer(3, ActivationFunctions::Logistic);
        net.addHiddenLayer(3, ActivationFunctions::Logistic);
        net.addOutputClassificationLayer(2);

        std::vector<std::pair<std::vector<double>, uint8_t>> trainingSets{
            { {0, 0}, 0},
            { {0, 1}, 1},
            { {1, 0}, 1},
            { {1, 1}, 0} };

        for (size_t n = 0; n < 100; n++)
        {
            for (const std::pair<std::vector<double>, uint8_t>& trainingSet : trainingSets)
            {
                net.propagateForward(trainingSet.first);
                net.propagateBackwardAndUpdateWeights(Utils::convertLabelToVect(trainingSet.second, 0, 1));
            }
        }

        MLPClassifer mlp({ 3, 3 },          // hidden_layer_sizes
            ActivationFunctions::Logistic,  // activation
            Solvers::SGD,                   // solver
            LearningRate::Constant,         // learning_rate
            0.1,                            // learning_rate_init
            0.5,                            // power_t
            100,                            // max_iter
            true,                           // use_random_state
            10,                             // random_state
            1.0E-4,                         // tol
            false,                          // verbose
            0.0,                            // momentum
            false,                          // early_stopping
            10                              // n_iter_no_change
        );
        mlp.fit({ {0, 0},  {0, 1},  {1, 0},  {1, 1} }, { 0, 1, 1, 0 });

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            net.propagateForward(testSet.first);
            assert(net.probableClass() == mlp.predict(testSet.first));
        }
    }

    void backPropRegressionBatch3N()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net.propagateForward({ 0.05, 0.1 });
        net.propagateBackward({ 0.01, 0.99 });
        net.propagateForward({ 0.08, 0.1 });
        net.propagateBackward({ 0.01, 0.99 });
        net.propagateForward({ 0.05, 0.1 });
        net.propagateBackward({ 0.01, 0.99 });
        net.updateWeights();
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void saveAndLoadNetworkAfterBatchTraining()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile(kTestDir + std::string(__func__) + kTestFileExt);

        {
            NeuralNetwork net1(2, 0.5);
            net1.addDropoutLayer(0.0);
            net1.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net1.addDropoutLayer(0.0);
            net1.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

            net1.propagateForward({ 0.05, 0.1 });
            net1.propagateBackward({ 0.01, 0.99 });
            net1.propagateForward({ 0.08, 0.1 });
            net1.propagateBackward({ 0.01, 0.99 });
            net1.propagateForward({ 0.05, 0.1 });

            net1.saveToFile(std::string(kOutputDir) + "net1.txt");
        }

        {
            NeuralNetwork net1 = NeuralNetwork::loadFromFile(std::string(kOutputDir) + "net1.txt");
            net1.propagateBackward({ 0.01, 0.99 });
            net1.updateWeights();
            net1.inspect(os);
        }

        compareLineByLine(__func__, os.str(), is.str());
    }

    std::stringstream readExpectedResultFile(const std::string& filepath)
    {
        std::ifstream is(filepath);
        std::stringstream buffer;
        buffer << is.rdbuf();
        return buffer;
    }

    void compareLineByLine(const std::string& callingFunction,
        const std::string& s1, const std::string& s2) const
    {
        std::istringstream is1(s1);
        std::istringstream is2(s2);
        std::string l1, l2;
        size_t lineN = 1;

        while (std::getline(is1, l1))
        {
            std::getline(is2, l2);

            if (l1 != l2)
            {
                std::cerr << "\n"
                    << "Assertion failed on line " << lineN << " of " << callingFunction << ".txt: \n"
                    << "Provided: " << l1 << "\n"
                    << "Expected: " << l2 << std::endl;
                abort();
            }

            lineN++;
        }
    }
};

#endif // YANNL_UNIT_TEST_H
