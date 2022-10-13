#include "NeuralNetwork.h"
#include "Utils.h"
#include "MnistReader.h"

//#include "MLP.h"

#include <iomanip> // std::setprecision
#include <cassert> // assert for testing purpose

using namespace YANNL;

class YANNL_UnitTests
{
public:
    static void execExceptionTests()
    {
        std::cout << ">> Testing exception returns... ";
        m_YANNL_UnitTest.exceptions();
        std::cout << "done. \n";
    }

    static void execNeuralNetworkTests()
    {
        std::cout << ">> Testing forward propagation and mean squared error calculation " <<
            "with a network of predefined weights... ";
        m_YANNL_UnitTest.forwardPropAndMSErrorDefinedWeights();
        std::cout << "done. \n";

        std::cout << ">> Testing back propagation with a network of predefined weights... ";
        m_YANNL_UnitTest.backPropDefinedWeights();
        std::cout << "done. \n";

        std::cout << ">> Testing back propagation with a dropout layer (rate = 0.4) " <<
            "after the input layer... ";
        m_YANNL_UnitTest.backPropDropoutInput0_4();
        std::cout << "done. \n";

        std::cout << ">> Testing back propagation with a dropout layer (rate = 1.0) " <<
            "after the input layer... ";
        m_YANNL_UnitTest.backPropDropoutInput1_0();
        std::cout << "done. \n";

        std::cout << ">> Testing back propagation with a dropout layer (rate = 0.4) " <<
            "after the hidden layer... ";
        m_YANNL_UnitTest.backPropDropoutHidden0_4();
        std::cout << "done. \n";

        std::cout << ">> Testing back propagation with a dropout layer (rate = 1.0) " <<
            "after the hidden layer... ";
        m_YANNL_UnitTest.backPropDropoutHidden1_0();
        std::cout << "done. \n";

        std::cout << ">> Testing back propagation with a network of predefined weights for 10000 epochs... ";
        m_YANNL_UnitTest.backPropDefinedWeightsFor10000epochs();
        std::cout << "done. \n";

        std::cout << ">> Testing back propagation with a momentum... ";
        m_YANNL_UnitTest.backPropMomentum();
        std::cout << "done. \n";

        std::cout << ">> Testing save and load of a neural network with predefined weights... ";
        m_YANNL_UnitTest.saveAndLoadNetwork();
        std::cout << "done. \n";
    }

private:
    static YANNL_UnitTests m_YANNL_UnitTest;

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
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

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
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net.propagateForward({ 0.05, 0.1 });
        net.propagateBackward({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutInput0_4()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

        NeuralNetwork net(2, 0.5, 0, true, 18);
        net.addDropoutLayer(0.4);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackward({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutInput1_0()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

        NeuralNetwork net(2, 0.5);
        net.addDropoutLayer(1.0);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackward({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutHidden0_4()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

        NeuralNetwork net(2, 0.5, 0, true, 18);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addDropoutLayer(0.4);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackward({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDropoutHidden1_0()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

        NeuralNetwork net(2, 0.5, 0, true, 18);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addDropoutLayer(1.0);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "MSE: " << net.calcError({ 0.01, 0.99 }) << "\n";

        net.propagateBackward({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void backPropDefinedWeightsFor10000epochs()
    {
        std::ostringstream os;
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

        NeuralNetwork net(2, 0.5);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        for (size_t n = 0; n < 10000; n++)
        {
            net.propagateForward({ 0.05, 0.1 });
            net.propagateBackward({ 0.01, 0.99 });

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
        std::stringstream is = readExpectedResultFile("test/" + std::string(__func__) + ".txt");

        NeuralNetwork net(2, 0.5, 0.4);
        net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net.inspect(os);

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "Error: " << net.calcError({ 0.01, 0.99 }) << "\n";
        net.propagateBackward({ 0.01, 0.99 });

        os << "Output: " << net.propagateForward({ 0.05, 0.1 }) << "\n";
        os << "Error: " << net.calcError({ 0.01, 0.99 }) << "\n";
        net.propagateBackward({ 0.01, 0.99 });
        net.inspect(os);

        compareLineByLine(__func__, os.str(), is.str());
    }

    void saveAndLoadNetwork()
    {
        NeuralNetwork net1(2, 0.5);
        net1.addDropoutLayer(0.0);
        net1.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
        net1.addDropoutLayer(0.0);
        net1.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackward({ 0.01, 0.99 });
        net1.propagateForward({ 0.05, 0.1 });
        net1.propagateBackward({ 0.01, 0.99 });

        net1.saveToFile("output/net1.txt");
        NeuralNetwork net2 = NeuralNetwork::loadFromFile("output/net1.txt");
        net2.saveToFile("output/net2.txt");

        {
            std::stringstream is1 = readExpectedResultFile("output/net1.txt");
            std::stringstream is2 = readExpectedResultFile("output/net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }

        net1.propagateBackward({ 0.01, 0.99 });
        net2.propagateBackward({ 0.01, 0.99 });
        net1.saveToFile("output/net1.txt");
        net2.saveToFile("output/net2.txt");
        NeuralNetwork net1b = NeuralNetwork::loadFromFile("output/net1.txt");
        NeuralNetwork net2b = NeuralNetwork::loadFromFile("output/net2.txt");
        net1b.saveToFile("output/net1.txt");
        net2b.saveToFile("output/net1.txt");

        {
            std::stringstream is1 = readExpectedResultFile("output/net1.txt");
            std::stringstream is2 = readExpectedResultFile("output/net2.txt");

            compareLineByLine(__func__, is1.str(), is2.str());
        }
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
                    << "Assertion failed on line " << lineN << " of " << callingFunction << ": \n"
                    << "Provided: " << l1 << "\n"
                    << "Expected: " << l2 << std::endl;
                abort();
            }

            lineN++;
        }
    }
};

YANNL_UnitTests YANNL_UnitTests::m_YANNL_UnitTest;


int main(int argc, char* argv[])
{
    YANNL_UnitTests::execExceptionTests();
    YANNL_UnitTests::execNeuralNetworkTests();

    try
    {
        // First test with classifier and cross-entropy
        if (false)
        {
            NeuralNetwork net(2, 0.5);
            net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55} }, 0.6);

            net.inspect(std::cout);
            std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
            std::cout << "CEE: " << net.calcError({ 0.01, 0.99 }) << std::endl;

            net.propagateBackward({ 0.01, 0.99 });
            net.inspect(std::cout);
        }

        // Classifier and cross-entropy with 3 outputs
        if (false)
        {
            NeuralNetwork net(2, 0.5);
            net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55}, {0.8, 0.4} }, 0.6);

            std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
            std::cout << "CEE: " << net.calcError({ 0.01, 0.99, 0.82 }) << std::endl;

            net.propagateBackward({ 0.01, 0.99, 0.82 });
            net.inspect(std::cout);
        }

        // XOR with output regression layer
        if (false)
        {
            NeuralNetwork net(2, 0.5);
            net.addHiddenLayer(5, ActivationFunctions::Logistic);
            net.addOutputRegressionLayer(1, ActivationFunctions::Logistic); // Output layer

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
                    net.propagateBackward(trainingSet.second);
                }
            }

            for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
            {
                std::cout << "Input: " << testSet.first
                    << "  Output: " << net.propagateForward(testSet.first)
                    << "  Expected: " << testSet.second << std::endl;
            }
        }

        // Read MNIST image file
        if (false)
        {
            MnistReader::ImageContainer images;
            MnistReader::MnistFileAttrs attrs;

            attrs = MnistReader::readMnist("data/t10k-images.idx3-ubyte", images);
            MnistReader::displayMnist(images, std::cout, 0, 10, attrs);

            for (auto& pixel : images[0])
            {
                std::cout << (int)pixel << " ";
            }

            std::cout << std::endl;

            MnistReader::NormalizedImageContainer normImages = MnistReader::normalize(images);

            for (auto& pixel : normImages[0])
            {
                std::cout << pixel << " ";
            }

            std::cout << std::endl;
        }

        // Read MNIST label file
        if (false)
        {
            MnistReader::LabelContainer labels;
            int32_t count;

            count = MnistReader::readMnist("data/t10k-labels.idx1-ubyte", labels);
            MnistReader::displayMnist(labels, std::cout, 9990, 10002, count);
        }

        // Read MNIST files and train network
        if (false)
        {
            ShowConsoleCursor(false);
            constexpr size_t kBarWidth = 50;

            // Read training images
            std::cout << "Opening training image file..." << std::endl;
            MnistReader::ImageContainer trainImages;
            const MnistReader::MnistFileAttrs trainAttrs(MnistReader::readMnist("data/train-images.idx3-ubyte", trainImages));
            MnistReader::NormalizedImageContainer trainNormImages(MnistReader::normalize(trainImages));
            std::cout << "Number of images: " << trainAttrs.count << "\n"
                << "Dimensions of images: ( " << trainAttrs.rowsN << " x " << trainAttrs.colsN << " )" << std::endl;

            // Read training labels
            std::cout << "Opening training label file..." << std::endl;
            MnistReader::LabelContainer trainLabels;
            const int32_t trainCount(MnistReader::readMnist("data/train-labels.idx1-ubyte", trainLabels));
            std::cout << "Number of labels: " << trainCount << std::endl;

            if (trainAttrs.count != trainCount)
            {
                throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                    << "[Train network/training set] Input and output sizes are inconsistent: "
                    << "image's set is " << trainAttrs.count << " label's set is "
                    << trainCount << ".").str()
                );
            }

            // Build neural network with 1 hidden layer of 128 and 1 output layer of 10 labels
            std::cout << "Setting up the neural network..." << std::endl;
            NeuralNetwork net(1 * 28 * 28, 0.0001, 0.4);
            net.addDropoutLayer(0.25);
            net.addHiddenLayer(392, ActivationFunctions::ISRLU);
            net.addDropoutLayer(0.5);
            net.addOutputRegressionLayer(10, ActivationFunctions::Tanh);
            size_t epochN = 3;
            std::cout << "Done." << std::endl;

            // Train the network with 3 epochs
            std::cout << "Start training the network on " << trainCount << " images "
                << "for " << epochN << " epochs..." << std::endl;

            for (size_t epoch = 0; epoch < epochN; epoch++)
            {
                std::cout << "Epoch " << (epoch + 1) << " / " << epochN << std::endl;
                char current_pos_char = '/';

                // For each set of image and label

                for (size_t n = 0; n < static_cast<size_t>(trainCount); n++)
                {
                    if (n % 50 == 0)
                    {
                        if (current_pos_char == '/') { current_pos_char = '\\'; }
                        else { current_pos_char = '/'; }
                    }

                    std::cout << (n + 1) << " / " << trainCount << " [ ";
                    double progress = n * 100.0 / trainCount;
                    size_t position = static_cast<size_t>(progress / 100.0 * kBarWidth);

                    net.propagateForward(trainNormImages[n]);
                    net.propagateBackward(convertLabelToVect(trainLabels[n], 0, 9));

                    for (size_t i = 0; i < kBarWidth; i++)
                    {
                        if (i < position) { std::cout << "="; }
                        else if (i == position) { std::cout << current_pos_char; }
                        else { std::cout << "_"; }
                    }

                    std::cout << " ] " << static_cast<int>(progress) << "% | Error: ";

                    if (n % 100 == 0)
                    {
                        std::cout << std::fixed << std::setprecision(4)
                            << net.calcError(convertLabelToVect(trainLabels[n], 0, 9));
                    }

                    std::cout << "\r";
                    std::cout.flush();
                }

                std::cout << trainCount << " / " << trainCount << " [ ";

                for (size_t i = 0; i < kBarWidth; i++)
                {
                    std::cout << "=";
                }

                std::cout << " ] 100% | Error: "
                    << net.calcError(convertLabelToVect(trainLabels.back(), 0, 9)) << std::endl;
                std::cout.unsetf(std::ios_base::fixed);
            }

            net.saveToFile("output/YANNL_net.txt");
            std::cout << "Network trained and saved." << std::endl;
        }

        // Load network and test MNIST file
        if (false)
        {
            ShowConsoleCursor(false);
            constexpr size_t kBarWidth = 50;

            // Read test images
            std::cout << "Opening test image file..." << std::endl;
            MnistReader::ImageContainer testImages;
            const MnistReader::MnistFileAttrs testAttrs(MnistReader::readMnist("data/t10k-images.idx3-ubyte", testImages));
            MnistReader::NormalizedImageContainer testNormImages(MnistReader::normalize(testImages));
            std::cout << "Number of images: " << testAttrs.count << "\n"
                << "Dimensions of images: ( " << testAttrs.rowsN << " x " << testAttrs.colsN << " )" << std::endl;

            // Read test labels
            std::cout << "Opening test label file..." << std::endl;
            MnistReader::LabelContainer testLabels;
            const int32_t testCount(MnistReader::readMnist("data/t10k-labels.idx1-ubyte", testLabels));
            std::cout << "Number of labels: " << testCount << std::endl;

            if (testAttrs.count != testCount)
            {
                throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                    << "[Train network/test set] Input and output sizes are inconsistent: "
                    << "image's set is " << testAttrs.count << " label's set is "
                    << testCount << ".").str()
                );
            }

            // Load the network
            const std::string path = "output/YANNL_net.txt";
            std::cout << "Loading neural network from file " << path << "..." << std::endl;
            NeuralNetwork net = NeuralNetwork::loadFromFile(path);
            std::cout << "Done." << std::endl;;

            // Validate the network

            std::cout << "Start validating the network on " << testCount << " images..." << std::endl;
            size_t passed = 0;

            for (size_t n = 0; n < static_cast<size_t>(testCount); n++)
            {
                std::cout << (n + 1) << " / " << testCount << " [ ";
                double progress = n * 100.0 / testCount;
                size_t position = static_cast<size_t>(progress / 100.0 * kBarWidth);
                constexpr bool ignoreDropout = true;

                net.propagateForward(testNormImages[n], ignoreDropout);

                if (net.probableClass() == testLabels[n])
                {
                    passed++;
                }

                for (size_t i = 0; i < kBarWidth; i++)
                {
                    if (i < position) { std::cout << "="; }
                    else if (i == position) { std::cout << "/"; }
                    else { std::cout << "_"; }
                }

                std::cout << " ] " << static_cast<int>(progress) << "% \r";
                std::cout.flush();
            }

            std::cout << testCount << " / " << testCount << " [ ";

            for (size_t i = 0; i < kBarWidth; i++)
            {
                std::cout << "=";
            }

            std::cout << " ] 100%" << std::endl;
            std::cout << "Validation results: passed " << passed << " / " << testCount << " "
                << "( accuracy " << (passed * 100.0 / testCount) << "% )." << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cout << "Exception! " << e.what() << "\n";
    }
}
