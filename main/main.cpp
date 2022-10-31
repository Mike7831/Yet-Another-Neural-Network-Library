//#include "MnistReader.h"
//#include "MnistPrediction.h"
#include "MLP.h"

using namespace YANNL;

int main(int argc, char* argv[])
{
    try
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
                net.propagateBackward(trainingSet.second);
            }
        }

        std::cout << "With the regular neural network: \n";

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            std::cout << "Input: " << testSet.first
                << "  Output: " << net.propagateForward(testSet.first)
                << "  Expected: " << testSet.second << "\n";
        }

        MLPRegressor mlp({ 5 },             // hidden_layer_sizes
            ActivationFunctions::Logistic,  // activation
            Solvers::SGD,                   // solver
            LearningRate::Adaptive,         // learning_rate
            0.5,                            // learning_rate_init
            0.5,                            // power_t
            10000,                          // max_iter
            true,                           // use_random_state
            10,                             // random_state
            1.0E-4,                         // tol
            true,                           // verbose
            0.9,                            // momentum
            true,                           // early_stopping
            10                              // n_iter_no_change
        );
        mlp.fit({ {0, 0},  {0, 1},  {1, 0},  {1, 1} }, { 0, 1, 1, 0 });

        std::cout << "With the MLPRegressor: \n";

        for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
        {
            std::cout << "Input: " << testSet.first
                << "  Output: " << mlp.predict(testSet.first)
                << "  Expected: " << testSet.second << "\n";
        }

        //mnistTrain("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", "../output/mnist-nn.txt");
        //mnistTest("../output/mnist-nn.txt", "../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");
    }
    catch (std::exception& e)
    {
        std::cout << "Exception! " << e.what() << "\n";
    }
}
