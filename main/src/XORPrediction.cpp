//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#include "XORPrediction.h"
#include "MLP.h"

using namespace YANNL;

namespace YANNL
{

std::vector<std::pair<std::vector<double>, double>> XORPrediction::getXORTrainingSet()
{
    return {
        { {0, 0}, 0},
        { {0, 1}, 1},
        { {1, 0}, 1},
        { {1, 1}, 0}
    };
}

void XORPrediction::xorTrainTestManualNN()
{
    std::cout << "Building and training the neural network (manually built)... \n";

    NeuralNetwork net(2, 0.5, 0.9, true, 10); // Random weights but with a fixed seed
    net.addHiddenLayer(5, ActivationFunctions::Logistic);
    net.addOutputRegressionLayer(1, ActivationFunctions::Logistic);

    std::vector<std::pair<std::vector<double>, double>> trainingSets = getXORTrainingSet();

    for (size_t epoch = 0; epoch < 10000; epoch++)
    {
        for (const std::pair<std::vector<double>, double>& trainingSet : trainingSets)
        {
            net.propagateForward(trainingSet.first);
            net.propagateBackwardAndUpdateWeights(trainingSet.second);
        }
    }

    std::cout << "Testing... \n";

    for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
    {
        std::cout << "Input: " << testSet.first
            << "  Output: " << net.propagateForward(testSet.first)
            << "  Expected: " << testSet.second << "\n";
    }

    std::cout << "done. \n";
}

void XORPrediction::xorTrainTestMLPRegressor()
{
    std::vector<std::pair<std::vector<double>, double>> trainingSets = getXORTrainingSet();

    std::cout << "Building and training the neural network (MLPRegressor)... \n";

    MLPRegressor mlp({ 5 },             // hidden_layer_sizes
        ActivationFunctions::Logistic,  // activation
        Solvers::SGD,                   // solver
        false,                          // use_batch_size
        1,                              // batch_size
        LearningRate::Constant,         // learning_rate
        0.5,                            // learning_rate_init
        0.5,                            // power_t
        10000,                          // max_iter
        true,                           // use_random_state
        10,                             // random_state
        1.0E-5,                         // tol
        true,                           // verbose
        0.9,                            // momentum
        false,                          // early_stopping
        10                              // n_iter_no_change
    );

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (size_t n = 0; n < trainingSets.size(); n++)
    {
        X.push_back(trainingSets[n].first);
        y.push_back(trainingSets[n].second);
    }

    mlp.fit(X, y);

    std::cout << "Testing... \n";

    for (const std::pair<std::vector<double>, double>& testSet : trainingSets)
    {
        std::cout << "Input: " << testSet.first
            << "  Output: " << mlp.predict(testSet.first)
            << "  Expected: " << testSet.second << "\n";
    }

    std::cout << "done. \n";
}

}

