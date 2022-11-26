//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#include "IrisClassification.h"
#include "MLP.h"

#include <iomanip>  // std::setprecision
#include <cassert>  // assert()

using namespace YANNL;

namespace YANNL
{

void IrisClassification::loadIrisDataset(const std::string& irisDataSetPath)
{
    Utils::ShowConsoleCursor(false);

    std::cout << "Loading " << irisDataSetPath << " file... ";

    char c; // To eat the commas

    double sepalLength = 0.0, sepalWidth = 0.0, petalLength = 0.0, petalWidth = 0.0;
    std::string irisClass;

    std::ifstream file(irisDataSetPath);
    std::string line;

    std::getline(file, line); // Ignore header line

    while (file >> sepalLength >> c >> sepalWidth >> c >> petalLength >> c
        >> petalWidth >> c >> irisClass)
    {
        t_IrisCls cls = 0;

        if (irisClass == "iris_setosa") { cls = 0; }
        else if (irisClass == "iris_versicolor") { cls = 1; }
        else if (irisClass == "iris_virginica") { cls = 2; }

        m_IrisData.push_back({
            t_IrisData({ sepalLength, sepalWidth, petalLength, petalWidth }),
            t_IrisCls(cls)
            });

    }

    assert(m_IrisData.size() == 150);

    std::cout << "done. \n";

    std::cout << "Shuffling the data... ";
    auto rng = std::default_random_engine{};
    std::shuffle(m_IrisData.begin(), m_IrisData.end(), rng);
    std::cout << "done. \n";
}

void IrisClassification::irisClassificationTrainTestManualNN(const std::string& irisDataSetPath)
{
    if (m_IrisData.empty())
    {
        loadIrisDataset(irisDataSetPath);
    }

    std::cout << "Building and training the neural network (manually built)... \n";

    NeuralNetwork net(4, kLearningRate, kMomentum, true, 10); // Random weights but with a fixed seed
    net.addHiddenLayer(3, ActivationFunctions::Logistic);
    net.addHiddenLayer(3, ActivationFunctions::Logistic);
    net.addOutputClassificationLayer(3);

    double error = 0.0;
    char current_pos_char = '/';

    for (size_t epoch = 0; epoch < kEpochN; epoch++)
    {
        std::cout << "Epoch " << (epoch + 1) << " / " << kEpochN << " [ ";
        double progress = epoch * 100.0 / kEpochN;
        size_t position = static_cast<size_t>(progress / 100.0 * kBarWidth);

        if (kEpochN >= kBarWidth && epoch % (kEpochN / kBarWidth) == 0)
        {
            if (current_pos_char == '/') { current_pos_char = '\\'; }
            else { current_pos_char = '/'; }
        }

        for (size_t i = 0; i < kBarWidth; i++)
        {
            if (i < position) { std::cout << "="; }
            else if (i == position) { std::cout << current_pos_char; }
            else { std::cout << "_"; }
        }

        std::cout << " ] " << static_cast<int>(progress) << "% | Error: ";

        error = 0.0;

        for (const std::pair<t_IrisData, t_IrisCls>& irisItem : m_IrisData)
        {
            std::vector<double> expectedOutput = Utils::convertLabelToVect(irisItem.second, 0, 2);
            net.propagateForward(irisItem.first);
            net.propagateBackward(expectedOutput);
            error += net.calcError(expectedOutput);
        }

        if (kEpochN >= kBarWidth && epoch % (kEpochN / kBarWidth) == 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                << (error / m_IrisData.size());
        }

        std::cout << "\r";
        std::cout.flush();
    }

    std::cout << "Epoch " << kEpochN << " / " << kEpochN << " [ ";

    for (size_t i = 0; i < kBarWidth; i++)
    {
        std::cout << "=";
    }

    std::cout << " ] 100% | Error: " << (error / m_IrisData.size()) << "\n";

    std::cout << "done. \n";

    std::cout << "Testing the network with the same dataset as the one for training... \n";

    size_t passed = 0;

    for (const std::pair<t_IrisData, t_IrisCls>& irisItem : m_IrisData)
    {
        net.propagateForward(irisItem.first);

        size_t expected = irisItem.second;
        size_t output = net.probableClass();

        bool correct = output == expected;
        if (correct) { passed++; }

        std::cout << std::fixed << std::setprecision(1) << "Input: " << irisItem.first
            << "  Output: " << kIrisClassMap.find(output)->second
            << "  Expected: " << kIrisClassMap.find(expected)->second
            << "  " << (correct ? "[X]" : "[ ]")
            << "  Error: " << net.calcError(Utils::convertLabelToVect(irisItem.second, 0, 2)) << "\n";
    }

    std::cout.unsetf(std::ios_base::fixed);
    std::cout << std::setprecision(4);

    double accuracy = passed * 100.0 / m_IrisData.size();
    std::cout << "done with accuracy of " << accuracy << " %. \n";
}

void IrisClassification::irisClassificationTrainTestMLPClassifier(const std::string& irisDataSetPath)
{
    if (m_IrisData.empty())
    {
        loadIrisDataset(irisDataSetPath);
    }

    std::cout << "Building and training the neural network (MLPClassifier)... \n";

    MLPClassifer mlp({ 3, 3 },          // hidden_layer_sizes
        ActivationFunctions::Logistic,  // activation
        Solvers::SGD,                   // solver
        LearningRate::Constant,         // learning_rate
        kLearningRate,                  // learning_rate_init
        0.5,                            // power_t
        kEpochN,                        // max_iter
        true,                           // use_random_state
        10,                             // random_state
        1.0E-5,                         // tol
        true,                           // verbose
        kMomentum,                      // momentum
        false,                          // early_stopping
        10                              // n_iter_no_change
    );


    std::vector<t_IrisData> X;
    std::vector<t_IrisCls> y;

    for (size_t n = 0; n < m_IrisData.size(); n++)
    {
        X.push_back(m_IrisData[n].first);
        y.push_back(m_IrisData[n].second);
    }

    mlp.fit(X, y);

    std::cout << "done. \n";

    mlp.inspect(std::cout);

    std::cout << "Testing the network with the same dataset as the one for training... \n";

    size_t passed = 0;

    for (const std::pair<t_IrisData, t_IrisCls>& irisItem : m_IrisData)
    {
        size_t expected = irisItem.second;
        size_t output = mlp.predict(irisItem.first);

        bool correct = output == expected;
        if (correct) { passed++; }

        std::cout << std::fixed << std::setprecision(1) << "Input: " << irisItem.first
            << "  Output: " << kIrisClassMap.find(output)->second
            << "  Expected: " << kIrisClassMap.find(expected)->second
            << "  " << (correct ? "[X]" : "[ ]") << "\n";
    }

    std::cout.unsetf(std::ios_base::fixed);
    std::cout << std::setprecision(4);

    double accuracy = passed * 100.0 / m_IrisData.size();
    std::cout << "done with accuracy of " << accuracy << " %. \n";
}

}