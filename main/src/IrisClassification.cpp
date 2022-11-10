#include "IrisClassification.h"
#include "MLP.h"

#include <iomanip>  // std::setprecision
#include <map>      // std::map
#include <cassert>  // assert()

using namespace YANNL;

namespace YANNL
{

void irisClassificationTrainTestManualNN()
{
    Utils::ShowConsoleCursor(false);

    constexpr size_t kBarWidth = 50;
    const char* kDataDir = "../data/";
    const std::map<size_t, std::string> kIrisClassMap{
        { 0, "iris_setosa    " },
        { 1, "iris_versicolor" },
        { 2, "iris_virginica " }
    };

    std::cout << "Loading iris_flowers.csv file... ";

    char c; // To eat the commas

    double sepalLength = 0.0, sepalWidth = 0.0, petalLength = 0.0, petalWidth = 0.0;
    std::string irisClass;
    typedef std::vector<double> t_IrisData;
    typedef std::vector<double> t_IrisCls;
    std::vector<std::pair<t_IrisData, t_IrisCls>> irisData;

    std::ifstream file(kDataDir + std::string("iris_flowers.csv"));
    std::string line;

    std::getline(file, line); // Ignore header line

    while (file >> sepalLength >> c >> sepalWidth >> c >> petalLength >> c
        >> petalWidth >> c >> irisClass)
    {
        irisData.push_back({
            t_IrisData({ sepalLength, sepalWidth, petalLength, petalWidth }),
            t_IrisCls({
            irisClass == "iris_setosa" ? 1.0 : 0.0,
            irisClass == "iris_versicolor" ? 1.0 : 0.0,
            irisClass == "iris_virginica" ? 1.0 : 0.0
                }) });
        assert(std::accumulate(irisData.back().second.cbegin(), irisData.back().second.cend(), 0.0) == 1.0);

    }

    assert(irisData.size() == 150);

    std::cout << "done. \n";

    std::cout << "Shuffling the data... ";
    auto rng = std::default_random_engine{};
    std::shuffle(irisData.begin(), irisData.end(), rng);
    std::cout << "done. \n";

    std::cout << "Building and training the neural network (manually built)... \n";

    NeuralNetwork net(4, 0.1, 0.0, true, 10); // Random weights but with a fixed seed
    net.addHiddenLayer(3, ActivationFunctions::Logistic);
    net.addHiddenLayer(3, ActivationFunctions::Logistic);
    net.addOutputClassificationLayer(3);

    size_t epochN = 1000;
    double error = 0.0;

    for (size_t epoch = 0; epoch < epochN; epoch++)
    {
        std::cout << "Epoch " << (epoch + 1) << " / " << epochN << " [ ";
        double progress = epoch * 100.0 / epochN;
        size_t position = static_cast<size_t>(progress / 100.0 * kBarWidth);

        char current_pos_char = '/';

        if (epoch % (epochN / 50) == 0)
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

        for (const std::pair<t_IrisData, t_IrisCls>& irisItem : irisData)
        {
            net.propagateForward(irisItem.first);
            net.propagateBackward(irisItem.second);
            error += net.calcError(irisItem.second);
        }

        if (epoch % (epochN / 50) == 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                << (error / irisData.size());
        }

        std::cout << "\r";
        std::cout.flush();
    }

    std::cout << "Epoch " << epochN << " / " << epochN << " [ ";

    for (size_t i = 0; i < kBarWidth; i++)
    {
        std::cout << "=";
    }

    std::cout << " ] 100% | Error: " << (error / irisData.size()) << "\n";

    std::cout << "done. \n";

    std::cout << "Testing the network with the same dataset as the one for training... \n";

    size_t passed = 0;

    for (const std::pair<t_IrisData, t_IrisCls>& irisItem : irisData)
    {
        net.propagateForward(irisItem.first);

        size_t expected = std::distance(irisItem.second.cbegin(),
            std::find_if(irisItem.second.cbegin(), irisItem.second.cend(),
                [](auto x)
                {
                    return x != 0.0;
                }));

        size_t output = static_cast<size_t>(net.probableClass());

        bool correct = output == expected;
        if (correct) { passed++; }

        std::cout << std::fixed << std::setprecision(1) << "Input: " << irisItem.first
            << "  Output: " << kIrisClassMap.find(output)->second
            << "  Expected: " << kIrisClassMap.find(expected)->second
            << "  " << (correct ? "[X]" : "[ ]")
            << "  Error: " << net.calcError(irisItem.second) << "\n";
    }

    std::cout.unsetf(std::ios_base::fixed);
    std::cout << std::setprecision(4);

    double accuracy = passed * 100.0 / irisData.size();
    std::cout << "done with accuracy of " << accuracy << " %. \n";
}

}