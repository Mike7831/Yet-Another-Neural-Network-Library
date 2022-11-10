#include "MnistPrediction.h"
#include "NeuralNetwork.h"
#include "MnistReader.h"

#include <iomanip> // std::setprecision

using namespace YANNL;

namespace YANNL
{

void mnistTrain(const std::string& trainImagePath, const std::string& trainLabelPath,
    const std::string& outputPath)
{
    Utils::ShowConsoleCursor(false);
    constexpr size_t kBarWidth = 50;

    // Read training images
    std::cout << "Opening training image file... \n";
    MnistReader::ImageContainer trainImages;
    const MnistReader::MnistFileAttrs trainAttrs(MnistReader::readMnist(trainImagePath, trainImages));
    MnistReader::NormalizedImageContainer trainNormImages(MnistReader::normalize(trainImages));
    std::cout << "Number of images: " << trainAttrs.count << "\n"
        << "Dimensions of images: ( " << trainAttrs.rowsN << " x " << trainAttrs.colsN << " ) \n";

    // Read training labels
    std::cout << "Opening training label file... \n";
    MnistReader::LabelContainer trainLabels;
    const int32_t trainCount(MnistReader::readMnist(trainLabelPath, trainLabels));
    std::cout << "Number of labels: " << trainCount << "\n";

    if (trainAttrs.count != trainCount)
    {
        throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
            << "[Train network/training set] Input and output sizes are inconsistent: "
            << "image's set is " << trainAttrs.count << " label's set is "
            << trainCount << ".").str()
        );
    }

    // Build neural network with 1 hidden layer of 128 and 1 output layer of 10 labels
    std::cout << "Setting up the neural network... \n";
    NeuralNetwork net(1 * 28 * 28, 0.0001, 0.4);
    net.addHiddenLayer(128, ActivationFunctions::ReLU);
    net.addDropoutLayer(0.5);
    net.addOutputRegressionLayer(10, ActivationFunctions::Tanh);
    size_t epochN = 3;
    std::cout << "Done. \n";

    // Train the network with 3 epochs
    std::cout << "Start training the network on " << trainCount << " images "
        << "for " << epochN << " epochs... \n";

    for (size_t epoch = 0; epoch < epochN; epoch++)
    {
        std::cout << "Epoch " << (epoch + 1) << " / " << epochN << "\n";
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
            net.propagateBackward(Utils::convertLabelToVect(trainLabels[n], 0, 9));

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
                    << net.calcError(Utils::convertLabelToVect(trainLabels[n], 0, 9));
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
            << net.calcError(Utils::convertLabelToVect(trainLabels.back(), 0, 9)) << "\n";
        std::cout.unsetf(std::ios_base::fixed);
    }

    net.saveToFile(outputPath);
    std::cout << "Network trained and saved. \n";
}

void mnistTest(const std::string& networkPath, const std::string& testImagePath,
    const std::string& testLabelPath)
{
    Utils::ShowConsoleCursor(false);
    constexpr size_t kBarWidth = 50;

    // Read test images
    std::cout << "Opening test image file... \n";
    MnistReader::ImageContainer testImages;
    const MnistReader::MnistFileAttrs testAttrs(MnistReader::readMnist(testImagePath, testImages));
    MnistReader::NormalizedImageContainer testNormImages(MnistReader::normalize(testImages));
    std::cout << "Number of images: " << testAttrs.count << "\n"
        << "Dimensions of images: ( " << testAttrs.rowsN << " x " << testAttrs.colsN << " ) \n";

    // Read test labels
    std::cout << "Opening test label file... \n";
    MnistReader::LabelContainer testLabels;
    const int32_t testCount(MnistReader::readMnist(testLabelPath, testLabels));
    std::cout << "Number of labels: " << testCount << "\n";

    if (testAttrs.count != testCount)
    {
        throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
            << "[Train network/test set] Input and output sizes are inconsistent: "
            << "image's set is " << testAttrs.count << " label's set is "
            << testCount << ".").str()
        );
    }

    // Load the network
    std::cout << "Loading neural network from file " << networkPath << "... \n";
    NeuralNetwork net = NeuralNetwork::loadFromFile(networkPath);
    std::cout << "Done. \n";

    // Validate the network

    std::cout << "Start validating the network on " << testCount << " images... \n";
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

    std::cout << " ] 100% \n";
    std::cout << "Validation results: passed " << passed << " / " << testCount << " "
        << "( accuracy " << (passed * 100.0 / testCount) << "% ). \n";
}

}

