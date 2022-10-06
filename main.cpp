#include "NeuralNetwork.h"
#include "Utils.h"
#include "MnistReader.h"

#include <iomanip> // std::setprecision
#include <cassert> // assert for testing purpose

using namespace YANNL;

int main(int argc, char* argv[])
{
    try
    {
        // Verify exceptions thrown
        if (false)
        {
            try
            {
                std::cout << "Add hidden layer after an output layer" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputClassificationLayer(2);
                net.addHiddenLayer(2, ActivationFunctions::Tanh);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add hidden layer with predefined weights after an output layer" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputClassificationLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } });
                net.addHiddenLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Tanh);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add hidden layer with inconsistent weights" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ {0.4, 0.45}, {0.5, 0.55, 0.1} }, ActivationFunctions::Tanh);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add hidden layer with inconsistent weights" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ {0.4, 0.45}, {0.5} }, ActivationFunctions::Tanh);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add output classification layer after an output layer" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputRegressionLayer(2, ActivationFunctions::Tanh);
                net.addOutputClassificationLayer(2);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add output classification layer with predefined weights after an output layer" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputRegressionLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Tanh);
                net.addOutputClassificationLayer(2);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add output classification layer with inconsistent weights" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55, 0.1} });
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add output regression layer after an output layer" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputRegressionLayer(2, ActivationFunctions::Tanh);
                net.addOutputRegressionLayer(2, ActivationFunctions::Tanh);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add output regression layer with predefined weights after an output layer" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputRegressionLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Tanh);
                net.addOutputRegressionLayer(2, ActivationFunctions::Tanh);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Add output regression layer with inconsistent weights" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55, 0.1} }, ActivationFunctions::Tanh);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Propagating forward with no layers" << std::endl;
                NeuralNetwork net(2, 0.5);
                std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Propagating forward with no output layers" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Propagating forward with no output layers" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                net.addHiddenLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
                std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Propagating forward with inconsistent input size" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
                std::cout << "Output: " << net.propagateForward({ 0.05, 0.1, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Propagating forward with inconsistent input size" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
                std::cout << "Output: " << net.propagateForward({ 0.05 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Get probable class with no output layer" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                std::cout << "Probable class: " << net.probableClass() << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Calculate error with no layers" << std::endl;
                NeuralNetwork net(2, 0.5);
                std::cout << "Error: " << net.calcError({ 0.05, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Calculate error with no output layers" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                std::cout << "Error: " << net.calcError({ 0.05, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Calculate error with inconsistent output size (regression)" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);
                std::cout << "Error: " << net.calcError({ 0.05, 0.1, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Calculate error with inconsistent output size (classification)" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                net.addOutputClassificationLayer({ {0.4, 0.45}, {0.5, 0.55} }, 0.6);
                std::cout << "Error: " << net.calcError({ 0.05, 0.1, 0.1 }) << std::endl;
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Propagate backward with no layers" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.propagateBackward({ 0.01, 0.99 });
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Propagate backward with no output layers" << std::endl;
                NeuralNetwork net(2, 0.5);
                net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
                net.propagateBackward({ 0.01, 0.99 });
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }

            try
            {
                std::cout << "Inaccessible file when loading network" << std::endl;
                NeuralNetwork net = NeuralNetwork::loadFromFile("dummyfile.txt");
                net.inspect(std::cout);
                assert(false);
            }
            catch (std::exception& e) { std::cout << "Exception! " << e.what() << std::endl; }
        }

        // First test with bias
        if (true)
        {
            NeuralNetwork net(2, 0.5);
            net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

            net.inspect(std::cout);
            std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
            std::cout << "MSE: " << net.calcError({ 0.01, 0.99 }) << std::endl;

            net.propagateBackward({ 0.01, 0.99 });
            net.inspect(std::cout);

            net.saveToFile("output/MM_NN_net.txt");
        }

        // With epochs
        if (false)
        {
            NeuralNetwork net(2, 0.5);
            net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

            for (size_t n = 0; n < 10000; n++)
            {
                net.propagateForward({ 0.05, 0.1 });
                net.propagateBackward({ 0.01, 0.99 });

                if (n == 0)
                {
                    std::cout << "Error after 1 case: " << net.calcError({ 0.01, 0.99 }) << std::endl;
                }
            }

            std::cout << "Error after 10000 cases: " << net.calcError({ 0.01, 0.99 }) << std::endl;
        }

        // Load first network
        if (false)
        {
            const std::string path = "output/MM_NN_net.txt";
            std::cout << "Loading neural network from file " << path << "..." << std::endl;
            NeuralNetwork net = NeuralNetwork::loadFromFile(path);
            std::cout << "Done." << std::endl;

            net.saveToFile("output/MM_NN_net_compare.txt");
        }

        // First test with momentum and bias
        if (false)
        {
            NeuralNetwork net(2, 0.5, 0.4);
            net.addHiddenLayer({ { 0.15, 0.2 }, { 0.25, 0.3 } }, ActivationFunctions::Logistic, 0.35);
            net.addOutputRegressionLayer({ {0.4, 0.45}, {0.5, 0.55} }, ActivationFunctions::Logistic, 0.6);

            net.inspect(std::cout);

            std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
            std::cout << "Error: " << net.calcError({ 0.01, 0.99 }) << std::endl;
            net.propagateBackward({ 0.01, 0.99 });

            std::cout << "Output: " << net.propagateForward({ 0.05, 0.1 }) << std::endl;
            std::cout << "Error: " << net.calcError({ 0.01, 0.99 }) << std::endl;
            net.propagateBackward({ 0.01, 0.99 });
            net.inspect(std::cout);
            net.saveToFile("output/MM_NN_net.txt");
        }

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
            net.addHiddenLayer(128, ActivationFunctions::Relu);
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

                net.propagateForward(testNormImages[n]);

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
        std::cout << "Exception! " << e.what() << std::endl;
    }
}
