#include "NeuralNetwork.h"
#include "Utils.h"
#include "MnistReader.h"

//#include "MLP.h"

#include <iomanip> // std::setprecision

using namespace YANNL;


int main(int argc, char* argv[])
{
    try
    {
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
