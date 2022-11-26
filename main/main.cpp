//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#include "MnistPrediction.h"
#include "XORPrediction.h"
#include "IrisClassification.h"
#include "MLP.h"

using namespace YANNL;

int main(int argc, char* argv[])
{
    try
    {
        std::cout << "Test iris flower classification with MLPClassifier vs. manually built classifier \n"
            << "================================================================================== \n\n";
        IrisClassification irisClassification;
        irisClassification.irisClassificationTrainTestManualNN("../data/iris_flowers.csv");
        irisClassification.irisClassificationTrainTestMLPClassifier("../data/iris_flowers.csv");
        std::cout  << "================================================================================== \n\n";
        
        std::cout << "Test XOR prediction with MLPRegressor vs. manually built regressors \n"
            << "================================================================================== \n\n";
        xorTrainTestManualNN();
        xorTrainTestMLPRegressor();
        std::cout << "================================================================================== \n\n";
        
        std::cout << "Test prediction of MNIST handwritten digits (0-9) \n"
            << "================================================================================== \n\n";
        mnistTrain("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", "../output/mnist-nn.txt");
        mnistTest("../output/mnist-nn.txt", "../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");
        std::cout << "================================================================================== \n\n";
    }
    catch (std::exception& e)
    {
        std::cout << "Exception! " << e.what() << "\n";
    }
}
