#include "MnistPrediction.h"
#include "XORPrediction.h"
#include "IrisClassification.h"
#include "MLP.h"

using namespace YANNL;

int main(int argc, char* argv[])
{
    try
    {
        std::cout << "================================================================================== \n"
            << "Test iris flower classification with MLPClassifier vs. manually built classifier \n"
            << "================================================================================== \n\n";
        irisClassificationTrainTestManualNN();
        std::cout << "================================================================================== \n"
            << "================================================================================== \n\n";

        std::cout << "================================================================================== \n"
            << "Test XOR prediction with MLPRegressor vs. manually built regressors \n"
            << "================================================================================== \n\n";
        xorTrainTestManualNN();
        xorTrainTestMLPRegressor();
        std::cout << "================================================================================== \n"
            << "================================================================================== \n\n";

        //mnistTrain("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", "../output/mnist-nn.txt");
        //mnistTest("../output/mnist-nn.txt", "../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");
    }
    catch (std::exception& e)
    {
        std::cout << "Exception! " << e.what() << "\n";
    }
}
