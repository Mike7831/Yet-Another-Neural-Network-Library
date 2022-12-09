//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#ifndef YANNL_MNIST_PREDICTION_H
#define YANNL_MNIST_PREDICTION_H

#include <string>

namespace YANNL
{

class MnistPrediction
{
public:
    void mnistTrain(const std::string& trainImagePath, const std::string& trainLabelPath,
        const std::string& outputPath);

    void mnistTest(const std::string& networkPath, const std::string& testImagePath,
        const std::string& testLabelPath);
};

}

#endif // YANNL_MNIST_PREDICTION_H
