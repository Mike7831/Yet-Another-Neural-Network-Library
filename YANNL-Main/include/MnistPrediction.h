#ifndef YANNL_MNIST_PREDICTION_H
#define YANNL_MNIST_PREDICTION_H

#include <string>

namespace YANNL
{

void mnistTrain(const std::string& trainImagePath, const std::string& trainLabelPath,
    const std::string& outputPath);

void mnistTest(const std::string& networkPath, const std::string& testImagePath,
    const std::string& testLabelPath);

}

#endif // YANNL_MNIST_PREDICTION_H
