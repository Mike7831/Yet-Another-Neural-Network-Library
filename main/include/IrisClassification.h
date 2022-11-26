//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#ifndef YANNL_IRIS_CLASSIFICATION_H
#define YANNL_IRIS_CLASSIFICATION_H

#include <string>   // std::string
#include <vector>   // std::vector
#include <map>      // std::map

namespace YANNL
{

class IrisClassification
{
    typedef std::vector<double> t_IrisData;
    typedef uint8_t t_IrisCls;

public:
    void irisClassificationTrainTestManualNN(const std::string& irisDataSetPath);
    void irisClassificationTrainTestMLPClassifier(const std::string& irisDataSetPath);

private:
    const size_t kBarWidth = 50;
    const size_t kEpochN = 1000;
    const double kLearningRate = 0.001;
    const double kMomentum = 0.9;
    const std::map<size_t, std::string> kIrisClassMap{
        { 0, "iris_setosa    " },
        { 1, "iris_versicolor" },
        { 2, "iris_virginica " }
    };

    std::vector<std::pair<t_IrisData, t_IrisCls>> m_IrisData;

    void loadIrisDataset(const std::string& irisDataSetPath);
};

}

#endif // YANNL_IRIS_CLASSIFICATION_H
