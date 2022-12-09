//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#ifndef YANNL_XOR_PREDICTION_H
#define YANNL_XOR_PREDICTION_H

#include <vector>

namespace YANNL
{

class XORPrediction
{
public:
    void xorTrainTestManualNN();
    void xorTrainTestMLPRegressor();

private:
    std::vector<std::pair<std::vector<double>, double>> getXORTrainingSet();
};

}

#endif // YANNL_XOR_PREDICTION_H
