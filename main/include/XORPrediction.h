#ifndef YANNL_XOR_PREDICTION_H
#define YANNL_XOR_PREDICTION_H

#include <vector>

namespace YANNL
{

std::vector<std::pair<std::vector<double>, double>> getXORTrainingSet();

void xorTrainTestManualNN();

void xorTrainTestMLPRegressor();

}

#endif // YANNL_XOR_PREDICTION_H
