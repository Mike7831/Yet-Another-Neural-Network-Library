#include "ActivationFunction.h"

using namespace YANNL;

std::shared_ptr<ActivationFunction> ActivationFunction::build(ActivationFunctions afunc)
{
    switch (afunc)
    {
    case ActivationFunctions::Logistic:
        return std::make_unique<Logistic>();
        break;

    case ActivationFunctions::Relu:
        return std::make_unique<Relu>();
        break;

    case ActivationFunctions::Tanh:
        return std::make_unique<Tanh>();
        break;

    default:
        return std::make_unique<Identity>();
        break;
    }
}
