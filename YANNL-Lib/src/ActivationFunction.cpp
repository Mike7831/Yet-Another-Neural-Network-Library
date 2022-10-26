#include "ActivationFunction.h"

using namespace YANNL;

std::shared_ptr<ActivationFunction> ActivationFunction::build(ActivationFunctions afunc)
{
    switch (afunc)
    {
    case ActivationFunctions::Logistic:
        return std::make_unique<Logistic>();
        break;

    case ActivationFunctions::Tanh:
        return std::make_unique<Tanh>();
    break;
    
    case ActivationFunctions::ReLU:
        return std::make_unique<ReLU>();
        break;

    case ActivationFunctions::ISRLU:
        return std::make_unique<ISRLU>();
        break;

    default:
        return std::make_unique<Identity>();
        break;
    }
}
