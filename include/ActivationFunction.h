#ifndef YANNL_ACTIVATION_FUNCTION_H
#define YANNL_ACTIVATION_FUNCTION_H

#include <string>   // std::string
#include <cmath>    // std::exp
#include <memory>   // std::unique_ptr & std::shared_ptr
#include <typeinfo> // typeid

namespace YANNL
{
enum class ActivationFunctions
{
    Logistic = 0,
    Relu,
    Tanh,
    Identity
};

class ActivationFunction
{
public:
    static std::shared_ptr<ActivationFunction> build(ActivationFunctions afunc);
    
    virtual ~ActivationFunction() = default;

    virtual double calc(const double& x) const = 0;
    virtual double calcDerivate(const double& x) const = 0;

    std::string name() const
    {
        return typeid(*this).name();
    }
};

class Logistic : public ActivationFunction
{
public:
    double calc(const double& x) const override { return (1.0 / (1.0 + std::exp(-x))); }
    double calcDerivate(const double& x) const override { return x * (1.0 - x); }
};

class Relu : public ActivationFunction
{
public:
    double calc(const double& x) const override { return std::max(0.0, x); }
    double calcDerivate(const double& x) const override { return 0.0; }
};

class Tanh : public ActivationFunction
{
public:
    double calc(const double& x) const override { return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x)); }
    double calcDerivate(const double& x) const override { return 1.0 - std::pow(calc(x), 2); }
};

class Identity : public ActivationFunction
{
public:
    double calc(const double& x) const override { return x; }
    double calcDerivate(const double& x) const override { return 1.0; }
};

}

#endif // YANNL_ACTIVATION_FUNCTION_H