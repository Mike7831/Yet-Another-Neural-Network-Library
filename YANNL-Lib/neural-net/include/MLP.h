#ifndef YANNL_MLP_H
#define YANNL_MLP_H

#include "NeuralNetwork.h"

namespace YANNL
{

enum class Solvers
{
    SGD = 0
};

enum class LearningRate
{
    Constant = 0,
    InvScaling,
    Adaptive
};

class MLPRegressor
{
public:
    explicit MLPRegressor(
        const std::vector<size_t>& hidden_layer_sizes = { 100 },
        ActivationFunctions activation = ActivationFunctions::ReLU,
        Solvers solver = Solvers::SGD,
        LearningRate learning_rate = LearningRate::Constant,
        double learning_rate_init = 0.001,
        double power_t = 0.5,
        size_t max_iter = 200,
        float random_state = NAN,
        double tol = 0.0001,
        double momentum = 0.9,
        size_t n_iter_no_change = 10) :
        m_HiddenLayerSizes(hidden_layer_sizes),
        m_AFunc(activation),
        m_Solver(solver),
        m_LearningRateType(learning_rate),
        m_LearningRate(learning_rate_init),
        m_PowerT(power_t),
        m_MaxIterations(max_iter),
        m_Seed(random_state),
        m_OptimizationTolerance(tol),
        m_Momentum(momentum),
        m_IterNoChangeN(n_iter_no_change)
    {

    }

    void fit(const std::vector<std::vector<double>>& inputs,
        const std::vector<double>& expectedOuputs)
    {

    }

    double predict(const std::vector<double>& input) const
    {
        double output = 0.0;

        return output;
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& inputs) const
    {
        std::vector<double> outputs;

        return outputs;
    }

    void inspect(std::ostream& os) const
    {
        
    }

private:
    std::vector<size_t> m_HiddenLayerSizes;
    ActivationFunctions m_AFunc;
    Solvers m_Solver;
    LearningRate m_LearningRateType;
    double m_LearningRate;
    double m_PowerT;
    size_t m_MaxIterations;
    float m_Seed;
    double m_OptimizationTolerance;
    double m_Momentum;
    size_t m_IterNoChangeN;
};

}

#endif // YANNL_MLP_H
