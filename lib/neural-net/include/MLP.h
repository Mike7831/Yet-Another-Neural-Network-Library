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
        bool use_random_state = false,
        unsigned int random_state = 0,
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
        m_UseSeed(use_random_state),
        m_Seed(random_state),
        m_OptimizationTolerance(tol),
        m_Momentum(momentum),
        m_IterNoChangeN(n_iter_no_change)
    {

    }

    void fit(const std::vector<std::vector<double>>& inputs,
        const std::vector<double>& expectedOuputs)
    {
        if (!inputs.empty())
        {
            // Check that output size is consistent with input size
            if (inputs.size() != expectedOuputs.size())
            {
                throw std::domain_error("Input and output size not consistent.");
            }

            // Check that all inputs are of the same

            size_t inputSize = inputs[0].size();

            for (size_t i = 1; i < inputs.size(); i++)
            {
                if (inputs[i].size() != inputSize)
                {
                    throw std::domain_error("Input size not consistent.");
                }
            }

            // Build neural network

            m_Net = std::make_unique<NeuralNetwork>(inputSize, m_LearningRate, m_Momentum, m_UseSeed, m_Seed);

            std::for_each(m_HiddenLayerSizes.cbegin(), m_HiddenLayerSizes.cend(),
                [&](size_t layerSize)
                {
                    m_Net->addHiddenLayer(layerSize, m_AFunc);
                });

            m_Net->addOutputRegressionLayer(1, m_AFunc);

            // Train neural network

            for (size_t n = 0; n < m_MaxIterations; n++)
            {
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    m_Net->propagateForward(inputs[i]);
                    m_Net->propagateBackward(expectedOuputs[i]);
                }
            }
        }
    }

    double predict(const std::vector<double>& input) const
    {
        if (m_Net.get() == nullptr)
        {
            throw std::domain_error("Use fit before predict.");
        }

        return m_Net->propagateForward(input)[0];
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
    bool m_UseSeed;
    unsigned int m_Seed;
    double m_OptimizationTolerance;
    double m_Momentum;
    size_t m_IterNoChangeN;

    std::unique_ptr<NeuralNetwork> m_Net;
};

}

#endif // YANNL_MLP_H
