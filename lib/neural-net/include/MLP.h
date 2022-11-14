//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#ifndef YANNL_MLP_H
#define YANNL_MLP_H

#include "NeuralNetwork.h"
#include <chrono>   // std::chrono
#include <list>    // std::list

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
        bool verbose = false,
        double momentum = 0.9,
        bool early_stopping = false,
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
        m_Verbose(verbose),
        m_Momentum(momentum),
        m_EarlyStopping(early_stopping),
        m_IterNoChangeN(n_iter_no_change),
        m_EffectiveLearningRate(learning_rate_init)
    {

    }

    void fit(const std::vector<std::vector<double>>& inputs,
        const std::vector<double>& expectedOuputs)
    {
        log("Checks whether input is empty.");

        if (!inputs.empty())
        {
            log("Checks that output size is consistent with input size.");

            // Check that output size is consistent with input size
            if (inputs.size() != expectedOuputs.size())
            {
                throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                    << "Input and output size are not consistent: input "
                    << inputs.size() << " output " << expectedOuputs.size() << ".").str()
                );
            }

            log("Checks that all inputs are of same size.");

            // Check that all inputs are of the same

            size_t inputSize = inputs[0].size();

            for (size_t i = 1; i < inputs.size(); i++)
            {
                if (inputs[i].size() != inputSize)
                {
                    throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                        << "All inputs do not have the same size: first "
                        << inputSize << " " << i << "th " << inputs[i].size() << ".").str()
                    );
                }
            }

            log("Output and input are of same size. All inputs are of same size.");

            log("Builds the neural network of input size " + std::to_string(inputSize) + ".");

            // Build neural network

            m_Net = std::make_unique<NeuralNetwork>(inputSize, m_LearningRate, m_Momentum, m_UseSeed, m_Seed);

            std::for_each(m_HiddenLayerSizes.cbegin(), m_HiddenLayerSizes.cend(),
                [&](size_t layerSize)
                {
                    log("Adds hidden layer of size " + std::to_string(layerSize) + ".");
                    m_Net->addHiddenLayer(layerSize, m_AFunc);
                });

            log("Adds output regression layer of size 1.");
            m_Net->addOutputRegressionLayer(1, m_AFunc);

            log("Trains the network with max " + std::to_string(m_MaxIterations) + " epochs.");

            // Train neural network

            auto t0 = std::chrono::high_resolution_clock::now();
            std::list<double> errors;
            double error = 0.0;

            for (size_t n = 0; n < m_MaxIterations; n++)
            {
                error = 0.0;

                for (size_t i = 0; i < inputs.size(); i++)
                {
                    m_Net->propagateForward(inputs[i]);
                    error += m_Net->calcError(expectedOuputs[i]);
                    m_Net->propagateBackward(expectedOuputs[i]);
                }

                error /= inputs.size();

                if (m_EarlyStopping || m_LearningRateType == LearningRate::Adaptive)
                {
                    // In case of early stopping or adaptive learning rate
                    // we need to keep track of the last errors.

                    if (errors.size() < m_IterNoChangeN + 1)
                    {
                        // +1 because the (N+1)th element is used as a reference
                        // and compared to the Nth other elements.
                        // No decision while the list is not complete, i.e there are
                        // not enough items to conclude that the training can stop.
                        errors.push_back(error);
                    }
                    else
                    {
                        errors.pop_front();
                        errors.push_back(error);

                        // Only in the case of early stopping and not in the case
                        // of adaptive learning rate, determine whether to stop.
                        if (m_LearningRateType != LearningRate::Adaptive)
                        {
                            bool earlyStop = true;

                            std::for_each(errors.cbegin(), errors.cend(),
                                [&](const double& e)
                                {
                                    if (errors.front() - e > m_OptimizationTolerance)
                                    {
                                        earlyStop = false;
                                    }
                                });

                            if (earlyStop)
                            {
                                log("Optimization tolerance of " + std::to_string(m_OptimizationTolerance)
                                    + " reached after " + std::to_string(n) + " iterations. Stopping.");
                                break;
                            }
                        }
                    }

                    // In case of adaptive learning rate and with sufficient
                    // error history (current + 2 = 3)
                    if (m_LearningRateType == LearningRate::Adaptive
                        && errors.size() >= 3)
                    {
                        auto it = errors.crbegin();
                        double error2 = *it; std::advance(it, 1);
                        double error1 = *it; std::advance(it, 1);
                        double error0 = *it;

                        // Each time two consecutive epochs fail to
                        // decrease training loss by at least tol,
                        // the current learning rate is divided by 5.
                        if (!(error0 - error1 > m_OptimizationTolerance
                            && error1 - error2 > m_OptimizationTolerance))
                        {
                            m_EffectiveLearningRate /= 5;
                            m_Net->updateLearningRate(m_EffectiveLearningRate);
                        }
                    }
                }

                if (m_LearningRateType == LearningRate::InvScaling)
                {
                    m_EffectiveLearningRate = m_LearningRate / std::pow(n + 1, m_PowerT);
                    m_Net->updateLearningRate(m_EffectiveLearningRate);
                }
            }

            log("Final error is " + std::to_string(error) + ".");

            auto t1 = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            log("Training completed in " + std::to_string(elapsed) + " ms.");
        }
        else
        {
            log("Input is empty. No training possible.");
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

    void log(const std::string& msg) const
    {
        if (m_Verbose)
        {
            std::cerr << msg << std::endl;
        }
    }

    void inspect(std::ostream& os) const
    {

    }

private:
    const std::vector<size_t> m_HiddenLayerSizes;
    const ActivationFunctions m_AFunc;
    const Solvers m_Solver;
    const LearningRate m_LearningRateType;
    const double m_LearningRate;
    const double m_PowerT;
    const size_t m_MaxIterations;
    const bool m_UseSeed;
    const unsigned int m_Seed;
    const double m_OptimizationTolerance;
    const bool m_Verbose;
    const double m_Momentum;
    const bool m_EarlyStopping;
    const size_t m_IterNoChangeN;

    std::unique_ptr<NeuralNetwork> m_Net;

    double m_EffectiveLearningRate;
};

}

#endif // YANNL_MLP_H
