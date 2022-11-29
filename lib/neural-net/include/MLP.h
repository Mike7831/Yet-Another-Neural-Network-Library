//! Yet Another Neural Network Library C++ (YANNL-C++)
//! @copyright  Copyright(c) 2022 - Mickael Deloison
//! @license    https://opensource.org/licenses/GPL-3.0 GPL-3.0

#ifndef YANNL_MLP_H
#define YANNL_MLP_H

#include "NeuralNetwork.h"
#include <chrono>   // std::chrono

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

enum class MLPType
{
    Regressor = 0,
    Classifier
};

typedef uint8_t t_Labels;

template<typename T>
class MLP
{
public:
    void fit(const std::vector<std::vector<double>>& inputs,
        const std::vector<T>& expectedOuputs)
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

            t_Labels max = 0; // max label if MLPClassifier
            t_Labels min = 0; // min label if MLPClassifier

            if (type() == MLPType::Classifier)
            {
                max = (t_Labels) *std::max_element(expectedOuputs.begin(), expectedOuputs.end());
                min = (t_Labels) *std::min_element(expectedOuputs.begin(), expectedOuputs.end());
                t_Labels size = max - min + 1; // min included to max included = max - min + 1

                log("Adds output classification layer of size " + std::to_string(size)
                    + " for range " + std::to_string(min) + "-" + std::to_string(max) + ".");
                m_Net->addOutputClassificationLayer(size);
            }
            else
            {
                log("Adds output regression layer of size 1.");
                m_Net->addOutputRegressionLayer(1, m_AFunc);
            }

            log("Trains the network with max " + std::to_string(m_MaxIterations) + " epochs.");

            // Train neural network

            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<double> errors;
            double error = 0.0;
            size_t nbBatches = 1, batchSize = m_BatchSize;

            // If the MLP should not use the batch size it means it is an on-line stochastic
            // gradient descent with batches of size 1.
            if (!m_UseBatchSize)
            {
                batchSize = 1;
                nbBatches = inputs.size();
            }
            else
            {
                nbBatches = inputs.size() / batchSize + (inputs.size() % batchSize == 0 ? 0 : 1);
            }

            for (size_t epoch = 0; epoch < m_MaxIterations; epoch++)
            {
                error = 0.0;

                for (size_t batch = 0; batch < nbBatches; batch++)
                {
                    for (size_t i = batch * batchSize; i < (batch + 1) * batchSize && i < inputs.size(); i++)
                    {
                        m_Net->propagateForward(inputs[i]);

                        if (type() == MLPType::Classifier)
                        {
                            std::vector<double> expectedOutput = Utils::convertLabelToVect((t_Labels)expectedOuputs[i], min, max);
                            error += m_Net->calcError(expectedOutput);
                            m_Net->propagateBackward(expectedOutput);
                        }
                        else
                        {
                            error += m_Net->calcError(expectedOuputs[i]);
                            m_Net->propagateBackward(expectedOuputs[i]);
                        }
                    }

                    m_Net->updateWeights();
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
                        errors.erase(errors.begin());
                        errors.push_back(error);

                        // Only in the case of early stopping and not in the case
                        // of adaptive learning rate, determine whether to stop.
                        if (m_LearningRateType != LearningRate::Adaptive)
                        {
                            bool earlyStop = true;

                            for (size_t i = errors.size() - 1; i > 0; i--)
                            {
                                if (errors[i - 1] - errors[i] > m_OptimizationTolerance)
                                {
                                    earlyStop = false;
                                }
                            }

                            if (earlyStop)
                            {
                                log("Optimization tolerance of " + std::to_string(m_OptimizationTolerance)
                                    + " reached after " + std::to_string(epoch) + " epochs. Stopping.");
                                break;
                            }
                        }
                    }

                    // In case of adaptive learning rate and with sufficient
                    // error history (current + 2 = 3)
                    if (m_LearningRateType == LearningRate::Adaptive
                        && errors.size() >= 3)
                    {
                        size_t i = errors.size() - 1;

                        // Each time two consecutive epochs fail to
                        // decrease training loss by at least tol,
                        // the current learning rate is divided by 5.
                        if (std::fabs(errors[i - 1] - errors[i]) < m_OptimizationTolerance
                            && std::fabs(errors[i - 2] - errors[i - 1]) < m_OptimizationTolerance)
                        {
                            m_EffectiveLearningRate /= 5;
                            m_Net->updateLearningRate(m_EffectiveLearningRate);
                        }
                    }
                }

                if (m_LearningRateType == LearningRate::InvScaling)
                {
                    m_EffectiveLearningRate = m_LearningRate / std::pow(epoch + 1, m_PowerT);
                    m_Net->updateLearningRate(m_EffectiveLearningRate);
                }
            }

            log("Final error is " + std::to_string(error) + ".");

            if (m_LearningRateType == LearningRate::Adaptive)
            {
                log("Final effective learning rate is " + std::to_string(m_EffectiveLearningRate) + ".");
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            log("Training completed in " + std::to_string(elapsed) + " ms.");
        }
        else
        {
            log("Input is empty. No training possible.");
        }
    }

    void inspect(std::ostream& os) const
    {
        if (m_Net.get() != nullptr)
        {
            m_Net->inspect(os);
        }
    }

    virtual MLPType type() const = 0;

protected:
    explicit MLP(
        const std::vector<size_t>& hidden_layer_sizes,
        ActivationFunctions activation,
        Solvers solver,
        bool use_batch_size,
        size_t batch_size,
        LearningRate learning_rate,
        double learning_rate_init,
        double power_t,
        size_t max_iter,
        bool use_random_state,
        unsigned int random_state,
        double tol,
        bool verbose,
        double momentum,
        bool early_stopping,
        size_t n_iter_no_change) :
        m_HiddenLayerSizes(hidden_layer_sizes),
        m_AFunc(activation),
        m_Solver(solver),
        m_UseBatchSize(use_batch_size),
        m_BatchSize(batch_size),
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

    std::unique_ptr<NeuralNetwork> m_Net;

    void log(const std::string& msg) const
    {
        if (m_Verbose)
        {
            std::cerr << msg << std::endl;
        }
    }

private:
    const std::vector<size_t> m_HiddenLayerSizes;
    const ActivationFunctions m_AFunc;
    const Solvers m_Solver;
    const bool m_UseBatchSize;
    const size_t m_BatchSize;
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

    double m_EffectiveLearningRate;
};

class MLPRegressor : public MLP<double>
{
public:
    explicit MLPRegressor(
        const std::vector<size_t>& hidden_layer_sizes = { 100 },
        ActivationFunctions activation = ActivationFunctions::ReLU,
        Solvers solver = Solvers::SGD,
        bool use_batch_size = false,
        size_t batch_size = 0,
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
        MLP(hidden_layer_sizes,
            activation,
            solver,
            use_batch_size,
            batch_size,
            learning_rate,
            learning_rate_init,
            power_t,
            max_iter,
            use_random_state,
            random_state,
            tol,
            verbose,
            momentum,
            early_stopping,
            n_iter_no_change)
    {

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

    MLPType type() const override
    {
        return MLPType::Regressor;
    }
};

class MLPClassifer : public MLP<t_Labels>
{
public:
    explicit MLPClassifer(
        const std::vector<size_t>& hidden_layer_sizes = { 100 },
        ActivationFunctions activation = ActivationFunctions::ReLU,
        Solvers solver = Solvers::SGD,
        bool use_batch_size = false,
        size_t batch_size = 0,
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
        MLP(hidden_layer_sizes,
            activation,
            solver,
            use_batch_size,
            batch_size,
            learning_rate,
            learning_rate_init,
            power_t,
            max_iter,
            use_random_state,
            random_state,
            tol,
            verbose,
            momentum,
            early_stopping,
            n_iter_no_change)
    {

    }

    size_t predict(const std::vector<double>& input) const
    {
        if (m_Net.get() == nullptr)
        {
            throw std::domain_error("Use fit before predict.");
        }

        m_Net->propagateForward(input);
        return m_Net->probableClass();
    }

    MLPType type() const override
    {
        return MLPType::Classifier;
    }
};

}

#endif // YANNL_MLP_H
