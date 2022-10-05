#ifndef YANNL_NEURON_H
#define YANNL_NEURON_H

#include "ActivationFunction.h"
#include <iostream> // std::ostream
#include <fstream>  // std::ofstream
#include <vector>   // std::vector
#include <random>   // std::random_device

namespace YANNL
{

class Neuron
{
public:
    explicit Neuron(size_t weightsN, ActivationFunctions afunc,
        double learningRate, double momentum, double bias = 0.0) :
        m_AFunc(ActivationFunction::build(afunc)), m_LearningRate(learningRate),
        m_Momentum(momentum), m_Bias(bias),
        m_WeightsPrevChange(weightsN), m_BiasPrevChange(0.0)
    {
        std::random_device rng;
        std::mt19937 generator(rng());
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        for (size_t n = 0; n < weightsN; n++)
        {
            m_Weights.push_back(dist(generator));
        }
    }

    explicit Neuron(const std::vector<double>& weights, ActivationFunctions afunc,
        double learningRate, double momentum, double bias = 0.0) :
        m_AFunc(ActivationFunction::build(afunc)), m_LearningRate(learningRate),
        m_Momentum(momentum), m_Bias(bias), m_Weights(weights),
        m_WeightsPrevChange(weights.size()), m_BiasPrevChange(0.0)
    {

    }

    // No need to apply the rule of five as the class contains no raw pointers

    void inspect(std::ostream& os, size_t& weightN) const
    {
        for (const auto& neuronWeight : m_Weights)
        {
            os << "  w" << weightN << ": " << neuronWeight << "\n";
            ++weightN;
        }

        os << "  Bias: " << m_Bias << std::endl;
    }

    double calcOutput(const std::vector<double>& inputs)
    {
        m_Inputs = inputs;
        double total = 0.0;

        for (size_t n = 0; n < inputs.size(); n++)
        {
            // /!\ If more inputs than weights then exception.
            // Should not occur as the number of weights is verified if weights are
            // provided manually, and dimensioned accordingly if not provided.
            total += inputs[n] * m_Weights[n];
        }

        total += m_Bias;

        m_Output = m_AFunc->calc(total);

        return m_Output;
    }

    double output() const
    {
        return m_Output;
    }

    double squaredError(double target) const
    {
        return std::pow(target - m_Output, 2);
    }

    void propagateBackwardOutputLayer(double target)
    {
        // dE/dw = dE/do * do/dn * dn/dw = Gradient
        // dE/do = -(t - o)
        // do/dn = f'(o)
        // dn/dw = i
        // dE/dw = [ -(t - o) * f'(o) ] * i = delta * i
        m_Delta = -(target - m_Output) * m_AFunc->calcDerivate(m_Output);
    }

    void propagateBackwardClassificationLayer(double delta)
    {
        // For a classification layer the delta is caculated at layer level
        // because serveral neuron values are necessary to calculate it.
        m_Delta = delta;
    }

    void propagateBackwardHiddenLayer(double sumWeightedDeltaNextLayer)
    {
        // dE/dw = dE/do * do/dn * dn/dw = Gradient
        // dE/do = Sum(deltaOutputNeurons * w)
        // do/dn = f'(oh)
        m_Delta = sumWeightedDeltaNextLayer * m_AFunc->calcDerivate(m_Output);
    }

    void updateWeights()
    {
        double change = 0.0;

        for (size_t n = 0; n < m_Inputs.size(); n++)
        {
            // dn/dw = i
            // Gradient = delta * dn/dw = delta * i
            double gradient = m_Delta * m_Inputs[n];

            // https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
            // m_PrevChange[] initialized to 0.0
            change = m_LearningRate * gradient + m_Momentum * m_WeightsPrevChange[n];
            m_Weights[n] -= change;
            m_WeightsPrevChange[n] = change;
        }

        change = m_LearningRate * (m_Delta * 1.0) + m_Momentum * m_BiasPrevChange;
        m_Bias -= change;
        m_BiasPrevChange = change;
    }

    double delta() const
    {
        return m_Delta;
    }

    //! Returns the nth weight of the neuron.
    //! @throws std::exception  If index is out of range
    double weight(size_t n) const
    {
        return m_Weights[n];
    }

    size_t inputSize() const
    {
        return m_Weights.size();
    }

    void saveToFile(std::ofstream& output) const
    {
        for (const auto& neuronWeight : m_Weights)
        {
            output << neuronWeight << " ";
        }

        output << std::endl;
    }

private:
    std::shared_ptr<ActivationFunction> m_AFunc;
    const double m_LearningRate = 0.0;
    const double m_Momentum = 0.0;
    double m_Bias = 0.0;
    std::vector<double> m_Weights;
    std::vector<double> m_WeightsPrevChange;
    double m_BiasPrevChange = 0.0;

    double m_Output = 0.0;
    std::vector<double> m_Inputs;
    double m_Delta = 0.0;
};

}

#endif // YANNL_NEURON_H