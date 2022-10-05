#ifndef YANNL_NEURON_LAYER_H
#define YANNL_NEURON_LAYER_H

#include "Neuron.h"
#include <sstream>  // std::ostringstream
#include <numeric>  // std::accumulate

#include <algorithm> // std::for_each in Code::Blocks

namespace YANNL
{
enum class LayerType
{
    Hidden = 0,
    Dropout,
    OutputClassification,
    OutputRegression
};

class NeuronLayer
{
public:
    virtual size_t size() const = 0;
    virtual LayerType type() const = 0;
    virtual void inspect(std::ostream& os, size_t& weightN) const = 0;
    virtual std::vector<double> propagateForward(const std::vector<double>& inputs) = 0;
    virtual size_t probableClass() const = 0;
    virtual double calcError(const std::vector<double>& expectedOutputs) const = 0;
    virtual void propagateBackwardOuputLayer(const std::vector<double>& expectedOutputs) = 0;
    virtual void propagateBackwardHiddenLayer(const NeuronLayer& nextLayer) = 0;
    virtual double sumDelta(size_t weightN) const = 0;
    virtual void updateWeights() = 0;
    virtual void saveToFile(std::ofstream& output) const = 0;
};

class DenseLayer : public NeuronLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit DenseLayer(size_t neuronsN, size_t prevLayerNeuronsN, ActivationFunctions afunc,
        double learningRate, double momentum, double bias = 0.0) :
        m_AFunc(afunc), m_LearningRate(learningRate), m_Momentum(momentum)
    {
        for (size_t i = 0; i < neuronsN; i++)
        {
            m_Neurons.push_back(Neuron(prevLayerNeuronsN, afunc, learningRate, momentum, bias));
        }
    }

    explicit DenseLayer(const std::vector<std::vector<double>>& layerWeights, ActivationFunctions afunc,
        double learningRate, double momentum, double bias = 0.0) :
        m_AFunc(afunc), m_LearningRate(learningRate), m_Momentum(momentum)
    {
        std::for_each(layerWeights.cbegin(), layerWeights.cend(),
            [&](const std::vector<double>& neuronWeigths)
            {
                m_Neurons.push_back(Neuron(neuronWeigths, afunc, learningRate, momentum, bias));
            });
    }

    // No need to apply the rule of five as the class contains no raw pointers

    size_t size() const override
    {
        return m_Neurons.size();
    }

    void inspect(std::ostream& os, size_t& weightN) const override
    {
        os << "Neurons: " << m_Neurons.size() << " activation: "
            << ActivationFunction::build(m_AFunc)->name() << "\n";

        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            os << " Neuron " << (n + 1) << "\n";
            m_Neurons[n].inspect(os, weightN);
        }
    }

    //! Propagates the input forward and calculates the outputs. To be specialized
    //! for output classification layer as the outputs are dependent of all the inputs.
    //! @param inputs Vector of inputs.
    //! @returns Vector of outputs.
    std::vector<double> propagateForward(const std::vector<double>& inputs) override
    {
        std::vector<double> outputs;

        std::for_each(m_Neurons.begin(), m_Neurons.end(),
            [&](Neuron& neuron)
            {
                // If some inputs are set to 0 due to a previous dropout layer
                // they will be set as 0 in the neuron and thus deactivated
                // during the back propagation weight update as if i == 0
                // in the following equations, the gradient is 0.
                // dn/dw = i
                // Gradient = delta * dn/dw = delta * i
                outputs.push_back(neuron.calcOutput(inputs));
            });

        return outputs;
    }

    size_t probableClass() const override
    {
        return std::distance(m_Neurons.cbegin(),
            std::max_element(m_Neurons.cbegin(), m_Neurons.cend(),
                [](const Neuron& n1, const Neuron& n2)
                {
                    return n1.output() < n2.output();
                }));
    }

    //! To be specialized with mean squared error for output regression layers
    //! and cross entropy error for output classification layers. Nothing recommended
    //! for hidden layers as it will not be used; can be mean squared error.
    // virtual double calcError(const std::vector<double>& expectedOutputs) const = 0;

    //! Propagates the expected output backward to first calculate the delta on each neuron
    //! of each layer, and second to update the weights. To be specialized for output
    //! classifcation layers.
    //! @param expectedOutputs Vector of expected outputs to be compared to the internal
    //!   vector of actual outputs.
    void propagateBackwardOuputLayer(const std::vector<double>& expectedOutputs) override
    {
        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            m_Neurons[n].propagateBackwardOutputLayer(expectedOutputs[n]);
        }
    }

    void propagateBackwardHiddenLayer(const NeuronLayer& nextLayer) override
    {
        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            // dE/do = Sum(deltaOutputNeurons * w)
            double sum = nextLayer.sumDelta(n);

            m_Neurons[n].propagateBackwardHiddenLayer(sum);
        }
    }

    double sumDelta(size_t weightN) const override
    {
        double sum = 0.0;

        for (size_t i = 0; i < m_Neurons.size(); i++)
        {
            // dE/do = Sum(deltaOutputNeurons * w)
            sum += m_Neurons[i].delta() * m_Neurons[i].weight(weightN);
        }

        return sum;
    }

    void updateWeights() override
    {
        for (Neuron& neuron : m_Neurons)
        {
            neuron.updateWeights();
        }
    }

    void saveToFile(std::ofstream& output, LayerType layerType) const
    {
        output << static_cast<int>(layerType) << " " << static_cast<int>(m_AFunc) << " ";

        if (m_Neurons.size() > 0)
        {
            output << m_Neurons[0].inputSize() << " ";
        }
        else
        {
            output << 0 << " ";
        }

        output << m_Neurons.size() << " " << std::endl;

        for (const Neuron& neuron : m_Neurons)
        {
            neuron.saveToFile(output);
        }
    }

protected:
    const ActivationFunctions m_AFunc;
    const double m_LearningRate = 0.0;
    const double m_Momentum = 0.0;
    std::vector<Neuron> m_Neurons;
};

class HiddenLayer : public DenseLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit HiddenLayer(size_t neuronsN, size_t prevLayerNeuronsN, ActivationFunctions afunc,
        double learningRate, double momentum, double bias = 0.0) :
        DenseLayer(neuronsN, prevLayerNeuronsN, afunc, learningRate, momentum, bias)
    {

    }

    explicit HiddenLayer(const std::vector<std::vector<double>>& layerWeights, ActivationFunctions afunc,
        double learningRate, double momentum, double bias = 0.0) :
        DenseLayer(layerWeights, afunc, learningRate, momentum, bias)
    {

    }

    // No need to apply the rule of five as the class contains no raw pointers

    LayerType type() const override
    {
        return LayerType::Hidden;
    }

    //! Calculates the mean squared error as the default loss function for hidden layers.
    //! Should be specialized in subsequent child classes, especially for classification
    //! layers where cross entropy error should be used.
    //! @throws std::domain_error   If the number of expected outputs is different
    //!                             from the number of neurons on the layer.
    double calcError(const std::vector<double>& expectedOutputs) const override
    {
        throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
            << "[Calculate error] Output layer cannot be a hidden one. Check that last "
            << "layer is either an output classification layer or regression layer.").str()
        );

        return 0.0;
    }

    void saveToFile(std::ofstream& output) const override
    {
        DenseLayer::saveToFile(output, LayerType::Hidden);
    }
};

class DropoutLayer : public NeuronLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit DropoutLayer(double rate, size_t size) :
        m_Neurons(size), m_DropoutRate(rate), m_SumDeltaNextLayer(size),
        m_Generator(m_Rng()), m_Dist(0.0, 1.0)

    {

    }

    // No need to apply the rule of five as the class contains no raw pointers

    size_t size() const override
    {
        return m_Neurons.size();
    }

    LayerType type() const override
    {
        return LayerType::Dropout;
    }

    void inspect(std::ostream& os, size_t& weightN) const override
    {
        os << "Neurons: " << m_Neurons.size() << "\n"
            << "Dropout layer of rate " << m_DropoutRate << "\n";
    }

    std::vector<double> propagateForward(const std::vector<double>& inputs) override
    {
        std::vector<double> outputs;

        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            if (m_Dist(m_Generator) >= m_DropoutRate)
            {
                // Keep neuron from previous layer and rescale output
                m_Neurons[n] = true;
                outputs.push_back(inputs[n] / (1 - m_DropoutRate));
            }
            else
            {
                // Deactivate neuron from previous layer
                m_Neurons[n] = false;
                outputs.push_back(0.0);
            }
        }

        return outputs;
    }

    size_t probableClass() const override
    {
        throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
            << "[Probable class] Output layer cannot be a dropout one.").str()
        );

        return 0;
    }

    double calcError(const std::vector<double>& expectedOutputs) const override
    {
        throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
            << "[Calculate error] Output layer cannot be a dropout one.").str()
        );

        return 0.0;
    }

    void propagateBackwardOuputLayer(const std::vector<double>& expectedOutputs) override
    {
        throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
            << "[Propagate backward] Output layer cannot be a dropout one.").str()
        );
    }

    void propagateBackwardHiddenLayer(const NeuronLayer& nextLayer) override
    {
        // Nothing to propagate backward.
        // Just calculate the sum of next layer delta

        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            m_SumDeltaNextLayer[n] = nextLayer.sumDelta(n);
        }
    }

    double sumDelta(size_t weightN) const override
    {
        return m_SumDeltaNextLayer[weightN];
    }

    void updateWeights() override
    {

    }

    void saveToFile(std::ofstream& output) const override
    {
        output << static_cast<int>(LayerType::Dropout) << " " << m_Neurons.size()
            << " " << m_DropoutRate << " " << std::endl;
    }

private:
    std::vector<bool> m_Neurons;
    double m_DropoutRate = 0.0;
    std::vector<double> m_SumDeltaNextLayer;

    std::random_device m_Rng;
    std::mt19937 m_Generator;
    std::uniform_real_distribution<double> m_Dist;
};

class OutputClassificationLayer : public DenseLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit OutputClassificationLayer(size_t neuronsN, size_t prevLayerNeuronsN,
        double learningRate, double momentum, double bias = 0.0) :
        DenseLayer(neuronsN, prevLayerNeuronsN, ActivationFunctions::Identity, learningRate,
            momentum, bias), m_Outputs(neuronsN)
    {

    }

    explicit OutputClassificationLayer(const std::vector<std::vector<double>>& layerWeights,
        double learningRate, double momentum, double bias = 0.0) :
        DenseLayer(layerWeights, ActivationFunctions::Identity, learningRate,
            momentum, bias), m_Outputs(layerWeights.size())
    {

    }

    // No need to apply the rule of five as the class contains no raw pointers

    size_t size() const override
    {
        return m_Outputs.size();
    }

    LayerType type() const override
    {
        return LayerType::OutputClassification;
    }

    std::vector<double> propagateForward(const std::vector<double>& inputs) override
    {
        m_Outputs.clear();

        std::for_each(m_Neurons.begin(), m_Neurons.end(),
            [&](Neuron& neuron)
            {
                m_Outputs.push_back(neuron.calcOutput(inputs));
            });

        double sumExp = std::accumulate(m_Outputs.cbegin(), m_Outputs.cend(), 0.0,
            [](double a, double b)
            {
                return a + std::exp(b);
            });

        std::for_each(m_Outputs.begin(), m_Outputs.end(),
            [&](double& output)
            {
                output = std::exp(output) / sumExp;
            });

        return m_Outputs;
    }

    //! Calculates the cross entropy error as this is a classification layer.
    //! @throws std::domain_error If the number of expected outputs is different
    //!   from the number of neurons on the layer.
    double calcError(const std::vector<double>& expectedOutputs) const override
    {
        if (expectedOutputs.size() != m_Outputs.size())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Classification layer/Cross entropy error] Expected output size is inconsistent "
                << "with layer size :  expected " << m_Outputs.size() << " provided "
                << expectedOutputs.size() << ".").str()
            );
        }

        double total_error = 0.0;

        for (size_t n = 0; n < m_Outputs.size(); n++)
        {
            total_error += -expectedOutputs[n] * std::log(m_Outputs[n]);
        }

        return total_error;
    }

    void propagateBackwardOuputLayer(const std::vector<double>& expectedOutputs) override
    {
        double sumExpectedOuputs = std::accumulate(expectedOutputs.cbegin(), expectedOutputs.cend(), 0.0);

        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            // -[expectedOutputs[n] - m_Outputs[n] * Sum(expectedOutputs)]
            // Equivalent to m_Outputs[n] - expectedOutputs[n] = out - target when Sum = 1.
            m_Neurons[n].propagateBackwardClassificationLayer(-(expectedOutputs[n]
                - m_Outputs[n] * sumExpectedOuputs));
        }
    }

    void saveToFile(std::ofstream& output) const override
    {
        DenseLayer::saveToFile(output, LayerType::OutputClassification);
    }

private:
    std::vector<double> m_Outputs;
};

class OutputRegressionLayer : public DenseLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit OutputRegressionLayer(size_t neuronsN, size_t prevLayerNeuronsN,
        ActivationFunctions afunc, double learningRate, double momentum, double bias = 0.0) :
        DenseLayer(neuronsN, prevLayerNeuronsN, afunc, learningRate, momentum, bias)
    {

    }

    explicit OutputRegressionLayer(const std::vector<std::vector<double>>& layerWeights,
        ActivationFunctions afunc, double learningRate, double momentum, double bias = 0.0) :
        DenseLayer(layerWeights, afunc, learningRate, momentum, bias)
    {

    }

    // No need to apply the rule of five as the class contains no raw pointers

    LayerType type() const override
    {
        return LayerType::OutputRegression;
    }

    //! Calculates the mean squared error as this is a regression layer.
    //! @throws std::domain_error If the number of expected outputs is different
    //!   from the number of neurons on the layer.
    double calcError(const std::vector<double>& expectedOutputs) const override
    {
        if (expectedOutputs.size() != m_Neurons.size())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Squared error/Layer] Expected output size is inconsistent with layer size: "
                << "expected " << m_Neurons.size() << " provided " << expectedOutputs.size() << ".").str()
            );
        }

        double total_error = 0.0;

        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            total_error += m_Neurons[n].squaredError(expectedOutputs[n]);
        }

        return total_error;
    }

    void saveToFile(std::ofstream& output) const override
    {
        DenseLayer::saveToFile(output, LayerType::OutputRegression);
    }
};

}

#endif // YANNL_NEURON_LAYER_H
