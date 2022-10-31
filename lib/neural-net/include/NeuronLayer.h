#ifndef YANNL_NEURON_LAYER_H
#define YANNL_NEURON_LAYER_H

#include "Neuron.h"
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
    virtual void updateLearningRate(double learningRate) = 0;
    virtual std::vector<double> propagateForward(const std::vector<double>& inputs, bool ignoreDropout) = 0;
    virtual size_t probableClass() const = 0;
    virtual double calcError(const std::vector<double>& expectedOutputs) const = 0;
    virtual void propagateBackwardOuputLayer(const std::vector<double>& expectedOutputs) = 0;
    virtual void propagateBackwardHiddenLayer(const NeuronLayer& nextLayer) = 0;
    virtual double sumDelta(size_t weightN) const = 0;
    virtual bool droppedNeuron(size_t neuronN) const = 0;
    virtual bool dropoutLayer() const = 0;
    virtual double dropoutRate() const = 0;
    virtual void updateWeights() = 0;
    virtual void saveToFile(std::ofstream& output) const = 0;
};

class DenseLayer : public NeuronLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit DenseLayer(size_t neuronsN, size_t prevLayerNeuronsN,
        ActivationFunctions afunc, double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        m_AFunc(afunc), m_LearningRate(learningRate), m_Momentum(momentum)
    {
        for (size_t i = 0; i < neuronsN; i++)
        {
            m_Neurons.push_back(Neuron(prevLayerNeuronsN, afunc, learningRate,
                momentum, seedGen, bias));
        }
    }

    explicit DenseLayer(const std::vector<std::vector<double>>& layerWeights,
        ActivationFunctions afunc,  double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        m_AFunc(afunc), m_LearningRate(learningRate), m_Momentum(momentum)
    {
        std::for_each(layerWeights.cbegin(), layerWeights.cend(),
            [&](const std::vector<double>& neuronWeigths)
            {
                m_Neurons.push_back(Neuron(neuronWeigths, afunc, learningRate,
                    momentum, bias));
            });
    }

    explicit DenseLayer(const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias, ActivationFunctions afunc,
        double learningRate, double momentum, const std::shared_ptr<SeedGenerator>& seedGen) :
        m_AFunc(afunc), m_LearningRate(learningRate), m_Momentum(momentum)
    {
        for (size_t n = 0; n < layerWeights.size(); n++)
        {
            m_Neurons.push_back(Neuron(layerWeights[n], afunc, learningRate,
                momentum, layerBias[n]));
        }
    }

    // No need to apply the rule of five as the class contains no raw pointers

    size_t size() const override
    {
        return m_Neurons.size();
    }

    void inspect(std::ostream& os, size_t& weightN) const override
    {
        os << "Neurons: " << m_Neurons.size() << " activation: "
            << ActivationFunctionFactory::build(m_AFunc)->name() << "\n";

        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            os << " Neuron " << (n + 1) << "\n";
            m_Neurons[n].inspect(os, weightN);
        }
    }

    void updateLearningRate(double learningRate) override
    {
        m_LearningRate = learningRate;

        std::for_each(m_Neurons.begin(), m_Neurons.end(),
            [&](Neuron& neuron)
            {
                neuron.updateLearningRate(learningRate);
            });
    }

    //! Propagates the input forward and calculates the outputs. To be specialized
    //! for output classification layer as the outputs are dependent of all the inputs.
    //! @param inputs Vector of inputs.
    //! @param ignoreDropout Tells to ignore dropout during testing or validation.
    //! @returns Vector of outputs.
    std::vector<double> propagateForward(const std::vector<double>& inputs, bool ignoreDropout) override
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
                outputs.push_back(neuron.propagateForward(inputs));
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
    //! classification layers.
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

            m_Neurons[n].propagateBackwardHiddenLayer(sum,
                nextLayer.dropoutLayer(),
                nextLayer.dropoutRate(),
                nextLayer.droppedNeuron(n));
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

    bool droppedNeuron(size_t neuronN) const override
    {
        // Neurons are dropped only on a dropout layer.
        // neuronN has no importance; no need to verify its range.

        return false;
    }

    bool dropoutLayer() const override
    {
        return false;
    }

    double dropoutRate() const override
    {
        // Not a dropout layer; so dropout rate is always 0.0, i.e. keep neuron.
        return 0.0;
    }

    void updateWeights() override
    {
        for (Neuron& neuron : m_Neurons)
        {
            neuron.updateWeights();
        }
    }

    void saveToFile(std::ofstream& output, LayerType layerType,
        const std::vector<double>* outputs = nullptr) const
    {
        output << "LayerType: " << static_cast<int>(layerType) << "\n"
            << "[LayerBegin] \n"
            << "ActivationFunction: " << static_cast<int>(m_AFunc) << "\n"
            << "Momentum: " << m_Momentum << "\n"
            << "LearningRate: " << m_LearningRate << "\n"
            << "InputSize: ";

        if (m_Neurons.size() > 0)
        {
            output << m_Neurons[0].inputSize() << "\n";
        }
        else
        {
            output << 0 << "\n";
        }

        output << "OutputSize: " << m_Neurons.size() << " " << "\n";

        if (layerType == LayerType::OutputClassification && outputs != nullptr)
        {
            output << "OutputClassification: ";

            std::for_each(outputs->cbegin(), outputs->cend(),
                [&](const double& o)
                {
                    output << o << " ";
                });

            output << "\n";
        }

        for (const Neuron& neuron : m_Neurons)
        {
            neuron.saveToFile(output);
        }

        output << "[LayerEnd] \n\n";
    }

protected:
    const ActivationFunctions m_AFunc;
    double m_LearningRate = 0.0;
    const double m_Momentum = 0.0;
    std::vector<Neuron> m_Neurons;

    explicit DenseLayer(ActivationFunctions afunc, double learningRate, double momentum) :
        m_AFunc(afunc), m_LearningRate(learningRate), m_Momentum(momentum)
    {

    }
};

class HiddenLayer : public DenseLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit HiddenLayer(size_t neuronsN, size_t prevLayerNeuronsN,
        ActivationFunctions afunc,  double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        DenseLayer(neuronsN, prevLayerNeuronsN, afunc, learningRate, momentum, seedGen, bias)
    {

    }

    explicit HiddenLayer(const std::vector<std::vector<double>>& layerWeights,
        ActivationFunctions afunc, double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        DenseLayer(layerWeights, afunc, learningRate, momentum, seedGen, bias)
    {

    }

    explicit HiddenLayer(const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias, ActivationFunctions afunc,
        double learningRate, double momentum, const std::shared_ptr<SeedGenerator>& seedGen) :
        DenseLayer(layerWeights, layerBias, afunc, learningRate, momentum, seedGen)
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

    static HiddenLayer readFromFile(std::ifstream& file)
    {
        std::string tag;

        Utils::checkTag(file, tag, "[LayerBegin]");

        int afunc = 0;
        double momentum = 0.0;
        double learningRate = 0.0;
        size_t inputN = 0;
        size_t outputN = 0;
        file >> tag >> afunc;
        file >> tag >> momentum;
        file >> tag >> learningRate;
        file >> tag >> inputN;
        file >> tag >> outputN;

        HiddenLayer layer(static_cast<ActivationFunctions>(afunc), learningRate, momentum);

        for (size_t n = 0; n < outputN; n++)
        {
            layer.m_Neurons.push_back(Neuron::readFromFile(file));
        }

        Utils::checkTag(file, tag, "[LayerEnd]");

        return layer;
    }

protected:
    explicit HiddenLayer(ActivationFunctions afunc, double learningRate, double momentum) :
        DenseLayer(afunc, learningRate, momentum)
    {

    }
};


class DropoutLayer : public NeuronLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit DropoutLayer(double rate, size_t size, const std::shared_ptr<SeedGenerator>& seedGen) :
        m_Neurons(size), m_DropoutRate(rate), m_SumDeltaNextLayer(size),
        m_Generator(seedGen->seed()), m_Dist(0.0, 1.0)

    {

    }

    explicit DropoutLayer(double rate, size_t size, const std::mt19937& generator) :
        m_Neurons(size), m_DropoutRate(rate), m_SumDeltaNextLayer(size),
        m_Generator(generator), m_Dist(0.0, 1.0)

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

    void updateLearningRate(double learningRate) override
    {

    }

    std::vector<double> propagateForward(const std::vector<double>& inputs, bool ignoreDropout) override
    {
        std::vector<double> outputs;

        for (size_t n = 0; n < m_Neurons.size(); n++)
        {
            if (ignoreDropout || m_Dist(m_Generator) >= m_DropoutRate)
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

    bool droppedNeuron(size_t neuronN) const override
    {
        return !m_Neurons[neuronN];
    }

    bool dropoutLayer() const override
    {
        return true;
    }

    double dropoutRate() const override
    {
        return m_DropoutRate;
    }

    void updateWeights() override
    {

    }

    void saveToFile(std::ofstream& output) const override
    {
        output << "LayerType: " << static_cast<int>(LayerType::Dropout) << "\n"
            << "[LayerBegin] \n"
            << "Size: " << m_Neurons.size() << "\n"
            << "DropoutRate: " << m_DropoutRate << "\n"
            << "Generator: " << m_Generator << "\n";

        output << "Activations: ";

        for (const auto& val : m_Neurons)
        {
            output << val << " ";
        }

        output << "\n"
            << "Deltas: ";

        for (const auto& delta : m_SumDeltaNextLayer)
        {
            output << delta << " ";
        }

        output << "\n"
            << "[LayerEnd] \n\n";
    }

    static DropoutLayer readFromFile(std::ifstream& file)
    {
        std::string tag;

        Utils::checkTag(file, tag, "[LayerBegin]");

        size_t sizeN = 0;
        double rate = 0.0;
        std::mt19937 generator;
        file >> tag >> sizeN;
        file >> tag >> rate;
        file >> tag >> generator;

        DropoutLayer layer(rate, sizeN, generator);

        Utils::checkTag(file, tag, "Activations:");
        int a;

        for (size_t n = 0; n < sizeN; n++)
        {
            file >> a;
            layer.m_Neurons[n] = a;
        }

        Utils::checkTag(file, tag, "Deltas:");

        for (size_t n = 0; n < sizeN; n++)
        {
            file >> layer.m_SumDeltaNextLayer[n];
        }

        Utils::checkTag(file, tag, "[LayerEnd]");

        return layer;
    }

private:
    std::vector<bool> m_Neurons;
    const double m_DropoutRate = 0.0;
    std::vector<double> m_SumDeltaNextLayer;

    std::mt19937 m_Generator;
    std::uniform_real_distribution<double> m_Dist;
};


class OutputClassificationLayer : public DenseLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit OutputClassificationLayer(size_t neuronsN, size_t prevLayerNeuronsN,
        double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        DenseLayer(neuronsN, prevLayerNeuronsN, ActivationFunctions::Identity, learningRate,
            momentum, seedGen, bias), m_Outputs(neuronsN)
    {

    }

    explicit OutputClassificationLayer(const std::vector<std::vector<double>>& layerWeights,
        double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        DenseLayer(layerWeights, ActivationFunctions::Identity, learningRate,
            momentum, seedGen, bias), m_Outputs(layerWeights.size())
    {

    }

    explicit OutputClassificationLayer(const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias, double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen) :
        DenseLayer(layerWeights, layerBias, ActivationFunctions::Identity, learningRate,
            momentum, seedGen), m_Outputs(layerWeights.size())
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

    std::vector<double> propagateForward(const std::vector<double>& inputs, bool ignoreDropout) override
    {
        m_Outputs.clear();

        std::for_each(m_Neurons.begin(), m_Neurons.end(),
            [&](Neuron& neuron)
            {
                m_Outputs.push_back(neuron.propagateForward(inputs));
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
        DenseLayer::saveToFile(output, LayerType::OutputClassification, &m_Outputs);
    }

    static OutputClassificationLayer readFromFile(std::ifstream& file)
    {
        std::string tag;

        Utils::checkTag(file, tag, "[LayerBegin]");

        int afunc = 0;
        double momentum = 0.0;
        double learningRate = 0.0;
        size_t inputN = 0;
        size_t outputN = 0;
        file >> tag >> afunc;
        file >> tag >> momentum;
        file >> tag >> learningRate;
        file >> tag >> inputN;
        file >> tag >> outputN;

        OutputClassificationLayer layer(outputN, learningRate, momentum);

        Utils::checkTag(file, tag, "OutputClassification:");

        for (size_t i = 0; i < outputN; i++)
        {
            file >> layer.m_Outputs[i];
        }

        for (size_t n = 0; n < outputN; n++)
        {
            layer.m_Neurons.push_back(Neuron::readFromFile(file));
        }

        Utils::checkTag(file, tag, "[LayerEnd]");

        return layer;
    }

protected:
    explicit OutputClassificationLayer(size_t neuronsN, double learningRate, double momentum) :
        DenseLayer(ActivationFunctions::Identity, learningRate, momentum), m_Outputs(neuronsN)
    {

    }

private:
    std::vector<double> m_Outputs;
};


class OutputRegressionLayer : public DenseLayer // public inheritance to be able to use std::make_shared
{
public:
    explicit OutputRegressionLayer(size_t neuronsN, size_t prevLayerNeuronsN,
        ActivationFunctions afunc, double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        DenseLayer(neuronsN, prevLayerNeuronsN, afunc, learningRate, momentum, seedGen, bias)
    {

    }

    explicit OutputRegressionLayer(const std::vector<std::vector<double>>& layerWeights,
        ActivationFunctions afunc, double learningRate, double momentum,
        const std::shared_ptr<SeedGenerator>& seedGen, double bias = 0.0) :
        DenseLayer(layerWeights, afunc, learningRate, momentum, seedGen, bias)
    {

    }

    explicit OutputRegressionLayer(const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias, ActivationFunctions afunc,
        double learningRate, double momentum, const std::shared_ptr<SeedGenerator>& seedGen) :
        DenseLayer(layerWeights, layerBias, afunc, learningRate, momentum, seedGen)
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

    static OutputRegressionLayer readFromFile(std::ifstream& file)
    {
        std::string tag;

        Utils::checkTag(file, tag, "[LayerBegin]");

        int afunc = 0;
        double momentum = 0.0;
        double learningRate = 0.0;
        size_t inputN = 0;
        size_t outputN = 0;
        file >> tag >> afunc;
        file >> tag >> momentum;
        file >> tag >> learningRate;
        file >> tag >> inputN;
        file >> tag >> outputN;

        OutputRegressionLayer layer(static_cast<ActivationFunctions>(afunc), learningRate, momentum);

        for (size_t n = 0; n < outputN; n++)
        {
            layer.m_Neurons.push_back(Neuron::readFromFile(file));
        }

        Utils::checkTag(file, tag, "[LayerEnd]");

        return layer;
    }

protected:
    explicit OutputRegressionLayer(ActivationFunctions afunc, double learningRate, double momentum) :
        DenseLayer(afunc, learningRate, momentum)
    {

    }
};

}

#endif // YANNL_NEURON_LAYER_H
