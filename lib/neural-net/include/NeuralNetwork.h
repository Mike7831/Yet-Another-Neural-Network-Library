#ifndef YANNL_NEURAL_NETWORK_H
#define YANNL_NEURAL_NETWORK_H

#include "NeuronLayer.h"

namespace YANNL
{

class NeuralNetwork
{
public:
    //! Creates a neural network with an input layer of @ref inputSize. It is mandatory
    //! to then add at least one dense layer which will be the output layer.
    //! @param inputSize Number of neurons on the input layer.
    //! @param learningRate Learning rate (eta).
    //! @param momentum The momentum (lambda). 0 by default; no momentum.
    //! @param useSeed Tells whether to use the provided seed (true) or random seed (false).
    //! @param seed The seed to initialize the random function for determining weights, and
    //!   dropout in dropout layers.
    explicit NeuralNetwork(size_t inputSize, double learningRate, double momentum = 0.0,
        bool useSeed = false, unsigned int seed = 0) :
        m_InputSize(inputSize), m_LearningRate(learningRate), m_Momentum(momentum),
        m_SeedGenerator(std::make_shared<SeedGenerator>(useSeed, seed))
    {
        // inputSize is useful to verify the consistency of the network when
        // adding a first hidden layer or when providing inputs.
    }

    // No need to apply the rule of five as the class contains no raw pointers

    //! Tells whether there are layers and if there are layers whether the last one
    //! is an output layer.
    //! @returns Whether there are layers and whether the last one is an output layer.
    bool isLastLayerAnOutput() const
    {
        if (m_Layers.size() == 0)
        {
            return false;
        }

        return m_Layers.back()->type() == LayerType::OutputClassification
            || m_Layers.back()->type() == LayerType::OutputRegression;
    }

    //! Adds a hidden layer which is a dense layer. See
    //!   @ref addDenseLayer(LayerType, size_t, ActivationFunctions, double)
    //! @throws std::domain_error If this hidden layer is added after an output layer.
    void addHiddenLayer(size_t neuronsN, ActivationFunctions afunc, double bias = 0.0)
    {
        // Verify whether the hidden layer is not added after an output layer
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add hidden layer] Cannot add a hidden layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::Hidden, neuronsN, afunc, bias);
    }

    //! Adds a hidden layer which is a dense layer. See
    //!   @ref addDenseLayer(LayerType, const std::vector<std::vector<double>>, ActivationFunctions, double)
    //! @throws std::domain_error If this hidden layer is added after an output layer.
    //! @throws std::domain_error In case on inconsistencies in layerWeights. See
    //!   @ref addDenseLayer(LayerType, const std::vector<std::vector<double>>, ActivationFunctions, double)
    void addHiddenLayer(const std::vector<std::vector<double>>& layerWeights,
        ActivationFunctions afunc, double bias = 0.0)
    {
        // Verify whether the hidden layer is not added after an output layer
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add hidden layer] Cannot add a hidden layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::Hidden, layerWeights, afunc, bias);
    }

    void addHiddenLayer(const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias, ActivationFunctions afunc)
    {
        // Verify whether the hidden layer is not added after an output layer
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add hidden layer] Cannot add a hidden layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::Hidden, layerWeights, layerBias, afunc);
    }


    //! Adds an output classification layer which is a dense layer. See
    //! @ref addDenseLayer(LayerType, size_t, ActivationFunctions, double)
    //! @throws std::domain_error If this output layer is added after an output layer.
    //! @throws std::domain_error See
    //!   @ref addDenseLayer(LayerType, size_t, ActivationFunctions, double)
    void addOutputClassificationLayer(size_t neuronsN, double bias = 0.0)
    {
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add output layer] Cannot add an output layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::OutputClassification, neuronsN, ActivationFunctions::Identity, bias);
    }

    //! Adds an output classification layer which is a dense layer. See
    //! @ref addDenseLayer(LayerType, const std::vector<std::vector<double>>, ActivationFunctions, double)
    //! @throws std::domain_error If this output layer is added after an output layer.
    //! @throws std::domain_error See
    //!   @ref addDenseLayer(LayerType, const std::vector<std::vector<double>>, ActivationFunctions, double)
    void addOutputClassificationLayer(const std::vector<std::vector<double>>& layerWeights,
        double bias = 0.0)
    {
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add output layer] Cannot add an output layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::OutputClassification, layerWeights, ActivationFunctions::Identity, bias);
    }

    void addOutputClassificationLayer(const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias)
    {
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add output layer] Cannot add an output layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::OutputClassification, layerWeights, layerBias, ActivationFunctions::Identity);
    }

    //! Adds an output regression layer which is a dense layer. See
    //! @ref addDenseLayer(LayerType, size_t, ActivationFunctions, double)
    //! @throws std::domain_error If this output layer is added after an output layer.
    //! @throws std::domain_error See
    //!   @ref addDenseLayer(LayerType, size_t, ActivationFunctions, double)
    void addOutputRegressionLayer(size_t neuronsN, ActivationFunctions afunc, double bias = 0.0)
    {
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add output layer] Cannot add an output layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::OutputRegression, neuronsN, afunc, bias);
    }

    //! Adds an output regression layer which is a dense layer. See
    //! @ref addDenseLayer(LayerType, const std::vector<std::vector<double>>, ActivationFunctions, double)
    //! @throws std::domain_error If this output layer is added after an output layer.
    //! @throws std::domain_error See
    //!   @ref addDenseLayer(LayerType, const std::vector<std::vector<double>>, ActivationFunctions, double)
    void addOutputRegressionLayer(const std::vector<std::vector<double>>& layerWeights,
        ActivationFunctions afunc, double bias = 0.0)
    {
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add output layer] Cannot add an output layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::OutputRegression, layerWeights, afunc, bias);
    }

    void addOutputRegressionLayer(const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias, ActivationFunctions afunc)
    {
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add output layer] Cannot add an output layer after an output layer.").str()
            );
        }

        addDenseLayer(LayerType::OutputRegression, layerWeights, layerBias, afunc);
    }

    void addDropoutLayer(double dropoutRate = 0.5)
    {
        if (isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add dropout layer] Cannot add a dropout layer after an output layer.").str()
            );
        }

        if (m_Layers.size() == 0)
        {
            m_Layers.push_back(std::make_shared<DropoutLayer>(dropoutRate,
                m_InputSize, m_SeedGenerator));
        }
        else
        {
            m_Layers.push_back(std::make_shared<DropoutLayer>(dropoutRate,
                m_Layers.back()->size(), m_SeedGenerator));
        }
    }

    //! Prints information on the neural network to the provided output stream.
    //! @param os Output stream where to print to information on the neural network.
    void inspect(std::ostream& os) const
    {
        os << "------" << "\n";
        os << "* Inputs: " << m_InputSize << "\n";
        os << "------" << "\n";
        size_t weightN = 1;

        for (size_t n = 0; n < m_Layers.size(); n++)
        {
            m_Layers[n]->inspect(os, weightN);
            os << "------" << "\n";
        }
    }

    void updateLearningRate(double learningRate)
    {
        m_LearningRate = learningRate;

        std::for_each(m_Layers.begin(), m_Layers.end(),
            [&](const std::shared_ptr<NeuronLayer>& layer)
            {
                layer->updateLearningRate(learningRate);
            });
    }

    //! Propagates the provided input forward through all the neural network and calculates
    //! the output. This is a prerequisite to calculating the mean squared error or
    //! propagating backward.
    //! @param inputs Vector of inputs. Should correspond to the number of input neurons.
    //! @param ignoreDropout Tells whether to ignore the dropout layer. Dropout layer should
    //!   be taken into account during the forward and back propagation, but not when
    //!   calculating the output once the model is trained.
    //! @returns Vector of calculated outputs, of size of the output layer.
    //! @throws std::domain_error If there are no output layers or if the size of input provided
    //!   is inconsistent with the size of the input layer (number of values provided <>
    //!   number of neurons on the input layer).
    std::vector<double> propagateForward(const std::vector<double>& inputs, bool ignoreDropout = false)
    {
        if (!isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Propagate forward] Neural network has no output layers.").str()
            );
        }
        else if (inputs.size() != m_InputSize)
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Propagate forward] Input size is inconsistent: expected "
                << m_InputSize << " provided " << inputs.size() << ".").str()
            );
        }

        std::vector<double> outputs(inputs);

        for (size_t n = 0; n < m_Layers.size(); n++)
        {
            // std::move is optional as RVO will do the same
            outputs = std::move(m_Layers[n]->propagateForward(outputs, ignoreDropout));
        }

        return outputs;
    }

    //! @returns Most probable class when using several neurons on the output layer
    //!   for classification problems, i.e. regressors.
    //! @throws std::domain_error If no output layers have been added.
    size_t probableClass() const
    {
        if (!isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Get probable output] Neural network has no output layers.").str()
            );
        }

        return m_Layers.back()->probableClass();
    }

    //! Calculates the error between the actual output and the expected output. Mean squared
    //! error in case the output layer is regression @ref OutputRegressionLayer, cross entropy
    //! error in case the output layer is classification @ref OutputClassificationLayer.
    //! Propagate forward first before calculating mean squared error.
    //! @param expectedOutputs Vector of expected outputs to be compared to the internal
    //!   vector of actual outputs.
    //! @returns Mean squared error. If there are N outputs the mean squared error shall
    //!   be Total squared error" / N.
    //! @throws std::domain_error If the number of nodes on output layer is different from
    //!   the number of outputs provided. Or if the neural network has no output layers.
    double calcError(const std::vector<double>& expectedOutputs) const
    {
        if (!isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Calculate error] Neural network has no output layers.").str()
            );
        }
        else if (expectedOutputs.size() != m_Layers.back()->size())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Calculate error] Output size is inconsistent with output layer size: expected "
                << m_Layers.back()->size() << " provided " << expectedOutputs.size() << ".").str()
            );
        }

        if (m_Layers.back()->type() == LayerType::OutputClassification)
        {
            return m_Layers.back()->calcError(expectedOutputs);
        }
        else
        {
            return m_Layers.back()->calcError(expectedOutputs) / m_Layers.back()->size();
        }
    }

    //! Converts the single-value expected output to a vector and calls
    //! the @ref calcError(const std::vector<double>&)
    //! @param expectedOutput Single-value output expected.
    //! @returns See @ref calcError(const std::vector<double>&)
    //! @throws See @ref calcError(const std::vector<double>&)
    double calcError(double expectedOutput)
    {
        const std::vector<double> expectedOutputs{ expectedOutput };
        return calcError(expectedOutputs);
    }

    //! Propagates the expected output backward to first calculate the delta on each neuron
    //! of each layer, and second to update the weights.
    //! Propagate forward first before propagating backward.
    //! @param expectedOutputs Vector of expected outputs to be compared to the internal
    //!   vector of actual outputs.
    //! @throws std::domain_error If the number of nodes on output layer is different from
    //!   the number of outputs provided. Or if the neural network has no output layers.
    void propagateBackward(const std::vector<double>& expectedOutputs)
    {
        if (!isLastLayerAnOutput())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Propagate backward] Neural network has no output layers.").str()
            );
        }
        else if (expectedOutputs.size() != m_Layers.back()->size())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Propagate backward] Output size is inconsistent with output layer size: expected "
                << m_Layers.back()->size() << " provided " << expectedOutputs.size() << ".").str()
            );
        }

        // Propagate backward on the output layer
        m_Layers.back()->propagateBackwardOuputLayer(expectedOutputs);

        // Propagate backward for each hidden layer if there are hidden layers
        for (auto layer = m_Layers.rbegin() + 1; layer != m_Layers.rend(); layer++)
        {
            std::shared_ptr<const NeuronLayer> nextLayer = *(layer - 1);
            (*layer)->propagateBackwardHiddenLayer(*nextLayer);
        }

        for (size_t n = 0; n < m_Layers.size(); n++)
        {
            m_Layers[n]->updateWeights();
        }
    }

    //! Converts the single-value expected output to a vector and calls
    //! the @ref propagateBackward(const std::vector<double>&)
    //! @param expectedOutput Single-value output expected.
    void propagateBackward(double expectedOutput)
    {
        const std::vector<double> expectedOutputs{ expectedOutput };
        propagateBackward(expectedOutputs);
    }

    bool saveToFile(const std::string& filepath) const
    {
        std::ofstream output(filepath);

        if (!output)
        {
            return false;
        }

        output.precision(std::numeric_limits<double>::max_digits10);

        output << "[NetworkBegin] \n"
            << "LayerNumber: " << m_Layers.size() << "\n"
            << "Momentum: " << m_Momentum << "\n"
            << "LearningRate: " << m_LearningRate << "\n"
            << "InputSize: " << m_InputSize << "\n"
            << "SeedGenerator: " << *m_SeedGenerator << "\n\n";

        for (size_t n = 0; n < m_Layers.size(); n++)
        {
            m_Layers[n]->saveToFile(output);
        }

        output << "[NetworkEnd] \n";

        output.close();

        return true;
    }

    //! Loads a neural network from a previously serialized network to a file with
    //! @ref saveToFile(const std::string&)
    //! @param filepath Path to the file containing the serialized neural network to load.
    //! @returns The reconstructed neural network.
    //! @throws std::ifstream::failure In case the file is not accessible.
    //! @throws std::domain_error In case the neural network has no layers, thus
    //!   no output layers, or if it is ill-formed i.e. not the expected tags.
    static NeuralNetwork loadFromFile(const std::string& filepath)
    {
        std::ifstream file(filepath);

        if (!file)
        {
            throw std::ifstream::failure(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Load network] Cannot build neural network from file. "
                << filepath << " is not accessible.").str()
            );
        }

        std::string tag;

        Utils::checkTag(file, tag, "[NetworkBegin]");
        file >> tag; // LayerNumber

        size_t layersN = 0;
        file >> layersN;

        if (layersN == 0)
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Load network] Neural network has no layers.").str()
            );
        }

        double momentum = 0.0;
        double learningRate = 0.0;
        size_t inputSize = 0;
        SeedGenerator generator;
        file >> tag >> momentum;
        file >> tag >> learningRate;
        file >> tag >> inputSize;
        file >> tag >> generator;

        NeuralNetwork net(inputSize, learningRate, momentum, generator);

        for (size_t l = 0; l < layersN; l++)
        {
            int layerType = 0;

            file >> tag >> layerType;

            if (static_cast<LayerType>(layerType) == LayerType::Hidden)
            {
                net.m_Layers.push_back(
                    std::make_shared<HiddenLayer>(
                        HiddenLayer::readFromFile(file)));
            }
            else if (static_cast<LayerType>(layerType) == LayerType::Dropout)
            {
                net.m_Layers.push_back(
                    std::make_shared<DropoutLayer>(
                        DropoutLayer::readFromFile(file)));
            }
            else if (static_cast<LayerType>(layerType) == LayerType::OutputClassification)
            {
                net.m_Layers.push_back(
                    std::make_shared<OutputClassificationLayer>(
                        OutputClassificationLayer::readFromFile(file)));
            }
            else // OutputRegressionLayer
            {
                net.m_Layers.push_back(
                    std::make_shared<OutputRegressionLayer>(
                        OutputRegressionLayer::readFromFile(file)));
            }
        }

        file.close();

        return net;
    }

private:
    const size_t m_InputSize = 0;
    double m_LearningRate = 0.0;
    const double m_Momentum = 0.0;
    const std::shared_ptr<SeedGenerator> m_SeedGenerator;
    std::vector<std::shared_ptr<NeuronLayer>> m_Layers;

    explicit NeuralNetwork(size_t inputSize, double learningRate, double momentum,
        const SeedGenerator& generator) :
        m_InputSize(inputSize), m_LearningRate(learningRate), m_Momentum(momentum),
        m_SeedGenerator(std::make_shared<SeedGenerator>(generator))
    {

    }

    //! Adds a dense layer to the neural network with random weights.
    //! @param layerType Either OutputRegressionLayer or HiddenLayer
    //! @param neuronsN Number of neurons.
    //! @param afunc Activation function for the layer.
    //! @param bias Additional bias if necessary. 0 by default.
    void addDenseLayer(LayerType layerType, size_t neuronsN, ActivationFunctions afunc, double bias = 0.0)
    {
        switch (layerType)
        {
        case LayerType::Hidden:
            m_Layers.push_back(std::make_shared<HiddenLayer>(
                neuronsN, lastLayerSize(), afunc, m_LearningRate, m_Momentum, m_SeedGenerator, bias));
            break;
        case LayerType::OutputClassification:
            m_Layers.push_back(std::make_shared<OutputClassificationLayer>(
                neuronsN, lastLayerSize(), m_LearningRate, m_Momentum, m_SeedGenerator, bias));
            break;
        case LayerType::OutputRegression:
            m_Layers.push_back(std::make_shared<OutputRegressionLayer>(
                neuronsN, lastLayerSize(), afunc, m_LearningRate, m_Momentum, m_SeedGenerator, bias));
            break;
        case LayerType::Dropout:
            // Nothing
            break;
        }

    }

    //! Adds a dense layer to the neural network with predefined weights.
    //! @param layerWeights Predefined weights. 2D vector. 1st D are the neurons,
    //!   2nd D are the weights.
    //! @param afunc Activation function for the layer.
    //! @param bias Additional bias if necessary. 0 by default.
    //! @throws std::domain_error If there are inconsistencies in the @ref layerWeights e.g.
    //!   input size of 3 with 2 weights on the first neuron of the first layer is
    //!   inconsistent (3 weights expected for each of the neurons of the first layer).
    void addDenseLayer(LayerType layerType, const std::vector<std::vector<double>>& layerWeights,
        ActivationFunctions afunc, double bias = 0.0)
    {
        // Check whether the number of weights provided is consistent with
        // the number of neurons from the previous layer.
        for (size_t n = 0; n < layerWeights.size(); n++)
        {
            if (layerWeights[n].size() != lastLayerSize())
            {
                throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                    << "[Add layer] Layer size is inconsistent: expected " << lastLayerSize()
                    << " provided " << layerWeights[n].size() << " on neuron "
                    << (n + 1) << ".").str()
                );
            }
        }

        // Add the layer if there is no exception before
        switch (layerType)
        {
        case LayerType::Hidden:
            m_Layers.push_back(std::make_shared<HiddenLayer>(
                layerWeights, afunc, m_LearningRate, m_Momentum, m_SeedGenerator, bias));
            break;
        case LayerType::OutputClassification:
            m_Layers.push_back(std::make_shared<OutputClassificationLayer>(
                layerWeights, m_LearningRate, m_Momentum, m_SeedGenerator, bias));
            break;
        case LayerType::OutputRegression:
            m_Layers.push_back(std::make_shared<OutputRegressionLayer>(
                layerWeights, afunc, m_LearningRate, m_Momentum, m_SeedGenerator, bias));
            break;
        case LayerType::Dropout:
            // Nothing
            break;
        }

    }

    void addDenseLayer(LayerType layerType, const std::vector<std::vector<double>>& layerWeights,
        const std::vector<double>& layerBias, ActivationFunctions afunc)
    {
        // Check whether the number of weights provided is consistent with
        // the number of neurons from the previous layer.
        for (size_t n = 0; n < layerWeights.size(); n++)
        {
            if (layerWeights[n].size() != lastLayerSize())
            {
                throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                    << "[Add layer] Layer size is inconsistent: expected " << lastLayerSize()
                    << " provided " << layerWeights[n].size() << " on neuron "
                    << (n + 1) << ".").str()
                );
            }
        }

        // There should be the same number of bias provided than outputs
        if (layerWeights.size() != layerBias.size())
        {
            throw std::domain_error(static_cast<const std::ostringstream&>(std::ostringstream()
                << "[Add layer] Bias list provided is inconsistent: expected " << layerWeights.size()
                << " provided " << layerBias.size() << ".").str()
            );
        }

        // Add the layer if there is no exception before
        switch (layerType)
        {
        case LayerType::Hidden:
            m_Layers.push_back(std::make_shared<HiddenLayer>(
                layerWeights, layerBias, afunc, m_LearningRate, m_Momentum, m_SeedGenerator));
            break;
        case LayerType::OutputClassification:
            m_Layers.push_back(std::make_shared<OutputClassificationLayer>(
                layerWeights, layerBias, m_LearningRate, m_Momentum, m_SeedGenerator));
            break;
        case LayerType::OutputRegression:
            m_Layers.push_back(std::make_shared<OutputRegressionLayer>(
                layerWeights, layerBias, afunc, m_LearningRate, m_Momentum, m_SeedGenerator));
            break;
        case LayerType::Dropout:
            // Nothing
            break;
        }

    }

    //! @returns Size of the last added layer (number of neurons) if there is a last added layer
    //!   otherwise the input size (number of input neurons) if this is the only one.
    size_t lastLayerSize() const
    {
        if (m_Layers.size() > 0)
        {
            return m_Layers.back()->size();
        }
        else
        {
            return m_InputSize;
        }
    }
};

}

#endif // YANNL_NEURAL_NETWORK_H
