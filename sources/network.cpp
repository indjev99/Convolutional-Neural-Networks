#include "../headers/network.h"
#include "../headers/fully_connected_layer.h"
#include "../headers/activation_layer.h"
#include <assert.h>
#include <random>

Network::Network(unsigned int inputSize, const std::vector<unsigned int>& topology, const ActivationFunction& activationFunction, double varianceFactor, int seed)
    : inputSize{inputSize}
    , eta{0}
    , batchCnt{0}
    , batchSize{0}
{
    input.resize(inputSize);
    if (topology.empty()) networkDepth=0;
    else networkDepth=topology.size()*2-1;
    const std::vector<double>* lastValues=&input;
    std::default_random_engine generator;
    generator.seed(seed);
    for (unsigned int i=0;i<networkDepth;++i)
    {
        if (i%2==0)
        {
            assert(topology[i/2]>=0);
            layers.push_back(new FullyConnectedLayer(topology[i/2],*lastValues,varianceFactor,generator));
        }
        else
        {
            layers.push_back(new ActivationLayer(*lastValues,activationFunction));
        }
        lastValues=&layers[i]->get_values();
    }
    output=lastValues;
    outputSize=output->size();
    maxLayerSize=inputSize;
    for (unsigned int i=0;i<topology.size();++i)
    {
        if (maxLayerSize<topology[i]) maxLayerSize=topology[i];
    }
    dCost0dValues[0].resize(maxLayerSize);
    dCost0dValues[1].resize(maxLayerSize);
}
Network::~Network()
{
    for (unsigned int i=0;i<networkDepth;++i)
    {
        delete layers[i];
    }
}
void Network::set_learning_rate(double eta)
{
    this->eta=eta;
}
void Network::set_batch_size(unsigned int batchSize)
{
    this->batchSize=batchSize;
}
const std::vector<double>& Network::get_output(const std::vector<double> input)
{
    assert(input.size()==inputSize);
    for (unsigned int i=0;i<inputSize;++i)
    {
        this->input[i]=input[i];
    }
    for (unsigned int i=0;i<networkDepth;++i)
    {
        layers[i]->calc_values();
    };
    return *output;
}
double Network::train(const std::vector<double> targetOutput)
{
    assert(targetOutput.size()==outputSize);
    double cost0=0;
    double currDiff;
    bool curr=0;
    for (unsigned int i=0;i<outputSize;++i)
    {
        currDiff=(*output)[i]-targetOutput[i];
        cost0+=currDiff*currDiff;
        dCost0dValues[curr][i]=2*currDiff;
    }
    for (unsigned int i=0;i<networkDepth;++i)
    {
        layers[networkDepth-i-1]->train(dCost0dValues[curr],dCost0dValues[!curr]);
        curr=!curr;
    }
    ++batchCnt;
    if (batchCnt==batchSize)
    {
        apply_training();
    }
    return cost0;
}
void Network::apply_training()
{
    for (unsigned int i=0;i<networkDepth;++i)
    {
        layers[i]->apply_training(eta/batchCnt);
    }
    batchCnt=0;
}
