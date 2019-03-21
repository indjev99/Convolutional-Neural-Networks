#include "network.h"
#include "../layers/layers.h"
#include <assert.h>

#include <iostream>
using namespace std;

Network::Network(const Structure& inputStructure, int seed)
    : networkDepth{0}
    , inputSize{inputStructure.size()}
    , input(inputStructure.size())
    , outputStructure{inputStructure}
    , output{&input}
    , eta{0}
    , batchCnt{0}
    , batchSize{0}
    , maxLayerSize{inputSize}
{
    generator.seed(seed);
}
Network::Network(unsigned int inputSize, int seed)
    : Network({inputSize,1,1},seed) {}
Network::~Network()
{
    for (unsigned int i=0;i<networkDepth;++i)
    {
        delete layers[i];
    }
}
void Network::add_layer(Layer* newLayer)
{
    output=&newLayer->get_values();
    outputStructure=newLayer->get_structure();
    layers.push_back(newLayer);
    if (outputStructure.size()>maxLayerSize)
    {
        maxLayerSize=outputStructure.size();
    }
    ++networkDepth;
}
void Network::add_padding_layer(const Structure& padding)
{
    assert(padding.depth==0);
    Layer* newLayer=new PaddingLayer(padding,outputStructure,*output);
    add_layer(newLayer);
}
void Network::add_polling_layer(const Structure& field)
{
    assert(field.depth==0);
    Layer* newLayer=new PollingLayer(field,outputStructure,*output);
    add_layer(newLayer);
}
void Network::add_activation_layer(const ActivationFunction& activationFunction)
{
    Layer* newLayer=new ActivationLayer(outputStructure,*output,activationFunction);
    add_layer(newLayer);
}
void Network::add_fully_connected_layer(unsigned int layerSize, double varianceFactor)
{
    assert(varianceFactor>=0);
    Layer* newLayer=new FullyConnectedLayer(layerSize,*output,varianceFactor,generator);
    add_layer(newLayer);
}
void Network::add_convolution_layer(unsigned int depth, const Structure& convolutionStructure, double varianceFactor)
{
    assert(convolutionStructure.depth==0);
    assert(varianceFactor>=0);
    Layer* newLayer=new ConvolutionLayer(depth,convolutionStructure,outputStructure,*output,varianceFactor,generator);
    add_layer(newLayer);
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
    assert(targetOutput.size()==outputStructure.size());
    if (maxLayerSize>dCost0dValues[0].size())
    {
        dCost0dValues[0].resize(maxLayerSize);
        dCost0dValues[1].resize(maxLayerSize);
    }
    double cost0=0;
    double currDiff;
    bool curr=0;
    for (unsigned int i=0;i<outputStructure.size();++i)
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
