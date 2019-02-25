#include "../headers/network.h"
#include <assert.h>

Network::Network(int inputSize, const std::vector<int>& topology, const ActivationFunction& activationFunction)
    : inputSize{inputSize}
    , networkDepth{topology.size()}
    , topology{topology}
    , activationFunction{activationFunction}
    , batchCnt{0}
{
    assert(inputSize>=0);
    currInput.resize(inputSize);
    int lastSize=inputSize;
    const std::vector<double>* lastValues=&currInput;
    for (int i=0;i<networkDepth;++i)
    {
        assert(topology[i]>=0);
        layers.push_back({topology[i],lastSize,*lastValues,activationFunction});
        lastSize=topology[i];
        lastValues=&layers[i].get_values();
    }
}

const std::vector<double>& Network::get_output(const std::vector<double> input)
{
    assert(input.size()==inputSize);
    for (int i=0;i<inputSize;++i)
    {
        currInput[i]=input[i];
    }
    for (int i=0;i<networkDepth;++i)
    {
        layers[i].calc_values();
    }
    return layers[networkDepth-1].get_values();
}

void Network::accumulate_training(const std::vector<double> targetOutput)
{
    assert(targetOutput.size()==topology[networkDepth-1]);
    ++batchCnt;
}

void Network::apply_training()
{
    for (int i=0;i<networkDepth;++i)
    {
        layers[i].apply_training(batchCnt);
    }
    batchCnt=0;
}
