#include "../headers/network.h"
#include <assert.h>

#include <iostream>
using namespace std;

Network::Network(int inputSize, const std::vector<int>& topology, const ActivationFunction& activationFunction, std::default_random_engine& generator)
    : inputSize{inputSize}
    , networkDepth{topology.size()}
    , outputSize{topology[networkDepth-1]}
    , activationFunction{activationFunction}
    , batchCnt{0}
{
    assert(inputSize>=0);
    input.resize(inputSize);
    const std::vector<double>* lastValues=&input;
    for (int i=0;i<networkDepth;++i)
    {
        assert(topology[i]>=0);
        layers.push_back(new Layer(topology[i],*lastValues,activationFunction,generator));
        lastValues=&layers[i]->get_values();
    }
    maxLayerSize=inputSize;
    for (int i=0;i<networkDepth;++i)
    {
        if (maxLayerSize<topology[i]) maxLayerSize=topology[i];
    }
    dCost0dValues[0].resize(maxLayerSize);
    dCost0dValues[1].resize(maxLayerSize);
}

Network::~Network()
{
    for (int i=0;i<networkDepth;++i)
    {
        delete layers[i];
    }
}

void Network::set_eta(double eta)
{
    this->eta=eta;
}

const std::vector<double>& Network::get_output(const std::vector<double> input)
{
    assert(input.size()==inputSize);
    for (int i=0;i<inputSize;++i)
    {
        this->input[i]=input[i];
    }
    for (int i=0;i<networkDepth;++i)
    {
        layers[i]->calc_values();
    }
    return layers[networkDepth-1]->get_values();
}

double Network::accumulate_training(const std::vector<double> targetOutput)
{
    assert(targetOutput.size()==outputSize);
    const std::vector<double>& output=layers[networkDepth-1]->get_values();
    double cost0=0;
    double currDiff;
    bool curr=0;
    for (int i=0;i<outputSize;++i)
    {
        currDiff=output[i]-targetOutput[i];
        //cerr<<output[i]<<" "<<targetOutput[i]<<"\n";
        cost0+=currDiff*currDiff;
        dCost0dValues[curr][i]=2*currDiff;
    }
    for (int i=networkDepth-1;i>=0;--i)
    {
        /*cerr<<"layer: "<<i+1<<": ";
        for (int j=0;j<layers[i]->get_values().size();++j)
        {
            cerr<<dCost0dValues[curr][j]<<" ";
        }
        cerr<<"\n";*/
        layers[i]->accumulate_training(dCost0dValues[curr],dCost0dValues[!curr]);
        curr=!curr;
    }
    ++batchCnt;
    return cost0;
}

void Network::apply_training()
{
    for (int i=0;i<networkDepth;++i)
    {
        layers[i]->apply_training(eta/batchCnt);
    }
    batchCnt=0;
}
