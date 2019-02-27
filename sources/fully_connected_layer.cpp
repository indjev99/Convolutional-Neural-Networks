#include "../headers/fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer(unsigned int layerSize, const std::vector<double>& prevValues, double varianceFactor, std::default_random_engine& generator)
    : ConvolutionLayer(layerSize,{1,1},{prevValues.size(),1,1},prevValues,varianceFactor,generator) {}
FullyConnectedLayer::~FullyConnectedLayer() {}
void FullyConnectedLayer::calc_values()
{
    for (unsigned int i=0;i<structure.depth;++i)
    {
        values[i]=biases[i];
        for (unsigned int j=0;j<prevStructure.depth;++j)
        {
            values[i]+=weigths[i*prevStructure.depth+j]*prevValues[j];
        }
    }
}
void FullyConnectedLayer::train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (unsigned int i=0;i<prevStructure.depth;++i)
    {
        dCost0dPrevValues[i]=0;
    }
    for (unsigned int i=0;i<structure.depth;++i)
    {
        dCostdBiases[i]+=dCost0dValues[i];
        for (unsigned int j=0;j<prevStructure.depth;++j)
        {
            dCostdWeigths[i*prevStructure.depth+j]+=dCost0dValues[i]*prevValues[j];
            dCost0dPrevValues[j]+=dCost0dValues[i]*weigths[i*prevStructure.depth+j];
        }
    }
}
