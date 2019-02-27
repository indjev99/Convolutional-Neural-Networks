#include "../headers/fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer(unsigned int layerSize, const std::vector<double>& prevValues, double varianceFactor, std::default_random_engine& generator)
    : ComputationLayer(layerSize,prevValues,layerSize,layerSize*prevValues.size(),varianceFactor/prevValues.size(),generator)
    , layerSize{layerSize}
    , prevLayerSize{prevValues.size()} {}
FullyConnectedLayer::~FullyConnectedLayer() {}
void FullyConnectedLayer::calc_values()
{
    for (unsigned int i=0;i<layerSize;++i)
    {
        values[i]=biases[i];
        for (unsigned int j=0;j<prevLayerSize;++j)
        {
            values[i]+=weigths[i*prevLayerSize+j]*prevValues[j];
        }
    }
}
void FullyConnectedLayer::train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (unsigned int i=0;i<prevLayerSize;++i)
    {
        dCost0dPrevValues[i]=0;
    }
    for (unsigned int i=0;i<layerSize;++i)
    {
        dCostdBiases[i]+=dCost0dValues[i];
        for (unsigned int j=0;j<prevLayerSize;++j)
        {
            dCostdWeigths[i*prevLayerSize+j]+=dCost0dValues[i]*prevValues[j];
            dCost0dPrevValues[j]+=dCost0dValues[i]*weigths[i*prevLayerSize+j];
        }
    }
}
