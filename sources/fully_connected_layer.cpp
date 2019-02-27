#include "../headers/fully_connected_layer.h"

#include <math.h>

FullyConnectedLayer::FullyConnectedLayer(int layerSize, const std::vector<double>& prevValues, std::default_random_engine& generator)
    : Layer(prevValues)
    , layerSize{layerSize}
    , prevLayerSize{prevValues.size()}
{
    values.resize(layerSize);
    biases.resize(layerSize);
    weigths.resize(layerSize*prevLayerSize);
    dCostdBiases.resize(layerSize);
    dCostdWeigths.resize(layerSize*prevLayerSize);
    std::normal_distribution<double> distribution(0,sqrt(2.0/prevLayerSize));
    for (int i=0;i<layerSize;++i)
    {
        biases[i]=0;
        dCostdBiases[i]=0;
        for (int j=0;j<prevLayerSize;++j)
        {
            weigths[i*prevLayerSize+j]=distribution(generator);
            dCostdWeigths[i*prevLayerSize+j]=0;
        }
    }
}
FullyConnectedLayer::~FullyConnectedLayer() {}
void FullyConnectedLayer::calc_values()
{
    for (int i=0;i<layerSize;++i)
    {
        values[i]=biases[i];
        for (int j=0;j<prevLayerSize;++j)
        {
            values[i]+=weigths[i*prevLayerSize+j]*prevValues[j];
        }
    }
}
void FullyConnectedLayer::accumulate_training(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (int i=0;i<prevLayerSize;++i)
    {
        dCost0dPrevValues[i]=0;
    }
    for (int i=0;i<layerSize;++i)
    {
        dCostdBiases[i]+=dCost0dValues[i];
        for (int j=0;j<prevLayerSize;++j)
        {
            dCostdWeigths[i*prevLayerSize+j]+=dCost0dValues[i]*prevValues[j];
            dCost0dPrevValues[j]+=dCost0dValues[i]*weigths[i*prevLayerSize+j];
        }
    }
}
