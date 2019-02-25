#include "../headers/layer.h"

#include "stdlib.h"

Layer::Layer(int layerSize, int prevLayerSize, const std::vector<double>& prevValues, const ActivationFunction& activationFunction)
    : layerSize{layerSize}
    , prevLayerSize{prevLayerSize}
    , prevValues{prevValues}
    , activationFunction{activationFunction}
    , values(layerSize)
    , biases(layerSize)
    , weigths(layerSize*prevLayerSize)
    , dCostdBiases(layerSize,0)
    , dCostdWeigths(layerSize*prevLayerSize,0)
{
    for (int i=0;i<layerSize;++i)
    {
        biases[i]=(rand()%1001-500)/500.0;
        for (int j=0;j<prevLayerSize;++j)
        {
            weigths[i*prevLayerSize+j]=(rand()%1001-500)/500.0;
        }
    }
}

const std::vector<double>& Layer::get_values() const
{
    return values;
}

void Layer::calc_values()
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

void Layer::accumulate_training(const double* dCost0dValues, double* dCost0dPrevValues, double lambda)
{

}

void Layer::apply_training(int batchSize)
{
    for (int i=0;i<layerSize;++i)
    {
        biases[i]+=dCostdBiases[i]/batchSize;
        dCostdBiases[i]=0;
        for (int j=0;j<prevLayerSize;++j)
        {
            weigths[i*prevLayerSize+j]+=dCostdWeigths[i*prevLayerSize+j]/batchSize;
            dCostdWeigths[i*prevLayerSize+j]=0;
        }
    }
}
