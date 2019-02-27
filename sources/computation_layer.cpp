#include "../headers/computation_layer.h"

#include <math.h>

ComputationLayer::ComputationLayer(unsigned int layerSize, const std::vector<double>& prevValues, unsigned int biasesSize, unsigned int weigthsSize, double variance, std::default_random_engine& generator)
    : Layer(layerSize,prevValues)
    , biases(biasesSize,0)
    , weigths(weigthsSize,0)
    , dCostdBiases(biasesSize,0)
    , dCostdWeigths(weigthsSize,0)
{
    std::normal_distribution<double> distribution(0,sqrt(variance));
    for (unsigned int i=0;i<weigths.size();++i)
    {
        weigths[i]=distribution(generator);
    }
}
ComputationLayer::~ComputationLayer() {};
void ComputationLayer::apply_training(double lambda)
{
    for (unsigned int i=0;i<biases.size();++i)
    {
        biases[i]-=dCostdBiases[i]*lambda;
        dCostdBiases[i]=0;
    }
    for (unsigned int i=0;i<weigths.size();++i)
    {
        weigths[i]-=dCostdWeigths[i]*lambda;
        dCostdWeigths[i]=0;
    }
}
