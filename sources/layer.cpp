#include "../headers/layer.h"

Layer::Layer(const std::vector<double>& prevValues)
    : prevValues{prevValues} {}
Layer::~Layer() {}
const std::vector<double>& Layer::get_values() const
{
    return values;
}
void Layer::apply_training(double lambda)
{
    for (int i=0;i<biases.size();++i)
    {
        biases[i]-=dCostdBiases[i]*lambda;
        dCostdBiases[i]=0;
    }
    for (int i=0;i<weigths.size();++i)
    {
        weigths[i]-=dCostdWeigths[i]*lambda;
        dCostdWeigths[i]=0;
    }
}
