#include "../headers/activation_layer.h"

ActivationLayer::ActivationLayer(const std::vector<double>& prevValues, const ActivationFunction& activationFunction)
    : Layer(prevValues)
    , layerSize{prevValues.size()}
    , activationFunction{activationFunction}
{
    values.resize(layerSize);
}
ActivationLayer::~ActivationLayer() {}
void ActivationLayer::calc_values()
{
    for (int i=0;i<layerSize;++i)
    {
        values[i]=activationFunction.evaluate(prevValues[i]);
    }
}
void ActivationLayer::accumulate_training(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (int i=0;i<layerSize;++i)
    {
        dCost0dPrevValues[i]=dCost0dValues[i]*activationFunction.evaluate_derivative(prevValues[i]);
    }
}
