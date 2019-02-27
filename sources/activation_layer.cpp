#include "../headers/activation_layer.h"

ActivationLayer::ActivationLayer(const std::vector<double>& prevValues, const ActivationFunction& activationFunction)
    : Layer(prevValues.size(),prevValues)
    , layerSize{prevValues.size()}
    , activationFunction{activationFunction} {}
ActivationLayer::~ActivationLayer() {}
void ActivationLayer::calc_values()
{
    for (unsigned int i=0;i<layerSize;++i)
    {
        values[i]=activationFunction.evaluate(prevValues[i]);
    }
}
void ActivationLayer::train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (unsigned int i=0;i<layerSize;++i)
    {
        dCost0dPrevValues[i]=dCost0dValues[i]*activationFunction.evaluate_derivative(prevValues[i]);
    }
}
void ActivationLayer::apply_training(double lambda) {}
