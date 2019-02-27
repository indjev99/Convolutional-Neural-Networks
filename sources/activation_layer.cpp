#include "../headers/activation_layer.h"

ActivationLayer::ActivationLayer(const Structure& prevStructure, const std::vector<double>& prevValues, const ActivationFunction& activationFunction)
    : Layer(prevStructure,prevStructure,prevValues)
    , activationFunction{activationFunction} {}
ActivationLayer::~ActivationLayer() {}
void ActivationLayer::calc_values()
{
    for (unsigned int i=0;i<structure.size();++i)
    {
        values[i]=activationFunction.evaluate(prevValues[i]);
    }
}
void ActivationLayer::train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (unsigned int i=0;i<structure.size();++i)
    {
        dCost0dPrevValues[i]=dCost0dValues[i]*activationFunction.evaluate_derivative(prevValues[i]);
    }
}
void ActivationLayer::apply_training(double lambda) {}
