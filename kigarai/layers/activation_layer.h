#ifndef ACTIVATION_LAYER_H_INCLUDED
#define ACTIVATION_LAYER_H_INCLUDED

#include "layer.h"
#include "../activation_functions/activation_function.h"

class ActivationLayer : public Layer
{
private:
    const ActivationFunction& activationFunction;
public:
    ActivationLayer(const Structure& prevStructure, const std::vector<double>& prevValues, const ActivationFunction& activationFunction);
    ~ActivationLayer();
    void calc_values();
    void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
    void apply_training(double lambda);
};

#endif // ACTIVATION_LAYER_H_INCLUDED
