#ifndef ACTIVATION_LAYER_H_INCLUDED
#define ACTIVATION_LAYER_H_INCLUDED

#include "layer.h"
#include "activation_function.h"

class ActivationLayer : public Layer
{
private:
    const int layerSize;
    const ActivationFunction& activationFunction;
public:
    ActivationLayer(const std::vector<double>& prevValues, const ActivationFunction& activationFunction);
    ~ActivationLayer();
    void calc_values();
    void accumulate_training(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
};

#endif // ACTIVATION_LAYER_H_INCLUDED
