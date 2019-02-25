#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "activation_function.h"
#include <vector>

class Layer
{
private:
    const int layerSize;
    const int prevLayerSize;
    const std::vector<double>& prevValues;
    const ActivationFunction& activationFunction;
    std::vector<double> values;
    std::vector<double> biases;
    std::vector<double> weigths;
    std::vector<double> dCostdBiases;
    std::vector<double> dCostdWeigths;
public:
    Layer(int layerSize, int prevLayerSize, const std::vector<double>& prevValues, const ActivationFunction& activationFunction);
    const std::vector<double>& get_values() const;
    void calc_values();
    void accumulate_training(const double* dCost0dValues, double* dCost0dPrevValues, double lambda);
    void apply_training(int batchSize);
};

#endif // LAYER_H_INCLUDED
