#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "activation_function.h"
#include <vector>
#include <random>

class Layer
{
private:
    const int layerSize;
    const int prevLayerSize;
    const std::vector<double>& prevValues;
    const ActivationFunction& activationFunction;
    std::vector<double> rawValues;
    std::vector<double> values;
    std::vector<double> biases;
    std::vector<double> weigths;
    std::vector<double> dCostdBiases;
    std::vector<double> dCostdWeigths;
public:

    Layer(int layerSize, const std::vector<double>& prevValues, const ActivationFunction& activationFunction, std::default_random_engine& generator);
    const std::vector<double>& get_values() const;
    void calc_values();
    void accumulate_training(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
    void apply_training(double lambda);
};

#endif // LAYER_H_INCLUDED
