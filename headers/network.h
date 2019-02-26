#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "activation_function.h"
#include "layer.h"
#include <vector>

class Network
{
private:
    int inputSize;
    int networkDepth;
    int outputSize;
    const ActivationFunction& activationFunction;
    std::vector<Layer*> layers;
    std::vector<double> input;
    double eta;
    int maxLayerSize;
    std::vector<double> dCost0dValues[2];
    int batchCnt;
public:
    Network(int inputSize, const std::vector<int>& topology, const ActivationFunction& activationFunction, std::default_random_engine& generator);
    ~Network();
    void set_eta(double eta);
    const std::vector<double>& get_output(const std::vector<double> input);
    double accumulate_training(const std::vector<double> targetOutput);
    void apply_training();
};

#endif // NETWORK_H_INCLUDED
