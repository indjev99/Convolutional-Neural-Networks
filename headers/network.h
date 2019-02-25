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
    const std::vector<int> topology;
    const ActivationFunction& activationFunction;
    std::vector<Layer> layers;
    std::vector<double> currInput;
    int batchCnt;
public:
    Network(int inputSize, const std::vector<int>& topology, const ActivationFunction& activationFunction);
    const std::vector<double>& get_output(const std::vector<double> input);
    void accumulate_training(const std::vector<double> targetOutput);
    void apply_training();
};

#endif // NETWORK_H_INCLUDED
