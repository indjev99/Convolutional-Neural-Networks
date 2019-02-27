#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "activation_function.h"
#include "layer.h"
#include <vector>

class Network
{
private:
    unsigned int networkDepth;
    std::vector<Layer*> layers;
    unsigned int inputSize;
    std::vector<double> input;
    unsigned int outputSize;
    const std::vector<double>* output;
    unsigned int maxLayerSize;
    std::vector<double> dCost0dValues[2];
    double eta;
    unsigned int batchCnt;
    unsigned int batchSize;
public:
    Network(unsigned int inputSize, const std::vector<unsigned int>& topology, const ActivationFunction& activationFunction, double varianceFactor, int seed=0);
    ~Network();
    void set_learning_rate(double eta);
    void set_batch_size(unsigned int batchSize);
    const std::vector<double>& get_output(const std::vector<double> input);
    double train(const std::vector<double> targetOutput);
    void apply_training();
};

#endif // NETWORK_H_INCLUDED
