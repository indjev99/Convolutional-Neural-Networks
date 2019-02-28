#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "../activation_functions/activation_function.h"
#include "../layers/layer.h"
#include "../layers/structure/structure.h"
#include <vector>
#include <random>

class Network
{
private:
    unsigned int networkDepth;
    std::vector<Layer*> layers;
    unsigned int inputSize;
    std::vector<double> input;
    Structure outputStructure;
    const std::vector<double>* output;
    std::vector<double> dCost0dValues[2];
    double eta;
    unsigned int batchCnt;
    unsigned int batchSize;
    unsigned int maxLayerSize;
    std::default_random_engine generator;
    void add_layer(Layer* layer);
public:
    Network(const Structure& inputStructure, int seed=0);
    Network(unsigned int inputSize, int seed=0);
    ~Network();
    void add_activation_layer(const ActivationFunction& activationFunction);
    void add_fully_connected_layer(unsigned int layerSize, double vairanceFactor);
    void add_convolution_layer(unsigned int depth, const Structure& convolutionStructure, double varianceFactor);
    void set_learning_rate(double eta);
    void set_batch_size(unsigned int batchSize);
    const std::vector<double>& get_output(const std::vector<double> input);
    double train(const std::vector<double> targetOutput);
    void apply_training();
};

#endif // NETWORK_H_INCLUDED
