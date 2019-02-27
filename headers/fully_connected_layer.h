#ifndef FULLY_CONNECTED_LAYER_H_INCLUDED
#define FULLY_CONNECTED_LAYER_H_INCLUDED

#include "computation_layer.h"
#include <random>

class FullyConnectedLayer : public ComputationLayer
{
private:
    const unsigned int layerSize;
    const unsigned int prevLayerSize;
public:
    FullyConnectedLayer(unsigned int layerSize, const std::vector<double>& prevValues, double varianceFactor, std::default_random_engine& generator);
    ~FullyConnectedLayer();
    void calc_values();
    void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
};

#endif // FULLY_CONNECTED_LAYER_H_INCLUDED
