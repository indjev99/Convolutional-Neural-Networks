#ifndef FULLY_CONNECTED_LAYER_H_INCLUDED
#define FULLY_CONNECTED_LAYER_H_INCLUDED

#include "layer.h"
#include <random>

class FullyConnectedLayer : public Layer
{
private:
    const int layerSize;
    const int prevLayerSize;
public:
    FullyConnectedLayer(int layerSize, const std::vector<double>& prevValues, std::default_random_engine& generator);
    ~FullyConnectedLayer();
    void calc_values();
    void accumulate_training(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
};

#endif // FULLY_CONNECTED_LAYER_H_INCLUDED
