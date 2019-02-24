#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "network.h"

class Network::Layer
{
private:
    int layerSize;
    int prevLayerSize;
    double* values;
    const double* prevValues;
    double* weigths;
    double* biases;
    double* dCost0dBiases;
    double* dCost0dWeigths;
public:
    Layer(int layerSize, int prevLayerSize, const double* prevValues);
    ~Layer();
    const double* getValues();
    void calculateDerrivatives(const double* dCost0dValues, double* dCost0dPrevValues, const double lambda);
    void applyTraining(const int batchSize);
};

#endif // LAYER_H_INCLUDED
