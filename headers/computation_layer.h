#ifndef COMPUTATION_LAYER_H_INCLUDED
#define COMPUTATION_LAYER_H_INCLUDED

#include "layer.h"
#include <random>

class ComputationLayer : public Layer
{
protected:
    std::vector<double> biases;
    std::vector<double> weigths;
    std::vector<double> dCostdBiases;
    std::vector<double> dCostdWeigths;
public:
    ComputationLayer(unsigned int layerSize, const std::vector<double>& prevValues, unsigned int biasesSize, unsigned int weigthsSize, double variance, std::default_random_engine& generator);
    virtual ~ComputationLayer();
    virtual void calc_values() =0;
    virtual void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues) =0;
    void apply_training(double lambda);
};


#endif // COMPUTATION_LAYER_H_INCLUDED
