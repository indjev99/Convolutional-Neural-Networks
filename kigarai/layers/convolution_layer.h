#ifndef CONVOLUTION_LAYER_H_INCLUDED
#define CONVOLUTION_LAYER_H_INCLUDED

#include "layer.h"
#include <random>

class ConvolutionLayer : public Layer
{
protected:
    const Structure convolutionStructure;
    std::vector<double> biases;
    std::vector<double> weigths;
    std::vector<double> dCostdBiases;
    std::vector<double> dCostdWeigths;
public:
    ConvolutionLayer(unsigned int depth, const Structure& convolutionStructure, const Structure& prevStructure, const std::vector<double>& prevValues, double varianceFactor, std::default_random_engine& generator);
    ~ConvolutionLayer();
    virtual void calc_values();
    virtual void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
    void apply_training(double lambda);
};


#endif // CONVOLUTION_LAYER_H_INCLUDED
