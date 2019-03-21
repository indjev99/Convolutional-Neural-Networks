#ifndef PADDING_LAYER_H_INCLUDED
#define PADDING_LAYER_H_INCLUDED

#include "layer.h"
#include <vector>

class PaddingLayer : public Layer
{
private:
    const Structure padding;
public:
    PaddingLayer(const Structure& padding, const Structure& prevStructure, const std::vector<double>& prevValues);
    ~PaddingLayer();
    void calc_values();
    void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
    void apply_training(double lambda);
};

#endif // PADDING_LAYER_H_INCLUDED
