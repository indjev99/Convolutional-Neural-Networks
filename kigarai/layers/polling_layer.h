#ifndef POLLING_LAYER_H_INCLUDED
#define POLLING_LAYER_H_INCLUDED

#include "layer.h"
#include <vector>

class PollingLayer : public Layer
{
private:
    const Structure field;
    std::vector<int> maxSource;
public:
    PollingLayer(const Structure& field, const Structure& prevStructure, const std::vector<double>& prevValues);
    ~PollingLayer();
    void calc_values();
    void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues);
    void apply_training(double lambda);
};

#endif // POLLING_LAYER_H_INCLUDED
