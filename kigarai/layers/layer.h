#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "structure/structure.h"
#include <vector>

class Layer
{
protected:
    const Structure structure;
    std::vector<double> values;
    const Structure prevStructure;
    const std::vector<double>& prevValues;
public:
    Layer(const Structure& structure, const Structure& prevStructure, const std::vector<double>& prevValues);
    virtual ~Layer();
    const std::vector<double>& get_values() const;
    const Structure& get_structure() const;
    virtual void calc_values() =0;
    virtual void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues) =0;
    virtual void apply_training(double lambda) =0;
};

#endif // LAYER_H_INCLUDED
