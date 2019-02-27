#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include <vector>

class Layer
{
protected:
    const std::vector<double>& prevValues;
    std::vector<double> values;
public:
    Layer(unsigned int layerSize, const std::vector<double>& prevValues);
    virtual ~Layer();
    const std::vector<double>& get_values() const;
    virtual void calc_values() =0;
    virtual void train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues) =0;
    virtual void apply_training(double lambda) =0;
};

#endif // LAYER_H_INCLUDED
