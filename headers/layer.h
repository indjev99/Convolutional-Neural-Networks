#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include <vector>

class Layer
{
protected:
    const std::vector<double>& prevValues;
    std::vector<double> values;
    std::vector<double> biases; ///TO-DO further separation because of ActivationLayer
    std::vector<double> weigths;
    std::vector<double> dCostdBiases;
    std::vector<double> dCostdWeigths;
public:
    Layer(const std::vector<double>& prevValues);
    virtual ~Layer();
    const std::vector<double>& get_values() const;
    virtual void calc_values() =0;
    virtual void accumulate_training(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues) =0;
    void apply_training(double lambda);
};

#endif // LAYER_H_INCLUDED
