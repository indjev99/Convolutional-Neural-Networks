#include "../headers/layer.h"

Layer::Layer(unsigned int layerSize, const std::vector<double>& prevValues)
    : values(layerSize)
    , prevValues{prevValues} {}
Layer::~Layer() {}
const std::vector<double>& Layer::get_values() const
{
    return values;
}
