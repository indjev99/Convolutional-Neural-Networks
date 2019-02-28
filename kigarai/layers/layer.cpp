#include "layer.h"

Layer::Layer(const Structure& structure, const Structure& prevStructure, const std::vector<double>& prevValues)
    : structure(structure)
    , values(structure.depth*structure.heigth*structure.width)
    , prevStructure(prevStructure)
    , prevValues{prevValues} {}
Layer::~Layer() {}
const std::vector<double>& Layer::get_values() const
{
    return values;
}
const Structure& Layer::get_structure() const
{
    return structure;
}
