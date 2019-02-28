#include "polling_layer.h"
#include <assert.h>

inline unsigned int encodeIndex(unsigned int d, unsigned int w, unsigned int h, const Structure& s)
{
    return d*s.heigth*s.width+h*s.width+w;
}
PollingLayer::PollingLayer(const Structure& field, const Structure& prevStructure, const std::vector<double>& prevValues)
    : Layer({prevStructure.depth,(prevStructure.heigth+field.heigth+1)/field.heigth,(prevStructure.width+field.width+1)/field.width},prevStructure,prevValues)
    , field{field}
    , maxSource(structure.size())
{
    assert(field.depth==1);
}
PollingLayer::~PollingLayer() {}
void PollingLayer::calc_values()
{
    for (int d=0;d<structure.depth;++d)
    {
        for (int h=0;h<structure.heigth;++h)
        {
            for (int w=0;w<structure.width;++w)
            {
                int i=encodeIndex(d,h,w,structure);
                int h2=h*field.heigth;
                int w2=w*field.width;
                for (int hOffset=0;hOffset<field.heigth && h2+hOffset<prevStructure.heigth;++hOffset)
                {
                    for (int wOffset=0;wOffset<field.heigth && w2+wOffset<prevStructure.width;++wOffset)
                    {
                        int j=encodeIndex(d,h2+hOffset,w2+wOffset,prevStructure);
                        if ((!hOffset && !wOffset) || prevValues[j]>values[i])
                        {
                            values[i]=prevValues[j];
                            maxSource[i]=j;
                        }
                    }
                }
            }
        }
    }
}
void PollingLayer::train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (unsigned int i=0;i<prevStructure.size();++i)
    {
        dCost0dPrevValues[i]=0;
    }
    for (unsigned int i=0;i<structure.size();++i)
    {
        dCost0dPrevValues[maxSource[i]]=dCost0dValues[i];
    }
}
void PollingLayer::apply_training(double lambda) {}
