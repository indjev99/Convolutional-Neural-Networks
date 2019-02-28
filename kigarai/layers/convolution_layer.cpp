#include "convolution_layer.h"

#include <math.h>

inline unsigned int encodeIndex(unsigned int d, unsigned int w, unsigned int h, const Structure& s)
{
    return d*s.heigth*s.width+h*s.width+w;
}
ConvolutionLayer::ConvolutionLayer(unsigned int depth, const Structure& convolutionStructure, const Structure& prevStructure, const std::vector<double>& prevValues, double varianceFactor, std::default_random_engine& generator)
    : Layer({depth,convolutionStructure,prevStructure},prevStructure,prevValues)
    , convolutionStructure{convolutionStructure}
    , biases(structure.depth,0)
    , weigths(structure.depth*prevStructure.depth*convolutionStructure.size(),0)
    , dCostdBiases(structure.depth,0)
    , dCostdWeigths(structure.depth*prevStructure.depth*convolutionStructure.size(),0)
{
    std::normal_distribution<double> distribution(0,sqrt(varianceFactor/(prevStructure.depth*convolutionStructure.size())));
    for (unsigned int i=0;i<weigths.size();++i)
    {
        weigths[i]=distribution(generator);
    }
}
ConvolutionLayer::~ConvolutionLayer() {};
void ConvolutionLayer::calc_values()
{
    for (unsigned int d=0;d<structure.depth;++d)
    {
        for (unsigned int h=0;h<structure.heigth;++h)
        {
            for (unsigned int w=0;w<structure.width;++w)
            {
                int i=encodeIndex(d,h,w,structure);
                values[i]=biases[d];
                for (unsigned int d2=0;d2<prevStructure.depth;++d2)
                {
                    for (unsigned int hOffset=0;hOffset<convolutionStructure.heigth;++hOffset)
                    {
                        for (unsigned int wOffset=0;wOffset<convolutionStructure.width;++wOffset)
                        {
                            int h2=h+hOffset;
                            int w2=w+wOffset;
                            int j=encodeIndex(d2,h2,w2,prevStructure);
                            int k=d*prevStructure.depth*convolutionStructure.size()+encodeIndex(d2,hOffset,wOffset,convolutionStructure);
                            values[i]+=weigths[k]*prevValues[j];
                        }
                    }
                }
            }
        }
    }
}
void ConvolutionLayer::train(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    for (unsigned int i=0;i<prevStructure.size();++i)
    {
        dCost0dPrevValues[i]=0;
    }
    for (unsigned int d=0;d<structure.depth;++d)
    {
        for (unsigned int h=0;h<structure.heigth;++h)
        {
            for (unsigned int w=0;w<structure.width;++w)
            {
                int i=encodeIndex(d,h,w,structure);
                dCostdBiases[d]+=dCost0dValues[i];
                for (unsigned int d2=0;d2<prevStructure.depth;++d2)
                {
                    for (unsigned int hOffset=0;hOffset<convolutionStructure.heigth;++hOffset)
                    {
                        for (unsigned int wOffset=0;wOffset<convolutionStructure.width;++wOffset)
                        {
                            int h2=h+hOffset;
                            int w2=w+wOffset;
                            int j=encodeIndex(d2,h2,w2,prevStructure);
                            int k=d*prevStructure.depth*convolutionStructure.size()+encodeIndex(d2,hOffset,wOffset,convolutionStructure);
                            dCostdWeigths[k]+=dCost0dValues[i]*prevValues[j];
                            dCost0dPrevValues[j]+=dCost0dValues[i]*weigths[k];
                        }
                    }
                }
            }
        }
    }
}
void ConvolutionLayer::apply_training(double lambda)
{
    for (unsigned int i=0;i<biases.size();++i)
    {
        biases[i]-=dCostdBiases[i]*lambda;
        dCostdBiases[i]=0;
    }
    for (unsigned int i=0;i<weigths.size();++i)
    {
        weigths[i]-=dCostdWeigths[i]*lambda;
        dCostdWeigths[i]=0;
    }
}
