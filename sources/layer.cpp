#include "../headers/layer.h"

#include <math.h>

#include <iostream>
using namespace std;

Layer::Layer(int layerSize, const std::vector<double>& prevValues, const ActivationFunction& activationFunction, std::default_random_engine& generator)
    : layerSize{layerSize}
    , prevLayerSize{prevValues.size()}
    , prevValues{prevValues}
    , activationFunction{activationFunction}
    , rawValues(layerSize)
    , values(layerSize)
    , biases(layerSize)
    , weigths(layerSize*prevLayerSize)
    , dCostdBiases(layerSize,0)
    , dCostdWeigths(layerSize*prevLayerSize,0)
{
    std::normal_distribution<double> distribution(0,sqrt(2.0/prevLayerSize));
    for (int i=0;i<layerSize;++i)
    {
        biases[i]=0;
        for (int j=0;j<prevLayerSize;++j)
        {
            weigths[i*prevLayerSize+j]=distribution(generator);
            cerr<<weigths[i*prevLayerSize+j]<<" ";
        }
        cerr<<endl;
    }
    cerr<<endl;
}

const std::vector<double>& Layer::get_values() const
{
    return values;
}

void Layer::calc_values()
{
    for (int i=0;i<layerSize;++i)
    {
        rawValues[i]=biases[i];
        for (int j=0;j<prevLayerSize;++j)
        {
            rawValues[i]+=weigths[i*prevLayerSize+j]*prevValues[j];
        }
        values[i]=activationFunction.evaluate(rawValues[i]);
    }
}

void Layer::accumulate_training(const std::vector<double>& dCost0dValues, std::vector<double>& dCost0dPrevValues)
{
    double dCost0dRawValue;
    for (int i=0;i<prevLayerSize;++i)
    {
        dCost0dPrevValues[i]=0;
    }
    //cerr<<"biases: ";
    for (int i=0;i<layerSize;++i)
    {
        dCost0dRawValue=dCost0dValues[i]*activationFunction.evaluate_derivative(rawValues[i]);
        dCostdBiases[i]+=dCost0dRawValue;
        //cerr<<dCost0dRawValue<<" ";
    }
    //cerr<<"\n weights: ";
    for (int i=0;i<layerSize;++i)
    {
        dCost0dRawValue=dCost0dValues[i]*activationFunction.evaluate_derivative(rawValues[i]);
        for (int j=0;j<prevLayerSize;++j)
        {
            dCostdWeigths[i*prevLayerSize+j]+=dCost0dRawValue*prevValues[j];
            //cerr<<dCost0dRawValue*prevValues[j]<<" ";
            dCost0dPrevValues[j]+=dCost0dRawValue*dCostdWeigths[i*prevLayerSize+j];
        }
    }
    //cerr<<"\n";
}

void Layer::apply_training(double lambda)
{
    for (int i=0;i<layerSize;++i)
    {
        biases[i]-=dCostdBiases[i]*lambda;
        dCostdBiases[i]=0;
        for (int j=0;j<prevLayerSize;++j)
        {
            weigths[i*prevLayerSize+j]-=dCostdWeigths[i*prevLayerSize+j]*lambda;
            dCostdWeigths[i*prevLayerSize+j]=0;
        }
    }
}
