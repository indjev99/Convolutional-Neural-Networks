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
    /*if (prevLayerSize==2)
    {
        biases[0]=-1;
        weigths[0]=1;
        weigths[1]=1;
    }
    else
    {
        biases[0]=0;
        weigths[0]=distribution(generator);
    }*/
    for (int i=0;i<layerSize;++i)
    {
        biases[i]=1;
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
        //cerr<<"value = "<<biases[i];
        for (int j=0;j<prevLayerSize;++j)
        {
            rawValues[i]+=weigths[i*prevLayerSize+j]*prevValues[j];
            //cerr<<" + "<<weigths[i*prevLayerSize+j]<<" * "<<prevValues[j];
        }
        values[i]=activationFunction.evaluate(rawValues[i]);
        //cerr<<"\nvalue = "<<"act( "<<rawValues[i]<<" ) = "<<values[i]<<"\n";c
        //cerr<<" = "<<values[i]<<"\n";
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
    //cerr<<"\nweights: ";
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
