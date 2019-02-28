#include "trainer.h"
#include <algorithm>
#include <assert.h>

#include<iostream>
using namespace std;

Trainer::Trainer(Network& network, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs)
    : network{network}
    , sampleNumber{inputs.size()}
    , inputs{inputs}
    , outputs{outputs}
    , permutation(sampleNumber)
{
    assert(outputs.size()==sampleNumber);
    for (int i=0;i<sampleNumber;++i)
    {
        permutation[i]=i;
    }
}
double Trainer::train_epoch()
{
    random_shuffle(permutation.begin(),permutation.end());
    double cost=0;
    for (int i=0;i<sampleNumber;++i)
    {
        int j=permutation[i];
        network.get_output(inputs[j]);
        cost+=network.train(outputs[j]);
        if ((i+1)%1000==0) cerr<<" "<<i+1<<endl;
    }
    return cost/sampleNumber;
}
