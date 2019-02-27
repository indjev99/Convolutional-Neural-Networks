#include "../headers/network.h"
#include "../headers/logistic_activation_function.h"
#include "../headers/relu_activation_function.h"
#include "../headers/leaky_relu_activation_function.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <random>

int main()
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,1);

    Network nn(1,{8,8,3},reluActivationFunction,generator);
    nn.set_eta(0.005);

    int a,b,c,d;
    int cnt=0;
    while(1)
    {
        a=distribution(generator);
        b=distribution(generator);
        c=distribution(generator);
        d=a*4+b*2+c;
        auto out=nn.get_output({d});
        std::cout<<"iteration: "<<cnt<<" ans: "<<d<<" prediction: "<<out[0]*4+out[1]*2+out[2]<<" cost: "<<nn.accumulate_training({a,b,c})<<"\n\n";
        ++cnt;
        if (cnt%20==0)
        {
            std::cout<<"Apply training."<<"\n\n";
            nn.apply_training();
        }
    }
    return 0;
}
