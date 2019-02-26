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

    Network nn(3,{4,4,2},leakyReluActivationFunction,generator);
    nn.set_eta(0.05);

    int a,b,c,d,e;
    int cnt=0;
    while(1)
    {
        a=distribution(generator);
        b=distribution(generator);
        c=distribution(generator);
        d=(a && b) || (!a && !c);
        e=(a != b) && (c == b);
        auto out=nn.get_output({a,b,c});
        std::cout<<"iteration: "<<cnt<<" ans: "<<d<<" "<<e<<" prediction: "<<out[0]<<" "<<out[1]<<" cost: "<<nn.accumulate_training({d,e})<<"\n\n";
        ++cnt;
        if (cnt%5==0)
        {
            std::cout<<"Apply training."<<"\n\n";
            nn.apply_training();
        }
    }
    return 0;
}
