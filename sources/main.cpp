#include "../headers/network.h"
#include "../headers/logistic_activation_function.h"
#include "../headers/relu_activation_function.h"
#include "../headers/leaky_relu_activation_function.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <random>

int main()
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,1);

    Network nn(1);
    nn.add_fully_connected_layer(10,2);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_fully_connected_layer(3,2);
    nn.set_learning_rate(0.005);
    nn.set_batch_size(20);

    double a,b,c,d;
    int cnt=0;
    while(1)
    {
        a=distribution(generator);
        b=distribution(generator);
        c=distribution(generator);
        d=a*4+b*2+c;
        auto out=nn.get_output({d});
        std::cout<<"iteration: "<<cnt<<" ans: "<<d<<" prediction: "<<out[0]*4+out[1]*2+out[2]<<" cost: "<<nn.train({a,b,c})<<"\n\n";
        ++cnt;
    }
    return 0;
}
