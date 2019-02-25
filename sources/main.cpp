#include "../headers/network.h"
#include "../headers/logistic_activation_function.h"
#include "../headers/relu_activation_function.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <random>

int main()
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,1);

    const int nums=3;

    Network nn(nums,{3*nums,3*nums,nums+1},reluActivationFunction,generator);
    nn.set_eta(0.01);

    std::vector<double> in(nums);
    std::vector<double> ans(nums+1);
    int cnt=0;
    while(1)
    {
        int sum=0;
        for (int i=0;i<nums;++i)
        {
            in[i]=distribution(generator);
            sum+=in[i];
        }
        for (int i=0;i<nums+1;++i)
        {
            ans[i]=0;
        }
        ans[sum]=1;
        const std::vector<double>& out=nn.get_output(in);
        int prediction=0;
        double maxAct=out[0];
        for (int i=1;i<nums+1;++i)
        {
            if (out[i]>maxAct)
            {
                maxAct=out[i];
                prediction=i;
            }
        }
        std::cout<<"iteration: "<<cnt<<" ans: "<<sum<<" prediction: "<<prediction<<" cost: "<<nn.accumulate_training(ans)<<"\n\n";
        ++cnt;
        if (cnt%5==0) nn.apply_training();
    }
    return 0;
}