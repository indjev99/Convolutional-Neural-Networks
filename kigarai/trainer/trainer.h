#ifndef TRAINER_H_INCLUDED
#define TRAINER_H_INCLUDED

#include "../network/network.h"
#include <vector>

class Trainer
{
private:
    Network& network;
    const unsigned int sampleNumber;
    const std::vector<std::vector<double>>& inputs;
    const std::vector<std::vector<double>>& outputs;
    std::vector<unsigned int> permutation;
    std::default_random_engine generator;
public:
    Trainer(Network& network, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs);
    double train_epoch();
};

#endif // TRAINER_H_INCLUDED
