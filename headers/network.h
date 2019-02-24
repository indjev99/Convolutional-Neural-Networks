#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

class Network
{
private:
    class Layer;
    int networkSize;
    int* topology;
    int* currentInput;
    Layer* layers;
    int batchCnt;
public:
    Network(networkSize, const int* topology);
    ~Network();
    const double* getOutput(const double* input);
    void train(const double* targetOutput);
    void applyTraining();
};

#endif // NETWORK_H_INCLUDED
