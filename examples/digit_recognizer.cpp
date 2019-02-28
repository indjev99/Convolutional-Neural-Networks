#include "../kigarai/kigarai.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

ifstream trainingData("data/train.csv");
ifstream testData("data/test.csv");
ofstream testPrediction("data/submission.csv");

vector<vector<double>> trainingInputs;
vector<vector<double>> trainingOutputs;
vector<int> trainingLabels;
vector<vector<double>> testInputs;
vector<vector<double>> testOutputs;
vector<int> testLabels;

void parseTrainingData()
{
    string line;
    getline(trainingData,line);
    int cnt=0;
    while (getline(trainingData,line))
    {
        istringstream lineStream(line);
        string elem;
        bool first=true;
        trainingInputs.push_back({});
        while(getline(lineStream,elem,','))
        {
            istringstream elemStream(elem);
            if (first)
            {
                int label;
                elemStream>>label;
                trainingLabels.push_back(label);
                first=false;
            }
            else
            {
                double val;
                elemStream>>val;
                trainingInputs[trainingInputs.size()-1].push_back(val/255.0);
            }
        }
        ++cnt;
        if (cnt%1000==0) cout<<"Training data: "<<cnt<<endl;;
        //if (cnt==2000) break;
    }
}
void parseTestData()
{
    string line;
    getline(testData,line);
    int cnt=0;
    while (getline(testData,line))
    {
        istringstream lineStream(line);
        string elem;
        testInputs.push_back({});
        while(getline(lineStream,elem,','))
        {
            istringstream elemStream(elem);
            double val;
            elemStream>>val;
            testInputs[testInputs.size()-1].push_back(val/255.0);
        }
        ++cnt;
        if (cnt%1000==0) cout<<"Test data: "<<cnt<<endl;
        //if (cnt==1000) break;
    }
}
void parseData()
{
    parseTrainingData();
    parseTestData();
    vector<double> curr(10,0);
    for (int i=0;i<trainingLabels.size();++i)
    {
        curr[trainingLabels[i]]=1;
        trainingOutputs.push_back(curr);
        curr[trainingLabels[i]]=0;
    }
    cerr<<trainingInputs[0].size()<<" -> "<<trainingOutputs[0].size()<<endl;
}
void printPredictions()
{
    vector<double> curr;
    double maxAct;
    int maxActLabel;
    for (int i=0;i<testOutputs.size();++i)
    {
        curr=testOutputs[i];
        maxAct=curr[0];
        maxActLabel=0;
        for (int i=1;i<10;++i)
        {
            if (curr[i]>maxAct)
            {
                maxAct=curr[i];
                maxActLabel=i;
            }
        }
        testLabels.push_back(maxActLabel);
    }
    testPrediction<<"ImageId,Label\n";
    for (int i=0;i<testLabels.size();++i)
    {
        testPrediction<<i+1<<','<<testLabels[i]<<'\n';
    }
    for (int i=testLabels.size();i<28000;++i)
    {
        testPrediction<<i+1<<','<<0<<'\n';
    }
}
Network nn({1,28,28},1337);
void trainOnData()
{
    nn.add_convolution_layer(10,{5,5},2);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_polling_layer({2,2});
    nn.add_convolution_layer(10,{5,5},2);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_polling_layer({2,2});
    nn.add_fully_connected_layer(75,2);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_fully_connected_layer(35,2);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_fully_connected_layer(10,2);
    nn.set_learning_rate(0.01);
    nn.set_batch_size(100);
    Trainer trainer(nn,trainingInputs,trainingOutputs);
    for (int i=0;i<5;++i)
    {
        if (i==0) nn.set_learning_rate(0.01);
        else if (i==1) nn.set_learning_rate(0.0033);
        else if (i==2) nn.set_learning_rate(0.001);
        else if (i==3) nn.set_learning_rate(0.00033);
        else if (i==4) nn.set_learning_rate(0.00001);
        cout<<"Epoch: "<<i+1<<" with cost: "<<trainer.train_epoch()<<endl;
    }
}
void makePredictions()
{
    for (int i=0;i<testInputs.size();++i)
    {
        testOutputs.push_back(nn.get_output(testInputs[i]));
    }
}
int main()
{
    parseData();
    trainOnData();
    makePredictions();
    printPredictions();
    return 0;
}
