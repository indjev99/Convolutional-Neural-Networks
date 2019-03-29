#include "../kigarai/kigarai.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <locale>

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

inline bool is_digit(char c)
{
    return c>='0' && c<='9';
}
int get_number(const string& word, int& pos)
{
    int a=0;
    while (!is_digit(word[pos]))
    {
        ++pos;
    }
    while (is_digit(word[pos]))
    {
        a=a*10+word[pos]-'0';
        ++pos;
    }
    return a;
}

void parse_training_data()
{
    string line;
    getline(trainingData,line);
    while (getline(trainingData,line))
    {
        bool first=true;
        int pos=0;
        trainingInputs.push_back({});
        while(pos<line.size())
        {
            if (first)
            {
                trainingLabels.push_back(get_number(line,pos));
                first=false;
            }
            else
            {
                trainingInputs[trainingInputs.size()-1].push_back(get_number(line,pos)/255.0);
            }
        }
    }
}
void parse_test_data()
{
    string line;
    getline(testData,line);
    while (getline(testData,line))
    {
        int pos=0;
        testInputs.push_back({});
        while(pos<line.size())
        {
            testInputs[testInputs.size()-1].push_back(get_number(line,pos)/255.0);
        }
    }
}
void parse_data()
{
    parse_training_data();
    cerr<<"Parsed training data."<<endl;
    parse_test_data();
    cerr<<"Parsed test data."<<endl;
    vector<double> curr(10,0);
    for (int i=0;i<trainingLabels.size();++i)
    {
        curr[trainingLabels[i]]=1;
        trainingOutputs.push_back(curr);
        curr[trainingLabels[i]]=0;
    }
    cerr<<"Network: "<<trainingInputs[0].size()<<" -> "<<trainingOutputs[0].size()<<endl;
}
void print_predictions()
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
void train_on_data()
{
    nn.add_convolution_layer(10,{4,4},2,true);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_polling_layer({2,2});

    nn.add_convolution_layer(10,{4,4},2,true);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_polling_layer({2,2});

    nn.add_convolution_layer(15,{4,4},2,false);
    nn.add_activation_layer(reluActivationFunction);
    nn.add_polling_layer({2,2});

    nn.add_fully_connected_layer(75,2);
    nn.add_activation_layer(reluActivationFunction);

    nn.add_fully_connected_layer(40,2);
    nn.add_activation_layer(reluActivationFunction);

    nn.add_fully_connected_layer(10,2);
    nn.add_activation_layer(reluActivationFunction);

    nn.set_batch_size(100);
    Trainer trainer(nn,trainingInputs,trainingOutputs);
    for (int i=0;i<10;++i)
    {
        if (i==0) nn.set_learning_rate(0.02);
        else if (i==2) nn.set_learning_rate(0.01);
        else if (i==4) nn.set_learning_rate(0.005);
        cout<<"Epoch: "<<i+1<<" with cost: "<<trainer.train_epoch()<<endl;
    }
}
void make_predictions()
{
    for (int i=0;i<testInputs.size();++i)
    {
        testOutputs.push_back(nn.get_output(testInputs[i]));
    }
}
int main()
{
    parse_data();
    train_on_data();
    make_predictions();
    print_predictions();
    return 0;
}
