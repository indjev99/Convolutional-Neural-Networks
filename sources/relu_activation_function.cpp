#include "../headers/relu_activation_function.h"
#include <math.h>

ReluActivationFunction reluActivationFunction;

double ReluActivationFunction::evaluate(double x) const
{
    if (x>0) return x;
    else return 0;
}
double ReluActivationFunction::evaluate_derivative(double x) const
{
    if (x>0) return 1;
    else return 0;
}
