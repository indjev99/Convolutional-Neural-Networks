#include "tanh_activation_function.h"
#include <math.h>

TanhActivationFunction tanhActivationFunction;

double TanhActivationFunction::evaluate(double x) const
{
    return tanh(x);
}
double TanhActivationFunction::evaluate_derivative(double x) const
{
    double th=tanh(x);
    return 1-th*th;
}
