#include "../headers/leaky_relu_activation_function.h"

LeakyReluActivationFunction leakyReluActivationFunction;

double LeakyReluActivationFunction::evaluate(double x) const
{
    if (x>0) return x;
    else return x*0.01;
}
double LeakyReluActivationFunction::evaluate_derivative(double x) const
{
    if (x>0) return 1;
    else return 0.01;
}
