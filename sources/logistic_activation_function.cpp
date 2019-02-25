#include "../headers/logistic_activation_function.h"
#include <math.h>

LogisticActivationFunction logisticActivationFunction;

double LogisticActivationFunction::evaluate(double x) const
{
    double ex=exp(x);
    return ex/(1+ex);
}
double LogisticActivationFunction::evaluate_derivative(double x) const
{
    double ex=exp(x);
    return ex/((1+ex)*(1+ex));
}
