#include "arctan_activation_function.h"
#include <math.h>

ArctanActivationFunction arctanActivationFunction;

double ArctanActivationFunction::evaluate(double x) const
{
    return atan(x);
}
double ArctanActivationFunction::evaluate_derivative(double x) const
{
    return 1/(1+x*x);
}
