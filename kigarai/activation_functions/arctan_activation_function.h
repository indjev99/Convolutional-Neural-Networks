#ifndef ARCTAN_ACTIVATION_FUNCTION_H_INCLUDED
#define ARCTAN_ACTIVATION_FUNCTION_H_INCLUDED

#include "activation_function.h"

class ArctanActivationFunction : public ActivationFunction
{
public:
    double evaluate(double x) const;
    double evaluate_derivative(double x) const;
} extern arctanActivationFunction;

#endif // ARCTAN_ACTIVATION_FUNCTION_H_INCLUDED
