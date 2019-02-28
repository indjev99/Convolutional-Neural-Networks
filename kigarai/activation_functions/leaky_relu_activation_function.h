#ifndef LEAKY_RELU_ACTIVATION_FUNCTION_H_INCLUDED
#define LEAKY_RELU_ACTIVATION_FUNCTION_H_INCLUDED

#include "activation_function.h"

class LeakyReluActivationFunction : public ActivationFunction
{
public:
    double evaluate(double x) const;
    double evaluate_derivative(double x) const;
} extern leakyReluActivationFunction;

#endif // LEAKY_RELU_ACTIVATION_FUNCTION_H_INCLUDED
