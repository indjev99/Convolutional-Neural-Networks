#ifndef RELU_ACTIVATION_FUNCTION_H_INCLUDED
#define RELU_ACTIVATION_FUNCTION_H_INCLUDED

#include "activation_function.h"

class ReluActivationFunction : public ActivationFunction
{
public:
    double evaluate(double x) const;
    double evaluate_derivative(double x) const;
} extern reluActivationFunction;

#endif // RELU_ACTIVATION_FUNCTION_H_INCLUDED
