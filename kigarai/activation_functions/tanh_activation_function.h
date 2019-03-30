#ifndef TANH_ACTIVATION_FUNCTION_H_INCLUDED
#define TANH_ACTIVATION_FUNCTION_H_INCLUDED

#include "activation_function.h"

class TanhActivationFunction : public ActivationFunction
{
public:
    double evaluate(double x) const;
    double evaluate_derivative(double x) const;
} extern tanhActivationFunction;

#endif // TANH_ACTIVATION_FUNCTION_H_INCLUDED
