#ifndef LOGISTIC_ACTIVATION_FUNCTION_H_INCLUDED
#define LOGISTIC_ACTIVATION_FUNCTION_H_INCLUDED

#include "activation_function.h"

class LogisticActivationFunction : public ActivationFunction
{
public:
    double evaluate(double x) const;
    double evaluate_derivative(double x) const;
} logisticActivationFunction;

#endif // LOGISTIC_ACTIVATION_FUNCTION_H_INCLUDED
