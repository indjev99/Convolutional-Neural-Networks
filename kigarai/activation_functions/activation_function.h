#ifndef ACTIVATION_FUNCTION_H_INCLUDED
#define ACTIVATION_FUNCTION_H_INCLUDED

class ActivationFunction
{
public:
    virtual double evaluate(double x) const =0;
    virtual double evaluate_derivative(double x) const =0;
};

#endif // ACTIVATION_FUNCTION_H_INCLUDED
