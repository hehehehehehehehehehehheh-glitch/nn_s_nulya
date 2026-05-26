#ifndef SAFE_HPP
#define SAFE_HPP

#pragma once
#include <Eigen/Dense>

template <typename T>
class Unsafe;

class Layer;

template <>
class Unsafe<Layer> {
private:
    Layer* obj_;
public:
    explicit Unsafe(Layer& obj) : obj_(&obj) {}
    void accumulate_weight_gradient_unchecked(const Eigen::VectorXd& x);
    void accumulate_bias_gradient_unchecked();
};

#endif // SAFE_HPP