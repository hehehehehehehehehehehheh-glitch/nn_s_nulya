#ifndef LAYER_HPP
#define LAYER_HPP

#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

struct ReLU {
    static double activ(double x) { return (x > 0.0) ? x : 0.0; }
    static Eigen::VectorXd activ(const Eigen::VectorXd& x) {
        return x.unaryExpr([](double v) { return activ(v); });
    }
};

struct Sigmoid {
    static double activ(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static Eigen::VectorXd activ(const Eigen::VectorXd& x) {
        return x.unaryExpr([](double v) { return activ(v); });
    }
};

class Layer {
public:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::string activation;

    Eigen::MatrixXd gradA;
    Eigen::VectorXd gradb;
    Eigen::VectorXd dy_buffer;

    Eigen::VectorXd lastX;
    Eigen::VectorXd lastY;

    Eigen::MatrixXd m_A, v_A;
    Eigen::VectorXd m_b, v_b;

    Eigen::MatrixXd vVel_A;
    Eigen::VectorXd vVel_b;

    Layer(int in_, int out_, std::string act = "relu") : activation(act) {
        double scale = (act == "relu") ? std::sqrt(2.0 / in_) : std::sqrt(1.0 / in_);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, scale);

        A = Eigen::MatrixXd::NullaryExpr(in_, out_, [&]() { return d(gen); });
        b = Eigen::VectorXd::Zero(out_);

        gradA = Eigen::MatrixXd::Zero(in_, out_);
        gradb = Eigen::VectorXd::Zero(out_);
        dy_buffer = Eigen::VectorXd::Zero(out_);

        m_A = Eigen::MatrixXd::Zero(in_, out_);
        v_A = Eigen::MatrixXd::Zero(in_, out_);
        m_b = Eigen::VectorXd::Zero(out_);
        v_b = Eigen::VectorXd::Zero(out_);

        vVel_A = Eigen::MatrixXd::Zero(in_, out_);
        vVel_b = Eigen::VectorXd::Zero(out_);
    }

    void clear_gradients() {
        gradA.setZero();
        gradb.setZero();
    }

    Eigen::VectorXd output(const Eigen::VectorXd& x) {
        return (x.transpose() * A).transpose() + b;
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& x) {
        lastX = x;
        lastY = output(x);
        
        if (activation == "relu") {
            return ReLU::activ(lastY);
        } else if (activation == "sigmoid") {
            return Sigmoid::activ(lastY);
        }
        return lastY;
    }
};

template <typename T>
class Unsafe;

template <>
class Unsafe<Layer> {
private:
    Layer* obj_;
public:
    explicit Unsafe(Layer& obj) : obj_(&obj) {}
    
    void accumulate_weight_gradient_unchecked(const Eigen::VectorXd& x) {
        obj_->gradA.noalias() += x * obj_->dy_buffer.transpose();
    }
    
    void accumulate_bias_gradient_unchecked() {
        obj_->gradb.noalias() += obj_->dy_buffer;
    }
};
#endif