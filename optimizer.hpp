#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#pragma once
#include "sequantial.hpp"
#include <cmath>

class AdamOptimizer {
private:
    double b1, b2, eps;
public:
    AdamOptimizer(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) 
        : b1(beta1), b2(beta2), eps(epsilon) {}

    void step(Sequential& model, double lr, int t, int batch_size) {
        double cor1 = 1.0 - std::pow(b1, t);
        double cor2 = 1.0 - std::pow(b2, t);

        for (auto& anyL : model.layers) {
            auto* l = anyL.getLayer();
            if (l) {
                Eigen::VectorXd g_b = l->gradb / batch_size;
                Eigen::MatrixXd g_A = l->gradA / batch_size;

                l->m_b = b1 * l->m_b + (1.0 - b1) * g_b;
                l->v_b = b2 * l->v_b + (1.0 - b2) * g_b.cwiseProduct(g_b);
                Eigen::VectorXd m_b_hat = l->m_b / cor1;
                Eigen::VectorXd v_b_hat = l->v_b / cor2;
                l->b -= lr * m_b_hat.binaryExpr(v_b_hat, [this](double m, double v) {
                    return m / (std::sqrt(v) + eps);
                });

                l->m_A = b1 * l->m_A + (1.0 - b1) * g_A;
                l->v_A = b2 * l->v_A + (1.0 - b2) * g_A.cwiseProduct(g_A);
                Eigen::MatrixXd m_A_hat = l->m_A / cor1;
                Eigen::MatrixXd v_A_hat = l->v_A / cor2;
                l->A -= lr * m_A_hat.binaryExpr(v_A_hat, [this](double m, double v) {
                    return m / (std::sqrt(v) + eps);
                });

                l->clear_gradients();
            }
        }
    }
};

#endif