#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#pragma once
#include "anylayer.hpp"
#include "loss.hpp"
#include <vector>

class Sequential {
public:
    std::vector<AnyLayer> layers;

    void add_layer(const Layer& layer) {
        layers.push_back(AnyLayer(layer));
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& x) {
        Eigen::VectorXd out = x;
        for (auto& l : layers) {
            out = l.forward(out);
        }
        return out;
    }

    void backward(const Eigen::VectorXd& pred, const Eigen::VectorXd& target) {
        Eigen::RowVectorXd u = Loss_MSE::gradient(pred, target);

        for (int i = layers.size() - 1; i >= 0; --i) {
            auto* lyr = layers[i].getLayer();
            u = layers[i].backward(lyr->lastX, u).transpose();
        }
    }

    void clear_gradients() {
        for (auto& l : layers) {
            l.getLayer()->clear_gradients();
        }
    }
};
#endif