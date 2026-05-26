#ifndef ANY_LAYER_HPP 
#define ANY_LAYER_HPP

#pragma once
#include "erase.hpp"
#include "layer.hpp"
#include <memory>
#include <utility>

template <typename TData>
class AnyLayerImpl : public Keeper<TData> {
public:
    using Keeper<TData>::Keeper;
    
    TData& get_data() {
        return this->object();
    }
    const TData& get_data() const {
        return this->object();
    }
    
    Eigen::VectorXd forward(const Eigen::VectorXd& x) override {
        return this->object().forward(x);
    }
    
    Eigen::VectorXd backward(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) override {
        auto& layer = this->object();
        layer.dy_buffer = u.transpose();
        if (layer.activation == "relu") {
            layer.dy_buffer = layer.dy_buffer.cwiseProduct(layer.lastY.unaryExpr([](double v) { return (v > 0.0) ? 1.0 : 0.0; }));
        } else if (layer.activation == "sigmoid") {
            layer.dy_buffer = layer.dy_buffer.cwiseProduct(layer.lastY.unaryExpr([](double v) {
                double s = 1.0 / (1.0 + std::exp(-v));
                return s * (1.0 - s);
            }));
        }
        
        Unsafe<Layer> unsafe_layer(layer);
        unsafe_layer.accumulate_weight_gradient_unchecked(x);
        unsafe_layer.accumulate_bias_gradient_unchecked();
        
        return layer.A * layer.dy_buffer;
    }
    
    std::unique_ptr<Concept> make_copy_() const override {
        return std::make_unique<AnyLayerImpl<TData>>(this->object());
    }
};

class AnyLayer {
private:
    std::unique_ptr<Concept> model_;
public:
    AnyLayer() = default;
    
    template <typename T>
    AnyLayer(T&& object)
        : model_(std::make_unique<AnyLayerImpl<typename std::decay<T>::type>>(std::forward<T>(object))) {}
        
    AnyLayer(const AnyLayer& other)
        : model_(other.model_ ? other.model_->make_copy_() : nullptr) {}
        
    AnyLayer(AnyLayer&& other) noexcept = default;
    
    AnyLayer& operator=(const AnyLayer& other) {
        if (this != &other) {
            model_ = other.model_ ? other.model_->make_copy_() : nullptr;
        }
        return *this;
    }
    
    AnyLayer& operator=(AnyLayer&& other) noexcept = default;
    
    Eigen::VectorXd forward(const Eigen::VectorXd& x) {
        return model_->forward(x);
    }
    
    Eigen::VectorXd backward(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) {
        return model_->backward(x, u);
    }
    
    Layer* getLayer() {
        return &dynamic_cast<AnyLayerImpl<Layer>*>(model_.get())->get_data();
    }
};

#endif