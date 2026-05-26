#ifndef ERASE_HPP
#define ERASE_HPP

#pragma once
#include <memory>
#include <utility>
#include <Eigen/Dense>

class IEmpty {
protected:
    virtual ~IEmpty() = default;
};

template <class Empty>
class IAnyLayer : public Empty {
public:
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& x) = 0;
    virtual Eigen::VectorXd backward(const Eigen::VectorXd& x, const Eigen::RowVectorXd& u) = 0;
};

class Concept : public IAnyLayer<IEmpty> {
public:
    virtual std::unique_ptr<Concept> make_copy_() const = 0;
};

template <class TData>
class Keeper : public Concept {
protected:
    TData& object() { return data_; }
    const TData& object() const { return data_; }
private:
    TData data_;
public:
    template <typename U>
    Keeper(U&& data) : data_(std::forward<U>(data)) {}
};

#endif