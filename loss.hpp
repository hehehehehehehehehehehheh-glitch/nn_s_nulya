#ifndef LOSS_HPP
#define LOSS_HPP

#include <iostream>
#include "eigen-master/Eigen/Dense"
#include <random>
#include <fstream>
#include <sstream>
#include <vector>

class Loss_MSE{
public:
double phi(Eigen::VectorXd w, Eigen::VectorXd y){
        return 0.5*(w - y).norm()*(w - y).norm();
}
static Eigen::RowVectorXd gradient(Eigen::VectorXd w, Eigen::VectorXd y){
        return (w-y).transpose();
}
};
#endif