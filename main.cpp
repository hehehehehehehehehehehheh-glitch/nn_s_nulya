#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>
#include <sstream>


class ReLU{
public:
    static double activ(double x) {
        return (x > 0.0) ? x : 0.0;
    }

    static double der(double x) {
        return (x > 0.0) ? 1.0 : 0.0;
    }
    static Eigen::VectorXd activ(const Eigen::VectorXd& x) {
        return x.unaryExpr([](double v) { return activ(v); });
    }

    static Eigen::VectorXd der(const Eigen::VectorXd& x) {
        return x.unaryExpr([](double v) { return der(v); });
    }
};


class Layer{
    public: 
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Layer(int in_, int out_):A(in_, out_), b(out_){
        A = A.setRandom()*0.01;
        b.setZero();
    }
    Eigen::VectorXd output(Eigen::VectorXd x){
        return (x.transpose()*A).transpose() + b;
    }
    Eigen::MatrixXd method_1A(Eigen::VectorXd x, Eigen::RowVectorXd u){
        return x*(u.transpose().cwiseProduct(ReLU::der(output(x)))).transpose();
    }
    Eigen::VectorXd method_1b(Eigen::VectorXd x, Eigen::RowVectorXd u){
        return u.transpose().cwiseProduct(ReLU::der(output(x)));
    }
    Eigen::VectorXd method_2(Eigen::VectorXd x, Eigen::RowVectorXd u){
        return A*(u.transpose().cwiseProduct(ReLU::der(output(x))));
    }


};


class Loss_MSE{
double phi(Eigen::VectorXd w, Eigen::VectorXd y){
        return 0.5*(w - y).norm()*(w - y).norm();
}
Eigen::RowVectorXd u(Eigen::VectorXd w, Eigen::VectorXd y){
        return (w-y).transpose();
}
};


int main() {
    std::cout<<"куку"<<std::endl;
    std::ifstream file("/Users/sofia.shapovalova05icloud.ru/Downloads/Прак/nn_s_nulya/eigen-master/mnist_train.csv");
    std::string line;
    std::cout<<line<<std::endl;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        std::string chiselka;
        int y;
        std::getline(ss, chiselka, ',');
        y = std::stoi(chiselka);
        std::cout<<y<<std::endl;
        return 0;
        }

}