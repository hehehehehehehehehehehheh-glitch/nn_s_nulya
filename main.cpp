#include <iostream>
#include "eigen-master/Eigen/Dense"
#include <random>
#include <fstream>
#include <sstream>
#include <vector>


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
    Eigen::MatrixXd gradient_A(Eigen::VectorXd x, Eigen::RowVectorXd u){
        return x*(u.transpose().cwiseProduct(ReLU::der(output(x)))).transpose();
    }
    Eigen::VectorXd gradient_b(Eigen::VectorXd x, Eigen::RowVectorXd u){
        return u.transpose().cwiseProduct(ReLU::der(output(x)));
    }
    Eigen::VectorXd gradient_input(Eigen::VectorXd x, Eigen::RowVectorXd u){
        return A*(u.transpose().cwiseProduct(ReLU::der(output(x))));
    }


};


class Loss_MSE{
public:
double phi(Eigen::VectorXd w, Eigen::VectorXd y){
        return 0.5*(w - y).norm()*(w - y).norm();
}
Eigen::RowVectorXd gradient(Eigen::VectorXd w, Eigen::VectorXd y){
        return (w-y).transpose();
}
};

std::vector<std::pair<Eigen::VectorXd, int>> Load_data(const std::string& filename, int max_samples = 1000){
    std::vector<std::pair<Eigen::VectorXd, int>> data;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()){
        std::cerr << "Cant open "<<filename<<std::endl; //https://en.cppreference.com/w/cpp/io/cerr.html 
        return data;
    }
    int count_dataset = 0;
    while (std::getline(file, line)&&count_dataset<max_samples){
        std::stringstream ss(line);
        std::string token;
        std::vector<double> pixels;
        std::getline(ss, token, ',');
        int label = std::stoi(token);
        while (std::getline(ss, token, ',')){
            pixels.push_back(std::stod(token)/255.0);
        }
        if (pixels.size() == 784){
            Eigen::VectorXd vec(784);
            for (size_t i = 0; i<784; ++i) vec(i) = pixels[i];
            data.emplace_back(vec, label);
            count_dataset++;
        }
    }
    file.close();
    std::cout << count_dataset << "samples" <<std::endl;
    return data;
    }

int main() {
    std::cout << "куку" << std::endl;
    auto train_data = Load_data("/Users/sofia.shapovalova05icloud.ru/Downloads/Прак/nn_s_nulya/eigen-master/mnist_train.csv", 1000);
    if (train_data.empty()) return 1;
    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;
    const double learning_rate = 0.001;
    const int epochs = 40;

    Layer layer1(input_size, hidden_size);
    Layer layer2(hidden_size, output_size);
    Loss_MSE loss;

    for (int epoch = 0; epoch < epochs; ++epoch){
        double total_loss = 0.0;
        int correct = 0;
        for (const auto& sample : train_data){
            const Eigen::VectorXd x = sample.first;
            int label = sample.second;
            Eigen::VectorXd arr = Eigen::VectorXd::Zero(output_size);
            arr(label) = 1.0;

            Eigen::VectorXd output_layer1 = layer1.output(x);
            Eigen::VectorXd activate1_ReLU = ReLU::activ(output_layer1);
            Eigen::VectorXd output_layer2 = layer2.output(activate1_ReLU);
            Eigen::VectorXd activate2_ReLU = ReLU::activ(output_layer2);

            Eigen::VectorXd prediction = activate2_ReLU;
            total_loss+=loss.phi(prediction, arr);

            Eigen::VectorXd gradient_output_layer2 = loss.gradient(prediction, arr);

            Eigen::RowVectorXd u2 = gradient_output_layer2.transpose();
            Eigen::MatrixXd back_gradient_A_layer2 = layer2.gradient_A(activate1_ReLU, u2);
            Eigen::VectorXd back_gradient_b_layer2 = layer2.gradient_b(activate1_ReLU, u2);
            Eigen::VectorXd back_gradient_output_layer2 = layer2.gradient_input(activate1_ReLU, u2);
            
            Eigen::RowVectorXd u1 = back_gradient_output_layer2.transpose();
            Eigen::MatrixXd back_gradient_A_layer1 = layer1.gradient_A(x, u1);
            Eigen::VectorXd back_gradient_b_layer1 = layer1.gradient_b(x, u1);

            layer1.A -= learning_rate*back_gradient_A_layer1;
            layer1.b -= learning_rate*back_gradient_b_layer1;
            layer2.A -= learning_rate*back_gradient_A_layer2;
            layer2.b -= learning_rate*back_gradient_b_layer2;

        int predicted_label;
        prediction.maxCoeff(&predicted_label);
        if (predicted_label == label) correct++;
    }
    std::cout << "Epoch " << epoch+1<< ", loss: " << total_loss / train_data.size()<< ", accuracy: " << 100.0 * correct / train_data.size() << "%" << std::endl;
}
}
