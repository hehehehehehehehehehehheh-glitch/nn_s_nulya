#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "sequantial.hpp"

std::vector<std::pair<Eigen::VectorXd, int>> Load_data(const std::string& filename, int max_samples = 1000) {
    std::vector<std::pair<Eigen::VectorXd, int>> data;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Can't open file: " << filename << std::endl;
        return data;
    }
    int count_dataset = 0;
    while (std::getline(file, line) && count_dataset < max_samples) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> pixels;
        std::getline(ss, token, ',');
        
        if (token.empty()) continue;
        int label = std::stoi(token);
        
        while (std::getline(ss, token, ',')) {
            pixels.push_back(std::stod(token) / 255.0);
        }
        if (pixels.size() == 784) {
            Eigen::VectorXd vec(784);
            for (size_t i = 0; i < 784; ++i) {
                vec(i) = pixels[i];
            }
            data.emplace_back(vec, label);
            count_dataset++;
        }
    }
    file.close();
    std::cout << count_dataset << " samples loaded successfully." << std::endl;
    return data;
}

int main() {
    auto train_data = Load_data("mnist_train.csv", 5000);
    auto test_data = Load_data("mnist_test.csv", 200);

    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;
    const double learning_rate = 0.005;
    const int epochs = 10;
    const int batch_size = 32;

    Sequential model;
    model.add_layer(Layer(input_size, hidden_size, "relu"));
    model.add_layer(Layer(hidden_size, output_size, "sigmoid"));

    Loss_MSE loss_calculator;

    AdamOptimizer optimizer(0.9, 0.999, 1e-8);

    int step_counter = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        int correct = 0;
        int current_batch_count = 0;

        model.clear_gradients();

        for (size_t sample_idx = 0; sample_idx < train_data.size(); ++sample_idx) {
            const auto& sample = train_data[sample_idx];
            const Eigen::VectorXd& x = sample.first;
            int label = sample.second;

            Eigen::VectorXd target = Eigen::VectorXd::Zero(output_size);
            target(label) = 1.0;

            Eigen::VectorXd prediction = model.forward(x);
            total_loss += loss_calculator.phi(prediction, target);

            model.backward(prediction, target);
            current_batch_count++;

            if (current_batch_count == batch_size || sample_idx == train_data.size() - 1) {
                step_counter++;
                optimizer.step(model, learning_rate, step_counter, current_batch_count);
                current_batch_count = 0;
            }

            int predicted_label;
            prediction.maxCoeff(&predicted_label);
            if (predicted_label == label) {
                correct++;
            }
        }

        double avg_loss = total_loss / train_data.size();
        double accuracy = 100.0 * correct / train_data.size();

        double test_loss = 0.0;
        int test_correct = 0;
        
        for (const auto& sample : test_data) {
            const Eigen::VectorXd& x = sample.first;
            int label = sample.second;

            Eigen::VectorXd target = Eigen::VectorXd::Zero(output_size);
            target(label) = 1.0;
            Eigen::VectorXd prediction = model.forward(x);
            test_loss += loss_calculator.phi(prediction, target);
            int predicted_label;
            prediction.maxCoeff(&predicted_label);
            if (predicted_label == label) {
                test_correct++;
            }
        }
        
        double avg_test_loss = test_loss / test_data.size();
        double test_accuracy = 100.0 * test_correct / test_data.size();

        std::cout << "Epoch " << (epoch + 1) << " | Loss: " << avg_loss << " | Accuracy: " << accuracy << "%" 
                  << " | Test Loss: " << avg_test_loss << " | Test Accuracy: " << test_accuracy << "%" << std::endl;
    }
    return 0;
}