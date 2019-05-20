#pragma once
#include "Eigen/Core"
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <ppl.h>
#include <tuple>
#include "config.h"

class NeuralNetwork
{
public:
    std::vector<int> structure;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    auto dataSet;
    NeuralNetwork(std::vector<int> const& structure, double inital_val, auto const& dataSet);
    std::tuple<double, int> learn(std::string filename);
    void softmax_test(std::ostream & out = std::cout);
    Eigen::MatrixXd calc_delta(int layerNo, std::vector<Eigen::VectorXd> const & output, Eigen::MatrixXd const & prevDelta);
    void back_propagation(std::vector<Eigen::VectorXd> const & output, Eigen::VectorXd const & teachData);
    double validate();
    double learn_proccess(Eigen::VectorXd const & input, Eigen::VectorXd const & teachData, std::ostream & out = std::cout);
    void pretrain(std::ostream & out = std::cout);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

