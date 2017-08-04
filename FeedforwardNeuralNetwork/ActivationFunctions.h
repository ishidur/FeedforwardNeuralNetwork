#pragma once
#include "stdafx.h"

Eigen::VectorXd Relu(Eigen::VectorXd inputs);
Eigen::VectorXd Tanh(Eigen::VectorXd inputs);
Eigen::VectorXd sigmoid(Eigen::VectorXd inputs);
Eigen::VectorXd softmax(Eigen::VectorXd inputs);

Eigen::VectorXd differentialSigmoid(Eigen::VectorXd input);
Eigen::VectorXd differentialTanh(Eigen::VectorXd input);
Eigen::VectorXd differentialRelu(Eigen::VectorXd input);