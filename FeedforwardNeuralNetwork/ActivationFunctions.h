#pragma once
#include "stdafx.h"

/**
 * \brief ReLU
 * \param inputs 
 * \return 
 */
Eigen::VectorXd Relu(Eigen::VectorXd inputs);
/**
 * \brief Tanh
 * \param inputs 
 * \return 
 */
Eigen::VectorXd Tanh(Eigen::VectorXd inputs);
/**
 * \brief Sigmoid
 * \param inputs 
 * \return 
 */
Eigen::VectorXd sigmoid(Eigen::VectorXd inputs);
/**
 * \brief Softmax
 * \param inputs 
 * \return 
 */
Eigen::VectorXd softmax(Eigen::VectorXd inputs);

/**
 * \brief differential sigmoid
 * \param input 
 * \return 
 */
Eigen::VectorXd differential_sigmoid(Eigen::VectorXd input);
/**
 * \brief differential tanh
 * \param input 
 * \return 
 */
Eigen::VectorXd differential_tanh(Eigen::VectorXd input);
/**
 * \brief differential ReLU
 * \param input 
 * \return 
 */
Eigen::VectorXd differential_relu(Eigen::VectorXd input);
