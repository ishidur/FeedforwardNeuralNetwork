#pragma once
#include "stdafx.h"
#include <vector>
#include "ActivationFunctions.h"
#include "MNISTDataSet.h"
#include "XORDataSet.h"
#include "FuncApproxDataSet.h"
#include "TwoSpiralDataSet.h"

#define SLIDE 100

bool needPretrain = false;
#define PRETRAIN_ERROR_BOTTOM 0.001
#define PRETRAIN_LEARNING_TIME 1000
//#define PRETRAIN_LEARNING_TIME 100000000

//XOR
const std::vector<double> initVals = {0.001};
#define TRIALS_PER_STRUCTURE 5
#define LEARNING_RATE 0.5
#define LEARNING_TIME 10000
#define ERROR_BOTTOM -0.01
//dataset
XORDataSet dataSet;
//Network structure.
const std::vector<std::vector<int>> structures = {{2, 3, 1}};
//const std::vector<std::vector<int>> structures = {{2, 2, 1},{2, 3, 1},{2, 4, 1},{ 2, 2, 2, 1 },{ 2, 3, 3, 1 },{ 2, 4, 4, 1 }, {2, 2, 2, 2, 1},{2, 3, 3, 3, 1},{2, 4, 4, 4, 1}};

////Function approximation
//const std::vector<double> initVals = {0.01};
//#define TRIALS_PER_STRUCTURE 1
//#define LEARNING_RATE 0.03
//#define LEARNING_TIME 5000
//#define ERROR_BOTTOM 0.00000001
////dataset
//FuncApproxDataSet dataSet;
////Network structure.
//const std::vector<std::vector<int>> structures = {{1, 8, 1}};
////const std::vector<std::vector<int>> structures = {{1, 4, 4, 4, 1}};

////MNIST
//const std::vector<double> initVals = {1.0};
//#define TRIALS_PER_STRUCTURE 1
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 1
//#define ERROR_BOTTOM 0.01
//MNISTDataSet dataSet;
////Network structure.
//const std::vector<std::vector<int>> structures = {{784, 10}};
//
// //TwoSpiral Prpblem
// const std::vector<double> initVals = {1.0};
// #define TRIALS_PER_STRUCTURE 1
// #define LEARNING_RATE 0.05
// #define LEARNING_TIME 100000
// #define ERROR_BOTTOM 0.001
// TwoSpiralDataSet dataSet;
// const std::vector<std::vector<int>> structures = {
// 	{2, 5, 5, 5, 1}, {2, 10, 10, 10, 1}, {2, 15, 15, 15, 1}, {2, 20, 20, 20, 1}
// };
//const std::vector<std::vector<int>> structures = {{2, 20, 20, 20, 1}};

inline Eigen::VectorXd activationFunc(Eigen::VectorXd const& inputs)
{
	Eigen::VectorXd result = sigmoid(inputs);
	return result;
}

inline Eigen::VectorXd differential(Eigen::VectorXd const& input)
{
	Eigen::VectorXd result = differential_sigmoid(input);
	return result;
}

inline Eigen::VectorXd outputActivationFunc(Eigen::VectorXd const& inputs)
{
	if (dataSet.useSoftmax) { return softmax(inputs); }
	//	return inputs;
	return activationFunc(inputs);
}

inline Eigen::VectorXd outputDifferential(Eigen::VectorXd const& input)
{
	if (dataSet.useSoftmax) { return Eigen::VectorXd::Ones(input.size()); }
	//	return Eigen::VectorXd::Ones(input.size());
	return differential(input);
}
