#pragma once
#include "stdafx.h"
#include <vector>
#include "MNISTDataSet.h"
#include "XORDataSet.h"
#include "FuncApproxDataSet.h"
#include "TwoSpiralDataSet.h"

#define SLIDE 10
#define PRETRAIN_ERROR_BOTTOM 0.0001
#define PRETRAIN_LEARNING_TIME 1000000

//XOR
//std::vector<double> initVals = {0.001, 0.01, 0.1};
const std::vector<double> initVals = {0.001};
#define TRIALS_PER_STRUCTURE 5
#define LEARNING_RATE 1.0
#define LEARNING_TIME 50000
#define ERROR_BOTTOM 0.01
//dataset
XORDataSet dataSet;
//Network structure.
//const std::vector<std::vector<int>> structures = {{2, 2, 1}, {2, 3, 1}, {2, 4, 1}};
const std::vector<std::vector<int>> structures = {{2, 2, 1},{2, 3, 1},{2, 4, 1},{ 2, 2, 2, 1 },{ 2, 3, 3, 1 },{ 2, 4, 4, 1 }, {2, 2, 2, 2, 1},{2, 3, 3, 3, 1},{2, 4, 4, 4, 1},{2, 3, 3, 3, 3, 1},{2, 4, 4, 4, 4, 1}};
//const std::vector<std::vector<int>> structures = {{2, 3, 3, 1}, {2, 3, 3, 3, 1}, {2, 4, 4, 4, 1}, {2, 3, 3, 3, 3, 1}};

////Function approximation
//const std::vector<double> initVals = {0.01};
//#define TRIALS_PER_STRUCTURE 1
//#define LEARNING_RATE 0.05
//#define LEARNING_TIME 10000
//#define ERROR_BOTTOM 0.00000001
////dataset
//FuncApproxDataSet dataSet;
////Network structure.
//const std::vector<std::vector<int>> structures = {{1, 5, 1}};
////const std::vector<std::vector<int>> structures = {{1, 2, 1}, {1, 3, 1}, {1, 4, 1}};

////MNIST
//const std::vector<double> initVals = {1.0};
//#define TRIALS_PER_STRUCTURE 1
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 1
//#define ERROR_BOTTOM 0.01
//MNISTDataSet dataSet;
////Network structure.
//const std::vector<std::vector<int>> structures = {{784, 10}};

////TwoSpiral Prpblem
//const std::vector<double> initVals = {0.1};
//#define TRIALS_PER_STRUCTURE 5
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 10
//#define ERROR_BOTTOM 0.01
//TwoSpiralDataSet dataSet;
//const std::vector<std::vector<int>> structures = {{2, 2, 1}, {2, 4, 1}, {2, 6, 1}, {2, 2, 2, 1}, {2, 2, 4, 1}, {2, 4, 2, 1}, {2, 2, 2, 2, 1}};

inline Eigen::VectorXd activationFunc(Eigen::VectorXd const& inputs)
{
	Eigen::VectorXd result = sigmoid(inputs);
	return result;
}

inline Eigen::VectorXd differential(Eigen::VectorXd const& input)
{
	Eigen::VectorXd result = differentialSigmoid(input);
	return result;
}

inline Eigen::VectorXd outputActivationFunc(Eigen::VectorXd const& inputs)
{
	if (dataSet.useSoftmax)
	{
		return softmax(inputs);
	}
	//	return inputs;
	return activationFunc(inputs);
}

inline Eigen::VectorXd outputDifferential(Eigen::VectorXd const& input)
{
	if (dataSet.useSoftmax)
	{
		return Eigen::VectorXd::Ones(input.size());
	}
	//	return Eigen::VectorXd::Ones(input.size());
	return differential(input);
}
