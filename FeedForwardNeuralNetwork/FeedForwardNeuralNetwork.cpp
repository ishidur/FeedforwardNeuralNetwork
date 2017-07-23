// FeedForwardNeuralNetwork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <array>
#include <vector>
#include "Eigen/Core"
#include <iostream>

#define LEARNING_RATE 0.5
using namespace std;
using namespace Eigen;

//XOR data
//(x, y, output)
vector<array<int, 3>> dataSet = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};

//Network structure
array<int, 3> structure = {2, 3, 1};
vector<MatrixXd> weights;
vector<VectorXd> biases;

void initWeightsAndBiases()
{
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(MatrixXd::Constant(structure[i], structure[i - 1], 0.5));
		biases.push_back(VectorXd::Constant(structure[i], 0.5));
	}
}

double sigmoid(double input)
{
	return 1.0 / (1 + exp(-input));
}

auto activationFunc = [](const double input)
{
	return sigmoid(input);
};

auto squared = [](const double x)
{
	return x * x;
};

VectorXd errorFunc(VectorXd outData, VectorXd teachData)
{
	//	Mean Square Error
	VectorXd error = (teachData - outData).unaryExpr(squared);
	error *= 1.0 / 2.0;
	return error;
}

VectorXd learnProccess(VectorXd input, VectorXd teachData)
{
	//	feedforward proccess
	VectorXd output[structure.size()];
	output[0] = input;
	for (int i = 0; i < structure.size() - 1; i++)
	{
		output[i + 1] = (weights[i] * output[i] + biases[i]).unaryExpr(activationFunc);
	}
	VectorXd delta = errorFunc(output[structure.size() - 1], teachData);
	cout << errorFunc(output[structure.size() - 1], teachData) << endl;

	//	backpropergation method


	return output[structure.size() - 1];
}

int main()
{
	initWeightsAndBiases();
	cout << sigmoid(1.0) << endl;
	for (int i = 0; i < 4; ++i)
	{
		VectorXd input(2);
		input << dataSet[i][0] , dataSet[i][1];
		cout << input << endl << endl;

		VectorXd teach(1);
		teach << dataSet[i][2];

		VectorXd out = learnProccess(input, teach);
		cout << out << endl;
		cout << "finish" << endl;
	}
	return 0;
}
