// FeedForwardNeuralNetwork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <array>
#include <vector>
#include "Eigen/Core"
#include <iostream>
#include <random>

#define LEARNING_RATE 0.5
#define LEARNING_TIME 4
using namespace std;
using namespace Eigen;

//XOR data
//(x, y)
MatrixXd dataSet(4, 2);
VectorXd teachSet(4);

//Network structure. make sure layer number is equal to array number.
array<int, 4> structure = {2, 3, 2, 1};
vector<MatrixXd> weights;
vector<VectorXd> biases;

void initWeightsAndBiases()
{
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(MatrixXd::Constant(structure[i - 1], structure[i], 0.5));
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

MatrixXd calcDelta(int layerNo, VectorXd output[structure.size()], MatrixXd prevDelta)
{
	VectorXd differential = output[layerNo + 1].array() * (VectorXd::Ones(structure[layerNo + 1]) - output[layerNo + 1]).array();
	MatrixXd delta = (prevDelta * weights[layerNo + 1].transpose()).array() * differential.transpose().array();
	return delta;
}

void backpropergation(VectorXd output[structure.size()], VectorXd teachData)
{
	VectorXd differential = output[structure.size() - 1].array() * (VectorXd::Ones(structure[structure.size() - 1]) - output[structure.size() - 1]).array();
	MatrixXd delta = (output[structure.size() - 1] - teachData).transpose().array() * differential.transpose().array();
	weights[structure.size() - 2] -= LEARNING_RATE * output[structure.size() - 2] * delta;
	biases[structure.size() - 2] -= LEARNING_RATE * delta.transpose();
	for (int i = 3; i <= structure.size(); ++i)
	{
		int n = structure.size() - i;
		cout << n << endl;
		delta = calcDelta(n, output, delta);
		weights[n] -= LEARNING_RATE * output[n] * delta;
		biases[n] -= LEARNING_RATE * delta.transpose();
	}
}

void validate()
{
	double error = 0.0;
	for (int i = 0; i < dataSet.size(); ++i)
	{
		//	feedforward proccess

		VectorXd output[structure.size()];
		output[0] = dataSet.row(i);

		for (int i = 0; i < structure.size() - 1; i++)
		{
			output[i + 1] = (output[i].transpose() * weights[i] + biases[i].transpose()).unaryExpr(activationFunc);
		}

		VectorXd teach(1);
		teach << teachSet[i];
		error += errorFunc(output[structure.size() - 1], teach).sum();
	}
	cout << error << endl;
}

VectorXd learnProccess(VectorXd input, VectorXd teachData)
{
	//	feedforward proccess
	VectorXd output[structure.size()];
	output[0] = input;

	for (int i = 0; i < structure.size() - 1; i++)
	{
		output[i + 1] = (output[i].transpose() * weights[i] + biases[i].transpose()).unaryExpr(activationFunc);
	}

	//	backpropergation method
	backpropergation(output, teachData);

	return output[structure.size() - 1];
}

int main()
{
	dataSet << 0 , 0 ,
		0 , 1 ,
		1 , 0 ,
		1 , 1;
	teachSet << 0 ,
		1 ,
		1 ,
		0;
	cout << dataSet << endl << endl;
	cout << dataSet.row(1) << endl;
	random_device rnd;
	mt19937 mt(rnd());
	initWeightsAndBiases();
	for (int i = 0; i < LEARNING_TIME; ++i)
	{
		int n = mt() % 4;
		VectorXd input = dataSet.row(n);

		VectorXd teach(1);
		teach << teachSet[n];
		VectorXd out = learnProccess(input, teach);
	}
	return 0;
}
