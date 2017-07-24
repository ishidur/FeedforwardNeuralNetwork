// FeedForwardNeuralNetwork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <array>
#include <vector>
#include "Eigen/Core"
#include <iostream>
#include <random>
#include <fstream>

#define LEARNING_RATE 0.7
#define LEARNING_TIME 24000
using namespace std;
using namespace Eigen;

//XOR data
//(x, y)
MatrixXd dataSet(4, 2);
VectorXd teachSet(4);

//Network structure. make sure layer number is equal to array number.
array<int, 3> structure = {2, 2, 1};
vector<MatrixXd> weights;
vector<VectorXd> biases;

void initWeightsAndBiases()
{
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(MatrixXd::Random(structure[i - 1], structure[i]) * 0.03);
		biases.push_back(VectorXd::Random(structure[i]) * 0.03);
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
		delta = calcDelta(n, output, delta);
		weights[n] -= LEARNING_RATE * output[n] * delta;
		biases[n] -= LEARNING_RATE * delta.transpose();
	}
}

void validate(ostream& out = cout)
{
	double error = 0.0;
	for (int i = 0; i < dataSet.rows(); ++i)
	{
		//	feedforward proccess
		VectorXd output[structure.size()];
		output[0] = dataSet.row(i).transpose();

		for (int i = 0; i < structure.size() - 1; i++)
		{
			output[i + 1] = (output[i].transpose() * weights[i] + biases[i].transpose()).unaryExpr(activationFunc);
		}

		VectorXd teach(1);
		teach << teachSet[i];
		error += errorFunc(output[structure.size() - 1], teach).sum();
	}
	out << error << endl;
}

void test()
{
	for (int i = 0; i < dataSet.rows(); ++i)
	{
		//	feedforward proccess
		VectorXd output[structure.size()];
		output[0] = dataSet.row(i).transpose();

		for (int i = 0; i < structure.size() - 1; i++)
		{
			output[i + 1] = (output[i].transpose() * weights[i] + biases[i].transpose()).unaryExpr(activationFunc);
		}
		cout << "input" << endl;
		cout << dataSet.row(i) << endl;
		cout << "output" << endl;
		cout << output[structure.size() - 1] << endl;
		cout << "answer" << endl;
		cout << teachSet[i] << endl;
	}
}

void learnProccess(VectorXd input, VectorXd teachData, ostream& out = cout)
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

	validate(out);
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
	random_device rnd;
	mt19937 mt(rnd());
	initWeightsAndBiases();
	int a[4] = {0};
	ofstream ofs("testResult.csv");

	for (int i = 0; i < LEARNING_TIME; ++i)
	{
		int n = mt() % 4;
		a[n]++;
		VectorXd input = dataSet.row(n);
		VectorXd teach(1);
		teach << teachSet[n];
		learnProccess(input, teach, ofs);
	}
	for (int i = 0; i < 4; ++i)
	{
		cout << i << ";" << endl;
		cout << a[i] << endl;
	}
	test();
	return 0;
}
