// FeedForwardNeuralNetwork.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"
#include <array>
#include <vector>
#include "Eigen/Core"
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <time.h>

#define INITIAL_VAL 0.03
#define LEARNING_RATE 0.7
#define LEARNING_TIME 100000
#define ERROR_BOTTOM 0.0001
using namespace std;
using namespace Eigen;

//XOR data
//(x, y)
MatrixXd dataSet(4, 2);
VectorXd teachSet(4);

//Network structure. make sure layer number is equal to array number.
array<int, 4> structure = {2, 2, 2, 1};
vector<MatrixXd> weights;
vector<VectorXd> biases;

void initWeightsAndBiases()
{
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(MatrixXd::Random(structure[i - 1], structure[i]) * INITIAL_VAL);
		biases.push_back(VectorXd::Random(structure[i]) * INITIAL_VAL);
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

double validate()
{
	double error = 0.0;
	for (int i = 0; i < dataSet.rows(); ++i)
	{
		//	feedforward proccess
		VectorXd output[structure.size()];
		output[0] = dataSet.row(i).transpose();

		for (int j = 0; j < structure.size() - 1; j++)
		{
			output[j + 1] = (output[j].transpose() * weights[j] + biases[j].transpose()).unaryExpr(activationFunc);
		}

		VectorXd teach(1);
		teach << teachSet[i];
		error += errorFunc(output[structure.size() - 1], teach).sum();
	}
	return error;
}

void test()
{
	for (int i = 0; i < dataSet.rows(); ++i)
	{
		//	feedforward proccess
		VectorXd output[structure.size()];
		output[0] = dataSet.row(i).transpose();

		for (int j = 0; j < structure.size() - 1; j++)
		{
			output[j + 1] = (output[j].transpose() * weights[j] + biases[j].transpose()).unaryExpr(activationFunc);
		}
		cout << "input" << endl;
		cout << dataSet.row(i) << endl;
		cout << "output" << endl;
		cout << output[structure.size() - 1] << endl;
		cout << "answer" << endl;
		cout << teachSet[i] << endl;
	}
}

double learnProccess(VectorXd input, VectorXd teachData, ostream& out = cout)
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

	for (int i = 0; i < structure.size() - 1; ++i)
	{
		for (int j = 0; j < weights[i].rows(); ++j)
		{
			for (int k = 0; k < weights[i].cols(); ++k)
			{
				out << weights[i](j, k) << ", ";
			}
		}
		for (int j = 0; j < biases[i].size(); ++j)
		{
			out << biases[i][j] << ", ";
		}
	}

	double error = validate();
	out << error << endl;
	return error;
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
	random_device rnd;
	mt19937 mt(rnd());
	time_t epoch_time;
	epoch_time = time(NULL);

	initWeightsAndBiases();
	int a[4] = {0};
	string filename = "result-";
	filename += to_string(epoch_time) + to_string(structure.size()) + "-layers-";
	for (int i = 0; i < structure.size(); ++i)
	{
		filename += to_string(structure[i]);
		if (i < structure.size() - 1)
		{
			filename += "X";
		}
	}
	filename += ".csv";
	ofstream ofs(filename);
	ofs << "learning time" << ", ";

	for (int i = 0; i < structure.size() - 1; ++i)
	{
		for (int j = 0; j < weights[i].rows(); ++j)
		{
			for (int k = 0; k < weights[i].cols(); ++k)
			{
				ofs << "weight:" << "l:" << i << ":" << j << ":" << k << ", ";
			}
		}
		for (int j = 0; j < biases[i].size(); ++j)
		{
			ofs << "bias:" << "l:" << i << ":" << j << ", ";
		}
	}
	ofs << "error" << endl;
	string progress = "";
	double error = 1.0;
	//	for (int i = 0; i < LEARNING_TIME && error > ERROR_BOTTOM; ++i)
	for (int i = 0; i < LEARNING_TIME; ++i)
	{
		double status = double(i * 100.0 / (LEARNING_TIME - 1));
		if (progress.size() < int(status) / 5)
		{
			progress += "#";
		}
		cout << "progress: " << setw(4) << right << fixed << setprecision(1) << (status) << "% " << progress << "\r" << flush;
		ofs << i << ", ";
		int n = mt() % 4;
		a[n]++;
		VectorXd input = dataSet.row(n);
		VectorXd teach(1);
		teach << teachSet[n];
		error = learnProccess(input, teach, ofs);
		//		error = learnProccess(input, teach);
	}
	cout << endl;
	cout << error << endl;

	for (int i = 0; i < 4; ++i)
	{
		cout << i << ";" << endl;
		cout << a[i] << endl;
	}
	test();
	return 0;
}
