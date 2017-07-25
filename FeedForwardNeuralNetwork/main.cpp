// FeedForwardNeuralNetwork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <time.h>
#include "XORDataSet.h"

#define TRIALS_PER_STRUCTURE 10
#define INITIAL_VAL 2.0
#define LEARNING_RATE 0.7
#define LEARNING_TIME 100000
#define ERROR_BOTTOM 0.0001

//XOR data
XORDataSet dataSet;

//Network structure.
vector<vector<int>> structures = {{2, 2, 1}, {2, 4, 1}, {2, 6, 1}, {2, 2, 2, 1}, {2, 2, 4, 1}, {2, 4, 2, 1}, {2, 4, 4, 1}, {2, 2, 2, 2, 1}};
vector<MatrixXd> weights;
vector<VectorXd> biases;

void initWeightsAndBiases(vector<int> structure)
{
	weights.clear();
	biases.clear();
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

MatrixXd calcDelta(vector<int> structure, int layerNo, vector<VectorXd> output, MatrixXd prevDelta)
{
	VectorXd differential = output[layerNo + 1].array() * (VectorXd::Ones(structure[layerNo + 1]) - output[layerNo + 1]).array();
	MatrixXd delta = (prevDelta * weights[layerNo + 1].transpose()).array() * differential.transpose().array();
	return delta;
}

void backpropergation(vector<int> structure, vector<VectorXd> output, VectorXd teachData)
{
	VectorXd differential = output[structure.size() - 1].array() * (VectorXd::Ones(structure[structure.size() - 1]) - output[structure.size() - 1]).array();
	MatrixXd delta = (output[structure.size() - 1] - teachData).transpose().array() * differential.transpose().array();
	weights[structure.size() - 2] -= LEARNING_RATE * output[structure.size() - 2] * delta;
	biases[structure.size() - 2] -= LEARNING_RATE * delta.transpose();
	for (int i = 3; i <= structure.size(); ++i)
	{
		int n = structure.size() - i;
		delta = calcDelta(structure, n, output, delta);
		weights[n] -= LEARNING_RATE * output[n] * delta;
		biases[n] -= LEARNING_RATE * delta.transpose();
	}
}

double validate(vector<int> structure)
{
	double error = 0.0;
	for (int i = 0; i < dataSet.dataSet.rows(); ++i)
	{
		//	feedforward proccess
		vector<VectorXd> output;
		output.push_back(dataSet.dataSet.row(i).transpose());

		for (int j = 0; j < structure.size() - 1; j++)
		{
			output.push_back((output[j].transpose() * weights[j] + biases[j].transpose()).unaryExpr(activationFunc));
		}

		VectorXd teach = dataSet.teachSet.row(i);
		error += errorFunc(output[structure.size() - 1], teach).sum();
	}
	return error;
}

void test(vector<int> structure)
{
	for (int i = 0; i < dataSet.dataSet.rows(); ++i)
	{
		//	feedforward proccess
		vector<VectorXd> output;
		output.push_back(dataSet.dataSet.row(i).transpose());

		for (int j = 0; j < structure.size() - 1; j++)
		{
			output.push_back((output[j].transpose() * weights[j] + biases[j].transpose()).unaryExpr(activationFunc));
		}
		cout << "input" << endl;
		cout << dataSet.dataSet.row(i) << endl;
		cout << "output" << endl;
		cout << output[structure.size() - 1] << endl;
		cout << "answer" << endl;
		cout << dataSet.teachSet.row(i) << endl;
	}
}

double learnProccess(vector<int> structure, VectorXd input, VectorXd teachData, ostream& out = cout)
{
	//	feedforward proccess
	vector<VectorXd> output;
	output.push_back(input);

	for (int i = 0; i < structure.size() - 1; i++)
	{
		output.push_back((output[i].transpose() * weights[i] + biases[i].transpose()).unaryExpr(activationFunc));
	}
	//	backpropergation method
	backpropergation(structure, output, teachData);

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

	double error = validate(structure);
	out << error << endl;
	return error;
}

void singleRun(vector<int> structure)
{
	random_device rnd;
	mt19937 mt(rnd());
	time_t epoch_time;
	epoch_time = time(NULL);

	initWeightsAndBiases(structure);
	//	int a[dataSet.dataSet.rows()] = {0};
	string filename = "result-";
	filename += to_string(structure.size()) + "-layers-";
	for (int i = 0; i < structure.size(); ++i)
	{
		filename += to_string(structure[i]);
		if (i < structure.size() - 1)
		{
			filename += "X";
		}
	}
	filename += "-" + to_string(epoch_time);
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
		int n = mt() % dataSet.dataSet.rows();
		//		a[n]++;
		VectorXd input = dataSet.dataSet.row(n);
		VectorXd teach = dataSet.teachSet.row(n);
		error = learnProccess(structure, input, teach, ofs);
		//		error = learnProccess(input, teach);
	}
	ofs.close();
	cout << endl;
	cout << "error; " << error << endl;
}

int main()
{
	for (vector<int> structure : structures)
	{
		for (int i = 0; i < TRIALS_PER_STRUCTURE; ++i)
		{
			cout << "try: " << i << endl;
			singleRun(structure);
		}
	}
	//	for (int i = 0; i < dataSet.dataSet.rows(); ++i)
	//	{
	//		cout << i << ";" << endl;
	//		cout << a[i] << endl;
	//	}
	//	test();
	return 0;
}
