// FeedForwardNeuralNetwork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include<windows.h>
#include <imagehlp.h>
#pragma comment(lib, "imagehlp.lib")
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <time.h>
#include "XORDataSet.h"
#include "MNISTDataSet.h"
#include <numeric>

#define TRIALS_PER_STRUCTURE 10
#define INITIAL_VAL 0.3
#define LEARNING_RATE 0.7
#define LEARNING_TIME 10000
#define ERROR_BOTTOM 0.01

vector<double> initVals = {0.3, 1.0, 2.0};
//dataset
MNISTDataSet dataSet;

//Network structure.
vector<vector<int>> structures = {{2, 2, 1}, {2, 4, 1}, {2, 6, 1}, {2, 2, 2, 1}, {2, 2, 4, 1}, {2, 4, 2, 1}, {2, 2, 2, 2, 1}};
vector<MatrixXd> weights;
vector<VectorXd> biases;

bool useSoftmax = true;

void initWeightsAndBiases(vector<int> structure, double iniitalVal)
{
	weights.clear();
	biases.clear();
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(MatrixXd::Random(structure[i - 1], structure[i]) * iniitalVal);
		biases.push_back(VectorXd::Random(structure[i]) * iniitalVal);
	}
}

double relu(double input)
{
	if (input < 0.0)
	{
		return 0.0;
	}
	return input;
}

VectorXd Relu(VectorXd inputs)
{
	return inputs.unaryExpr(relu);
}

double tanhype(double input)
{
	return tanh(input);
}

VectorXd Tanh(VectorXd inputs)
{
	return inputs.unaryExpr(tanhype);
}

double sigm(double input)
{
	return 1.0 / (1 + exp(-input));
}

VectorXd sigmoid(VectorXd inputs)
{
	return inputs.unaryExpr(sigm);
}

VectorXd activationFunc(VectorXd inputs)
{
	return sigmoid(inputs);
}

auto soft = [](const double x)
{
	return exp(x);
};

VectorXd differential(VectorXd input)
{
	return input.array() * (VectorXd::Ones(input.size()) - input).array();
}

VectorXd softmax(VectorXd inputs)
{
	VectorXd a = inputs.unaryExpr(soft);
	VectorXd b = a / a.sum();
	return b;
}

auto squared = [](const double x)
{
	return x * x;
};
auto cross = [](const double x)
{
	return log(x);
};

double errorFunc(VectorXd outData, VectorXd teachData)
{
	double error;
	if (useSoftmax)
	{
		VectorXd v1 = outData.unaryExpr(cross);
		VectorXd v2 = teachData;
		error = -v2.dot(v1);
	}
	else
	{
		//	Mean Square Error
		VectorXd err = (teachData - outData).unaryExpr(squared);
		err *= 1.0 / 2.0;
		error = err.sum();
	}
	return error;
}

MatrixXd calcDelta(vector<int> structure, int layerNo, vector<VectorXd> output, MatrixXd prevDelta)
{
	VectorXd diff = differential(output[layerNo + 1]);
	MatrixXd delta = (prevDelta * weights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void backpropergation(vector<int> structure, vector<VectorXd> output, VectorXd teachData)
{
	VectorXd diff = differential(output[structure.size() - 1]);
	MatrixXd delta;

	if (useSoftmax)
	{
		delta = (output[structure.size() - 1] - teachData).transpose();
	}
	else
	{
		delta = (output[structure.size() - 1] - teachData).transpose().array() * diff.transpose().array();
	}
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
	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	{
		//	feedforward proccess
		vector<VectorXd> outputs;
		outputs.push_back(dataSet.testDataSet.row(i).transpose());

		for (int j = 0; j < structure.size() - 1; j++)
		{
			VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
			VectorXd output;
			if (useSoftmax && j == structure.size() - 2)
			{
				output = softmax(inputs);
			}
			else
			{
				output = activationFunc(inputs);
			}
			outputs.push_back(output);
		}

		VectorXd teach = dataSet.testTeachSet.row(i);
		error += errorFunc(outputs[structure.size() - 1], teach);
	}
	return error;
}

void test(vector<int> structure)
{
	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	{
		//	feedforward proccess
		vector<VectorXd> outputs;
		outputs.push_back(dataSet.testDataSet.row(i).transpose());

		for (int j = 0; j < structure.size() - 1; j++)
		{
			VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
			VectorXd output;
			if (useSoftmax && j == structure.size() - 2)
			{
				output = softmax(inputs);
			}
			else
			{
				output = activationFunc(inputs);
			}
			outputs.push_back(output);
		}
		cout << "input" << endl;
		cout << dataSet.testDataSet.row(i) << endl;
		cout << "output" << endl;
		cout << outputs[structure.size() - 1] << endl;
		cout << "answer" << endl;
		cout << dataSet.testTeachSet.row(i) << endl;
	}
}

double learnProccess(vector<int> structure, VectorXd input, VectorXd teachData, ostream& out = cout)
{
	//	feedforward proccess
	vector<VectorXd> outputs;
	outputs.push_back(input);

	for (int i = 0; i < structure.size() - 1; i++)
	{
		VectorXd inputs = (outputs[i].transpose() * weights[i] + biases[i].transpose());
		VectorXd output;
		if (useSoftmax && i == structure.size() - 2)
		{
			output = softmax(inputs);
		}
		else
		{
			output = activationFunc(inputs);
		}
		outputs.push_back(output);
	}
	//	backpropergation method
	backpropergation(structure, outputs, teachData);

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

double singleRun(vector<int> structure, double initVal, string filename)
{
	initWeightsAndBiases(structure, initVal);
	//	int a[dataSet.dataSet.rows()] = {0};

	ofstream ofs(filename);
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

		vector<int> ns(dataSet.dataSet.rows());
		iota(ns.begin(), ns.end(), 0);
		shuffle(ns.begin(), ns.end(), mt19937());
		for (int n : ns)
		{
			VectorXd input = dataSet.dataSet.row(n);
			VectorXd teach = dataSet.teachSet.row(n);
			error = learnProccess(structure, input, teach, ofs);
			//		error = learnProccess(input, teach);
		}
	}
	ofs.close();
	cout << endl;
	return error;
}

int main()
{
	for (double init_val : initVals)
	{
		string dirName = "data\\";
		ostringstream sout;
		sout << fixed << setprecision(1) << init_val;
		string s = sout.str();
		dirName += s;
		dirName += "\\";
		if (!MakeSureDirectoryPathExists(dirName.c_str()))
		{
			break;
		}
		string filename = dirName;
		filename += "static.csv";
		ofstream ofs2(filename);
		for (vector<int> structure : structures)
		{
			string layers = "";
			for (int i = 0; i < structure.size(); ++i)
			{
				layers += to_string(structure[i]);
				if (i < structure.size() - 1)
				{
					layers += "X";
				}
			}
			ofs2 << "structures, " << layers << endl;
			cout << "structures; " << layers << endl;
			int correct = 0;
			for (int i = 0; i < TRIALS_PER_STRUCTURE; ++i)
			{
				time_t epoch_time;
				epoch_time = time(NULL);
				ofs2 << "try, " << i << endl;
				cout << "try: " << i << endl;
				string filename = dirName;
				filename += "result-";
				filename += to_string(structure.size()) + "-layers-";
				filename += layers;
				filename += "-" + to_string(epoch_time);
				filename += ".csv";
				ofs2 << "file, " << filename << endl;
				cout << filename << endl;
				double err = singleRun(structure, init_val, filename);
				ofs2 << "error, " << err << endl;
				cout << "error; " << err << endl;
				if (err < ERROR_BOTTOM)
				{
					correct++;
				}
			}
			ofs2 << correct << " / " << TRIALS_PER_STRUCTURE << " success" << endl;
			cout << correct << " / " << TRIALS_PER_STRUCTURE << " success" << endl;
		}
		ofs2.close();
	}
	//	for (int i = 0; i < dataSet.dataSet.rows(); ++i)
	//	{
	//		cout << i << ";" << endl;
	//		cout << a[i] << endl;
	//	}
	//	test();
	return 0;
}
