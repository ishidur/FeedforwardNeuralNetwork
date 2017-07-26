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

//#define TRIALS_PER_STRUCTURE 10
#define TRIALS_PER_STRUCTURE 1
#define LEARNING_RATE 0.7
#define LEARNING_TIME 1
//#define LEARNING_TIME 10000
#define ERROR_BOTTOM 0.01

std::vector<double> initVals = {2.0};
//vector<double> initVals = {0.3, 1.0, 2.0};
//dataset
MNISTDataSet dataSet;

//Network structure.
std::vector<std::vector<int>> structures = {{2, 2, 2, 2, 1}};
//vector<vector<int>> structures = {{2, 2, 1}, {2, 4, 1}, {2, 6, 1}, {2, 2, 2, 1}, {2, 2, 4, 1}, {2, 4, 2, 1}, {2, 2, 2, 2, 1}};
std::vector<Eigen::MatrixXd> weights;
std::vector<Eigen::VectorXd> biases;

bool useSoftmax = true;

void initWeightsAndBiases(std::vector<int> structure, double iniitalVal)
{
	weights.clear();
	biases.clear();
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(Eigen::MatrixXd::Random(structure[i - 1], structure[i]) * iniitalVal);
		biases.push_back(Eigen::VectorXd::Random(structure[i]) * iniitalVal);
	}
}

auto relu = [](const double input)
{
	if (input < 0.0)
	{
		return 0.0;
	}
	return input;
};

Eigen::VectorXd Relu(Eigen::VectorXd inputs)
{
	return inputs.unaryExpr(relu);
}

auto tanhype = [](const double input)
{
	return tanh(input);
};

Eigen::VectorXd Tanh(Eigen::VectorXd inputs)
{
	return inputs.unaryExpr(tanhype);
}

auto sigm = [](const double input)
{
	return 1.0 / (1 + exp(-input));
};

Eigen::VectorXd sigmoid(Eigen::VectorXd inputs)
{
	return inputs.unaryExpr(sigm);
}

Eigen::VectorXd activationFunc(Eigen::VectorXd inputs)
{
	Eigen::VectorXd result = sigmoid(inputs);
	return result;
}

auto soft = [](const double x)
{
	return exp(x);
};

Eigen::VectorXd softmax(Eigen::VectorXd inputs)
{
	Eigen::VectorXd a = inputs.unaryExpr(soft);
	double s = a.sum();
	Eigen::VectorXd b = a / s;
	return b;
}

Eigen::VectorXd differential(Eigen::VectorXd input)
{
	Eigen::VectorXd result = input.array() * (Eigen::VectorXd::Ones(input.size()) - input).array();
	return result;
}

auto squared = [](const double x)
{
	return x * x;
};
auto cross = [](const double x)
{
	double y = log(x);
	return y;
};

double errorFunc(Eigen::VectorXd outData, Eigen::VectorXd teachData)
{
	double error;
	if (useSoftmax)
	{
		Eigen::VectorXd v1 = outData.unaryExpr(cross);
		Eigen::VectorXd v2 = teachData;
		error = -v2.dot(v1);
	}
	else
	{
		//	Mean Square Error
		Eigen::VectorXd err = (teachData - outData).unaryExpr(squared);
		err *= 1.0 / 2.0;
		error = err.sum();
	}
	return error;
}

Eigen::MatrixXd calcDelta(std::vector<int> structure, int layerNo, std::vector<Eigen::VectorXd> output, Eigen::MatrixXd prevDelta)
{
	Eigen::VectorXd diff = differential(output[layerNo + 1]);
	Eigen::MatrixXd delta = (prevDelta * weights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void backpropergation(std::vector<int> structure, std::vector<Eigen::VectorXd> output, Eigen::VectorXd teachData)
{
	Eigen::VectorXd diff = differential(output[structure.size() - 1]);
	Eigen::MatrixXd delta;

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

double validate(std::vector<int> structure)
{
	double error = 0.0;
	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	{
		//	feedforward proccess
		std::vector<Eigen::VectorXd> outputs;
		outputs.push_back(dataSet.testDataSet.row(i).transpose());

		for (int j = 0; j < structure.size() - 1; j++)
		{
			Eigen::VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
			Eigen::VectorXd output;
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

		Eigen::VectorXd teach = dataSet.testTeachSet.row(i);
		error += errorFunc(outputs[structure.size() - 1], teach);
	}
	return error;
}

void test(std::vector<int> structure)
{
	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	{
		//	feedforward proccess
		std::vector<Eigen::VectorXd> outputs;
		outputs.push_back(dataSet.testDataSet.row(i).transpose());

		for (int j = 0; j < structure.size() - 1; j++)
		{
			Eigen::VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
			Eigen::VectorXd output;
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
		std::cout << "input" << std::endl;
		std::cout << dataSet.testDataSet.row(i) << std::endl;
		std::cout << "output" << std::endl;
		std::cout << outputs[structure.size() - 1] << std::endl;
		std::cout << "answer" << std::endl;
		std::cout << dataSet.testTeachSet.row(i) << std::endl;
	}
}

double learnProccess(std::vector<int> structure, Eigen::VectorXd input, Eigen::VectorXd teachData, std::ostream& out = std::cout)
{
	//	feedforward proccess
	std::vector<Eigen::VectorXd> outputs;
	outputs.push_back(input);

	for (int i = 0; i < structure.size() - 1; i++)
	{
		Eigen::VectorXd inputs = (outputs[i].transpose() * weights[i] + biases[i].transpose());
		Eigen::VectorXd output;
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
	out << error << std::endl;
	return error;
}

double singleRun(std::vector<int> structure, double initVal, std::string filename)
{
	initWeightsAndBiases(structure, initVal);
	//	int a[dataSet.dataSet.rows()] = {0};

	std::ofstream ofs(filename);
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
	ofs << "error" << std::endl;
	std::string progress = "";
	double error = 1.0;
	//	for (int i = 0; i < LEARNING_TIME && error > ERROR_BOTTOM; ++i)
	for (int i = 0; i < LEARNING_TIME; ++i)
	{
		double status = double(i * 100.0 / (LEARNING_TIME - 1));
		if (progress.size() < int(status) / 5)
		{
			progress += "#";
		}
		std::cout << "progress: " << std::setw(4) << std::right << std::fixed << std::setprecision(1) << (status) << "% " << progress << "\r" << std::flush;

		std::vector<int> ns(dataSet.dataSet.rows());
		iota(ns.begin(), ns.end(), 0);
		shuffle(ns.begin(), ns.end(), std::mt19937());
		for (int n : ns)
		{
			Eigen::VectorXd input = dataSet.dataSet.row(n);
			Eigen::VectorXd teach = dataSet.teachSet.row(n);
			error = learnProccess(structure, input, teach, ofs);
			//		error = learnProccess(input, teach);
		}
	}
	ofs.close();
	std::cout << std::endl;
	return error;
}

int main()
{
	dataSet.load();
	for (double init_val : initVals)
	{
		std::string dirName = "data\\";
		std::ostringstream sout;
		sout << std::fixed << std::setprecision(1) << init_val;
		std::string s = sout.str();
		dirName += s;
		dirName += "\\";
		if (!MakeSureDirectoryPathExists(dirName.c_str()))
		{
			break;
		}
		std::string filename = dirName;
		filename += "static.csv";
		std::ofstream ofs2(filename);
		for (std::vector<int> structure : structures)
		{
			std::string layers = "";
			for (int i = 0; i < structure.size(); ++i)
			{
				layers += std::to_string(structure[i]);
				if (i < structure.size() - 1)
				{
					layers += "X";
				}
			}
			ofs2 << "structures, " << layers << std::endl;
			std::cout << "structures; " << layers << std::endl;
			int correct = 0;
			for (int i = 0; i < TRIALS_PER_STRUCTURE; ++i)
			{
				time_t epoch_time;
				epoch_time = time(NULL);
				ofs2 << "try, " << i << std::endl;
				std::cout << "try: " << i << std::endl;
				std::string fileName = dirName;
				fileName += "result-";
				fileName += std::to_string(structure.size()) + "-layers-";
				fileName += layers;
				fileName += "-" + std::to_string(epoch_time);
				fileName += ".csv";
				ofs2 << "file, " << fileName << std::endl;
				std::cout << fileName << std::endl;
				double err = singleRun(structure, init_val, fileName);
				ofs2 << "error, " << err << std::endl;
				std::cout << "error; " << err << std::endl;
				if (err < ERROR_BOTTOM)
				{
					correct++;
				}
			}
			ofs2 << correct << " / " << TRIALS_PER_STRUCTURE << " success" << std::endl;
			std::cout << correct << " / " << TRIALS_PER_STRUCTURE << " success" << std::endl;
		}
		ofs2.close();
	}
	//	for (int i = 0; i < dataSet.dataSet.rows(); ++i)
	//	{
	//		std::cout << i << ";" << std::endl;
	//		std::cout << a[i] << std::endl;
	//	}
	//	test();
	return 0;
}
