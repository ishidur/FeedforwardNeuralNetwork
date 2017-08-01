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
#include <ppl.h>
#include "XORDataSet.h"
#include "MNISTDataSet.h"
#include "TwoSpiralDataSet.h"
#include <numeric>

#define OUTPUTDATANUM 10000

//XOR
//std::vector<double> initVals = {0.001, 0.01, 0.1};
std::vector<double> initVals = {0.01};
#define TRIALS_PER_STRUCTURE 2
#define LEARNING_RATE 1.0
#define MOMENT 0.9
#define LEARNING_TIME 1000000
#define ERROR_BOTTOM 0.0001
//dataset
XORDataSet dataSet;
//Network structure.
std::vector<std::vector<int>> structures = {{2, 2, 1}, {2, 3, 1}, {2, 4, 1}, {2, 2, 2, 1}, {2, 3, 3, 1}, {2, 4, 4, 1}, {2, 2, 2, 2, 1}};
//std::vector<std::vector<int>> structures = {{2, 4, 4, 4, 1}};
bool useSoftmax = false;

//MNIST
//std::vector<double> initVals = {1.0};
//#define TRIALS_PER_STRUCTURE 1
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 1
//#define ERROR_BOTTOM 0.0001
//MNISTDataSet dataSet;
//Network structure.
//std::vector<std::vector<int>> structures = {{784, 100, 10}, {784, 100, 100, 10}, {784, 100, 100, 100, 10}};
//bool useSoftmax = true;

//TwoSpiral Prpblem
//std::vector<double> initVals = {0.1};
//#define TRIALS_PER_STRUCTURE 5
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 10
//#define ERROR_BOTTOM 0.0001
//TwoSpiralDataSet dataSet;
//std::vector<std::vector<int>> structures = {{2, 2, 1}, {2, 4, 1}, {2, 6, 1}, {2, 2, 2, 1}, {2, 2, 4, 1}, {2, 4, 2, 1}, {2, 2, 2, 2, 1}};
//bool useSoftmax = false;

std::vector<Eigen::MatrixXd> weights;
std::vector<Eigen::VectorXd> biases;
bool isFirst = true;
void initWeightsAndBiases(std::vector<int> structure, double iniitalVal)
{
	weights.clear();
	biases.clear();
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(Eigen::MatrixXd::Random(structure[i - 1], structure[i]) * iniitalVal);
		biases.push_back(Eigen::VectorXd::Random(structure[i]) * iniitalVal);
		//		biases.push_back(Eigen::VectorXd::Zero(structure[i]));
	}
}

Eigen::VectorXd Relu(Eigen::VectorXd inputs);
Eigen::VectorXd Tanh(Eigen::VectorXd inputs);
Eigen::VectorXd sigmoid(Eigen::VectorXd inputs);
Eigen::VectorXd softmax(Eigen::VectorXd inputs);

Eigen::VectorXd activationFunc(Eigen::VectorXd inputs)
{
	Eigen::VectorXd result = sigmoid(inputs);
	return result;
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
		//		Cross Entropy
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

std::vector<Eigen::MatrixXd> prevWeightDelta;
std::vector<Eigen::MatrixXd> prevBiasDelta;

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
	if (isFirst)
	{
		prevWeightDelta.clear();
		prevBiasDelta.clear();
		prevWeightDelta.push_back(-LEARNING_RATE * output[structure.size() - 2] * delta);
		prevBiasDelta.push_back(-LEARNING_RATE * delta);
		weights[structure.size() - 2] -= LEARNING_RATE * output[structure.size() - 2] * delta;
		biases[structure.size() - 2] -= LEARNING_RATE * delta.transpose();
		for (int i = 3; i <= structure.size(); ++i)
		{
			int n = structure.size() - i;
			delta = calcDelta(structure, n, output, delta);
			prevWeightDelta.push_back(-LEARNING_RATE * output[n] * delta);
			prevBiasDelta.push_back(-LEARNING_RATE * delta.transpose());
			weights[n] -= LEARNING_RATE * output[n] * delta;
			biases[n] -= LEARNING_RATE * delta.transpose();
		}
		isFirst = false;
	}
	else
	{
		weights[structure.size() - 2] += -LEARNING_RATE * output[structure.size() - 2] * delta + MOMENT * prevWeightDelta[0];
		biases[structure.size() - 2] += -LEARNING_RATE * delta.transpose() + MOMENT * prevBiasDelta[0];
		prevWeightDelta[0] = -LEARNING_RATE * output[structure.size() - 2] * delta + MOMENT * prevWeightDelta[0];
		prevBiasDelta[0] = -LEARNING_RATE * delta.transpose() + MOMENT * prevBiasDelta[0];
		for (int i = 3; i <= structure.size(); ++i)
		{
			int n = structure.size() - i;
			delta = calcDelta(structure, n, output, delta);
			weights[n] += -LEARNING_RATE * output[n] * delta + MOMENT * prevWeightDelta[i - 2];
			biases[n] += -LEARNING_RATE * delta.transpose() + MOMENT * prevBiasDelta[i - 2];
			prevWeightDelta[i - 2] = -LEARNING_RATE * output[n] * delta + MOMENT * prevWeightDelta[i - 2];
			prevBiasDelta[i - 2] = -LEARNING_RATE * delta.transpose() + MOMENT * prevBiasDelta[i - 2];
		}
	}
}

double validate(std::vector<int> structure)
{
	double error = 0.0;
	Concurrency::parallel_for<int>(0, dataSet.testDataSet.rows(), 1, [&error, structure](int i)
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
                               });
	//	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	//	{
	//		//	feedforward proccess
	//		std::vector<Eigen::VectorXd> outputs;
	//		outputs.push_back(dataSet.testDataSet.row(i).transpose());
	//
	//		for (int j = 0; j < structure.size() - 1; j++)
	//		{
	//			Eigen::VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
	//			Eigen::VectorXd output;
	//			if (useSoftmax && j == structure.size() - 2)
	//			{
	//				output = softmax(inputs);
	//			}
	//			else
	//			{
	//				output = activationFunc(inputs);
	//			}
	//			outputs.push_back(output);
	//		}
	//
	//		Eigen::VectorXd teach = dataSet.testTeachSet.row(i);
	//		error += errorFunc(outputs[structure.size() - 1], teach);
	//	}
	return error;
}

void MNISTtest(std::vector<int> structure, std::ostream& out = std::cout)
{
	int correct[10] = {0};
	int num[10] = {0};
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
		std::vector<double> y;
		y.resize(outputs[structure.size() - 1].size());
		Eigen::VectorXd::Map(&y[0], outputs[structure.size() - 1].size()) = outputs[structure.size() - 1];
		std::vector<double>::iterator result = std::max_element(y.begin(), y.end());
		std::vector<double> t;
		t.resize(dataSet.testTeachSet.row(i).size());
		Eigen::VectorXd::Map(&t[0], dataSet.testTeachSet.row(i).size()) = dataSet.testTeachSet.row(i);
		std::vector<double>::iterator teach = std::max_element(t.begin(), t.end());
		//		out << "input, " << std::endl;
		//		out << dataSet.testDataSet.row(i) << std::endl;
		if (i == 0)
		{
			out << outputs[structure.size() - 1].transpose() << std::endl;
		}
		out << "answer, " << "output" << std::endl;
		out << std::distance(y.begin(), result) << ", " << std::distance(t.begin(), teach) << std::endl;
		std::cout << std::distance(y.begin(), result) << ", " << std::distance(t.begin(), teach) << std::endl;
		int a = std::distance(t.begin(), teach);
		num[a]++;
		if (std::distance(y.begin(), result) == a)
		{
			correct[a]++;
		}
	}
	for (int i = 0; i < 10; ++i)
	{
		out << std::endl << i << "correct, " << correct[i] << ", /, " << num[i] << std::endl;
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

	//	double error = errorFunc(outputs[structure.size() - 1], teachData);
	double error = validate(structure);
	if (&out != &std::cout)
	{
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
		out << error << std::endl;
	}
	return error;
}

double singleRun(std::vector<int> structure, double initVal, std::string filename)
{
	initWeightsAndBiases(structure, initVal);
	//	int a[dataSet.dataSet.rows()] = {0};
	std::ofstream ofs(filename + ".csv");
	ofs << "step,";
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
	//	std::string progress = "";
	double error = 1.0;

	int s = 0;
	std::string progress = "";

	for (int i = 0; i < LEARNING_TIME && error > ERROR_BOTTOM; ++i)
	//	for (int i = 0; i < LEARNING_TIME; ++i)
	{
		//		double status = double((i + 1) * 100.0 / (LEARNING_TIME));
		//		if (progress.size() < int(status) / 5)
		//		{
		//			progress += "#";
		//		}
		//		std::cout << "progress: " << std::setw(4) << std::right << std::fixed << std::setprecision(1) << (status) << "% " << progress << "\r" << std::flush;
		std::vector<int> ns(dataSet.dataSet.rows());
		iota(ns.begin(), ns.end(), 0);
		shuffle(ns.begin(), ns.end(), std::mt19937());
		for (int n : ns)
		{
			s++;
			double status = double((s) * 100.0 / (ns.size() * LEARNING_TIME));
			if (progress.size() < int(status) / 5)
			{
				progress += "#";
			}

			Eigen::VectorXd input = dataSet.dataSet.row(n);
			Eigen::VectorXd teach = dataSet.teachSet.row(n);
			if (s % (ns.size() * LEARNING_TIME / OUTPUTDATANUM) == 0)
			{
				ofs << i << ",";
				error = learnProccess(structure, input, teach, ofs);
			}
			else
			{
				error = learnProccess(structure, input, teach);
			}
			std::cout << "progress: " << error << ", " << s << "/ " << (ns.size() * LEARNING_TIME) << " " << progress << "\r" << std::flush;
		}
	}
	ofs.close();
	//	std::ofstream ofs2(filename + "-test" + ".csv");
	//	MNISTtest(structure, ofs2);
	//	ofs2.close();
	std::cout << std::endl;
	return error;
}

int main()
{
	//	FILE* fp = _popen("gnuplot", "w");
	//	if (fp == nullptr)
	//		return -1;
	//	fputs("plot sin(x)\n", fp);
	//	fflush(fp);
	//	std::cin.get();
	//	_pclose(fp);
	//	return 0;
	dataSet.load();
	for (double init_val : initVals)
	{
		std::string dirName = "data\\";
		std::ostringstream sout;
		sout << std::fixed << init_val;
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
			ofs2 << "structures," << layers << std::endl;
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
				ofs2 << "file, " << fileName << std::endl;
				std::cout << fileName << std::endl;
				double err = singleRun(structure, init_val, fileName);
				ofs2 << "error, " << err << std::endl;
				std::cout << "error; " << err << std::endl;
				if (err < ERROR_BOTTOM)
				{
					correct++;
				}
				isFirst = true;
			}
			ofs2 << correct << ", /, " << TRIALS_PER_STRUCTURE << ", success" << std::endl;
			std::cout << correct << " / " << TRIALS_PER_STRUCTURE << " success" << std::endl;
		}
		ofs2.close();
	}
	return 0;
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
