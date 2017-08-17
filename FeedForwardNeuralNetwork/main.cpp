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

#include "ActivationFunctions.h"

#include "XORDataSet.h"
#include "MNISTDataSet.h"
#include "TwoSpiralDataSet.h"
#include "FuncApproxDataSet.h"
#include <numeric>

#define SLIDE 1

////XOR
////std::vector<double> initVals = {0.001, 0.01, 0.1};
//std::vector<double> initVals = {0.01};
//#define TRIALS_PER_STRUCTURE 5
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 1000000
//#define ERROR_BOTTOM 0.01
////dataset
//XORDataSet dataSet;
////Network structure.
////std::vector<std::vector<int>> structures = {{2, 2, 1}, {2, 3, 1}, {2, 4, 1}};
//std::vector<std::vector<int>> structures = {{2, 2, 2, 1},{2, 3, 3, 1},{2, 4, 4, 1},{2, 2, 2, 2, 1},{2, 3, 3, 3, 1},{2, 4, 4, 4, 1}};

//Function approximation
std::vector<double> initVals = {0.01};
#define TRIALS_PER_STRUCTURE 1
#define LEARNING_RATE 0.05
#define LEARNING_TIME 10000
#define ERROR_BOTTOM 0.00000001
//dataset
FuncApproxDataSet dataSet;
//Network structure.
std::vector<std::vector<int>> structures = {{1, 5, 1}};
//std::vector<std::vector<int>> structures = {{1, 2, 1}, {1, 3, 1}, {1, 4, 1}};

//MNIST
//std::vector<double> initVals = {1.0};
//#define TRIALS_PER_STRUCTURE 1
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 1
//#define ERROR_BOTTOM 0.01
//MNISTDataSet dataSet;
//Network structure.
//std::vector<std::vector<int>> structures = {{784, 100, 10}, {784, 100, 100, 10}, {784, 100, 100, 100, 10}};

//TwoSpiral Prpblem
//std::vector<double> initVals = {0.1};
//#define TRIALS_PER_STRUCTURE 5
//#define LEARNING_RATE 1.0
//#define LEARNING_TIME 10
//#define ERROR_BOTTOM 0.01
//TwoSpiralDataSet dataSet;
//std::vector<std::vector<int>> structures = {{2, 2, 1}, {2, 4, 1}, {2, 6, 1}, {2, 2, 2, 1}, {2, 2, 4, 1}, {2, 4, 2, 1}, {2, 2, 2, 2, 1}};
//bool useSoftmax = false;

Eigen::VectorXd activationFunc(Eigen::VectorXd inputs)
{
	Eigen::VectorXd result = sigmoid(inputs);
	return result;
}

Eigen::VectorXd differential(Eigen::VectorXd input)
{
	Eigen::VectorXd result = differentialSigmoid(input);
	return result;
}

Eigen::VectorXd outputActivationFunc(Eigen::VectorXd inputs)
{
	if (dataSet.useSoftmax)
	{
		return softmax(inputs);
	}
	return inputs;
	//	return activationFunc(inputs);
}

Eigen::VectorXd outputDifferential(Eigen::VectorXd input)
{
	if (dataSet.useSoftmax)
	{
		return Eigen::VectorXd::Ones(input.size());
	}
	return Eigen::VectorXd::Ones(input.size());
	//	return differential(input);
}

std::vector<Eigen::MatrixXd> weights;
std::vector<Eigen::VectorXd> biases;

void initWeightsAndBiases(std::vector<int> structure, double iniitalVal)
{
	weights.clear();
	biases.clear();
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(Eigen::MatrixXd::Random(structure[i - 1], structure[i]) * iniitalVal);
		//		biases.push_back(Eigen::VectorXd::Random(structure[i]) * iniitalVal);
		biases.push_back(Eigen::VectorXd::Zero(structure[i]));
	}
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
	if (dataSet.useSoftmax)
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


void backpropergation(std::vector<int> structure, std::vector<Eigen::VectorXd> output, Eigen::VectorXd teachData)
{
	Eigen::VectorXd diff = outputDifferential(output[structure.size() - 1]);
	Eigen::MatrixXd delta = (output[structure.size() - 1] - teachData).transpose().array() * diff.transpose().array();
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

double validate(std::vector<int> structure, bool show = false)
{
	double error = 0.0;
	Eigen::VectorXd outs = Eigen::VectorXd::Zero(dataSet.testDataSet.rows());
	std::mutex mtx;

	Concurrency::parallel_for<int>(0, dataSet.testDataSet.rows(), 1, [&error, &outs, &mtx, structure](int i)
                               {
	                               //	feedforward proccess
	                               std::vector<Eigen::VectorXd> outputs;
	                               outputs.push_back(dataSet.testDataSet.row(i).transpose());

	                               for (int j = 0; j < structure.size() - 1; j++)
	                               {
		                               Eigen::VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
		                               Eigen::VectorXd output;
		                               if (j == structure.size() - 2)
		                               {
			                               output = outputActivationFunc(inputs);
		                               }
		                               else
		                               {
			                               output = activationFunc(inputs);
		                               }
		                               outputs.push_back(output);
	                               }
	                               mtx.lock();
	                               outs[i] = outputs[structure.size() - 1].sum();
	                               mtx.unlock();
	                               Eigen::VectorXd teach = dataSet.testTeachSet.row(i);
	                               error += errorFunc(outputs[structure.size() - 1], teach);
                               });

	if (show && typeid(dataSet) == typeid(FuncApproxDataSet))
	{
		dataSet.update(outs);
	}
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
	return error / dataSet.testDataSet.rows();
}

void Softmaxtest(std::vector<int> structure, std::ostream& out = std::cout)
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
			if (j == structure.size() - 2)
			{
				output = outputActivationFunc(inputs);

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

double learnProccess(std::vector<int> structure, int iterator, Eigen::VectorXd input, Eigen::VectorXd teachData, std::ostream& out = std::cout)
{
	//	feedforward proccess
	std::vector<Eigen::VectorXd> outputs;
	outputs.push_back(input);

	for (int i = 0; i < structure.size() - 1; i++)
	{
		Eigen::VectorXd inputs = (outputs[i].transpose() * weights[i] + biases[i].transpose());
		Eigen::VectorXd output;
		if (i == structure.size() - 2)
		{
			output = outputActivationFunc(inputs);

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
	double error = validate(structure, iterator % 100 == 0);
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

	for (int i = 0; i < LEARNING_TIME; ++i)
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
			if (s % (ns.size() * SLIDE) == 0)
			{
				ofs << i << ",";
				error = learnProccess(structure, s, input, teach, ofs);
				if (error < ERROR_BOTTOM)
				{
					goto learn_end;
				}
			}
			else
			{
				error = learnProccess(structure, s, input, teach);
			}
			std::cout << "progress: " << error << ", " << s << "/ " << (ns.size() * LEARNING_TIME) << " " << progress << "\r" << std::flush;
		}
	}
learn_end:
	ofs.close();
	//	std::ofstream ofs2(filename + "-test" + ".csv");
	//	Softmaxtest(structure, ofs2);
	//	ofs2.close();
	std::cout << std::endl;
	return error;
}

int main()
{
	dataSet.load();
	if (typeid(dataSet) == typeid(FuncApproxDataSet))
	{
		dataSet.show();
	}
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
			}
			ofs2 << correct << ", /, " << TRIALS_PER_STRUCTURE << ", success" << std::endl;
			std::cout << correct << " / " << TRIALS_PER_STRUCTURE << " success" << std::endl;
		}
		ofs2.close();
	}
	return 0;
}
