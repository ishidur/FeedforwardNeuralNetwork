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
#include <tuple>

#include "ActivationFunctions.h"
#include "config.h"

#include <numeric>
using namespace std;
using namespace Eigen;


std::vector<Eigen::MatrixXd> weights;
std::vector<Eigen::VectorXd> biases;

void initWeightsAndBiases(std::vector<int> const& structure, double iniitalVal)
{
	weights.clear();
	biases.clear();
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(Eigen::MatrixXd::Random(structure[i - 1], structure[i]) * iniitalVal);
		biases.push_back(Eigen::VectorXd::Random(structure[i]) * iniitalVal);
	}
}

void Softmaxtest(std::vector<int> const& structure, std::ostream& out = std::cout);
double learnProccess(std::vector<int> const& structure, int iterator, Eigen::VectorXd const& input, Eigen::VectorXd const& teachData, std::ostream& out = std::cout);
void pretrain(std::vector<int> const& structure, std::ostream& out = std::cout);

std::tuple<double, int> singleRun(std::vector<int> const& structure, double const& initVal, std::string filename)
{
	initWeightsAndBiases(structure, initVal);
	//pretraining process
	std::ofstream preofs(filename + "-ae" + ".csv");
	pretrain(structure, preofs);
	preofs.close();
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
	int times = LEARNING_TIME;
	std::vector<int> ns(dataSet.dataSet.rows());
	const int c = ns.size() * SLIDE;
	for (int i = 0; i < LEARNING_TIME; ++i)
	//	for (int i = 0; i < LEARNING_TIME; ++i)
	{
		//		double status = double((i + 1) * 100.0 / (LEARNING_TIME));
		//		if (progress.size() < int(status) / 5)
		//		{
		//			progress += "#";
		//		}
		//		std::cout << "progress: " << std::setw(4) << std::right << std::fixed << std::setprecision(1) << (status) << "% " << progress << "\r" << std::flush;
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
			if (s % c == 0)
			{
				ofs << i << ",";
				error = learnProccess(structure, s, input, teach, ofs);
				if (error < ERROR_BOTTOM)
				{
					times = i;
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
	if (dataSet.useSoftmax)
	{
		std::ofstream ofs2(filename + "-test" + ".csv");
		Softmaxtest(structure, ofs2);
		ofs2.close();
	}
	std::cout << std::endl;
	return std::forward_as_tuple(error, times);
}

int main()
{
	dataSet.load();
	//	if (typeid(dataSet) == typeid(FuncApproxDataSet))
	//	{
	//		dataSet.show();
	//	}
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
				epoch_time = time(nullptr);
				ofs2 << "try," << i << std::endl;
				std::cout << "try: " << i << std::endl;
				std::string fileName = dirName;
				fileName += "result-";
				fileName += std::to_string(structure.size()) + "-layers-";
				fileName += layers;
				fileName += "-" + std::to_string(epoch_time);
				ofs2 << "file," << fileName << std::endl;
				std::cout << fileName << std::endl;
				double err;
				int n;
				std::tie(err, n) = singleRun(structure, init_val, fileName);
				ofs2 << "error," << err << ",,learning time," << n << std::endl;
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

void Softmaxtest(std::vector<int> const& structure, std::ostream& out)
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

auto squared = [](const double x)
{
	return x * x;
};
auto cross = [](const double x)
{
	double y = log(x);
	return y;
};

double errorFunc(Eigen::VectorXd const& outData, Eigen::VectorXd const& teachData)
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

Eigen::MatrixXd calcDelta(int layerNo, std::vector<Eigen::VectorXd> const& output, Eigen::MatrixXd const& prevDelta)
{
	Eigen::VectorXd diff = differential(output[layerNo + 1]);
	Eigen::MatrixXd delta = (prevDelta * weights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void backpropergation(std::vector<int> const& structure, std::vector<Eigen::VectorXd> const& output, Eigen::VectorXd const& teachData)
{
	Eigen::VectorXd diff = outputDifferential(output[structure.size() - 1]);
	Eigen::MatrixXd delta = (output[structure.size() - 1] - teachData).transpose().array() * diff.transpose().array();
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

double validate(std::vector<int> const& structure, bool show)
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

	//	if (show && typeid(dataSet) == typeid(FuncApproxDataSet))
	//	{
	//		dataSet.update(outs);
	//	}
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
	//	return error / dataSet.testDataSet.rows();
}

double learnProccess(std::vector<int> const& structure, int iterator, Eigen::VectorXd const& input, Eigen::VectorXd const& teachData, std::ostream& out)
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


	if (typeid(dataSet) == typeid(MNISTDataSet))
	{
		double error = errorFunc(outputs[structure.size() - 1], teachData);
		return error;
	}
	else
	{
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
}

Eigen::MatrixXd pretrainDelta(std::vector<Eigen::MatrixXd>& AEweights, int layerNo, std::vector<Eigen::VectorXd> const& output, Eigen::MatrixXd const& prevDelta)
{
	Eigen::VectorXd diff = differential(output[layerNo + 1]);
	Eigen::MatrixXd delta = (prevDelta * AEweights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void pretrainBP(std::vector<int> const& structure, std::vector<Eigen::MatrixXd>& AEweights, std::vector<Eigen::VectorXd>& AEbiases, std::vector<Eigen::VectorXd> output, Eigen::VectorXd teachData)
{
	Eigen::VectorXd diff = differential(output[structure.size() - 1]);
	Eigen::MatrixXd delta = (output[structure.size() - 1] - teachData).transpose().array() * diff.transpose().array();
	AEweights[structure.size() - 2] -= LEARNING_RATE * output[structure.size() - 2] * delta;
	AEbiases[structure.size() - 2] -= LEARNING_RATE * delta.transpose();
	for (int i = 3; i <= structure.size(); ++i)
	{
		int n = structure.size() - i;
		delta = pretrainDelta(AEweights, n, output, delta);
		AEweights[n] -= LEARNING_RATE * output[n] * delta;
		AEbiases[n] -= LEARNING_RATE * delta.transpose();
	}
}

double pretrainValidate(std::vector<int> const& structure, std::vector<Eigen::MatrixXd>& AEweights, std::vector<Eigen::VectorXd>& AEbiases, Eigen::MatrixXd& inputData)
{
	double error = 0.0;
	Eigen::VectorXd outs = Eigen::VectorXd::Zero(inputData.rows());
	std::mutex mtx;

	Concurrency::parallel_for<int>(0, inputData.rows(), 1, [&error, &outs, &mtx, inputData, structure, AEweights, AEbiases](int i)
                               {
	                               //	feedforward proccess
	                               std::vector<Eigen::VectorXd> outputs;
	                               Eigen::VectorXd input = inputData.row(i);
	                               outputs.push_back(input.transpose());

	                               for (int j = 0; j < structure.size() - 1; j++)
	                               {
		                               Eigen::VectorXd inputs = (outputs[j].transpose() * AEweights[j] + AEbiases[j].transpose());
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
	                               error += errorFunc(outputs[structure.size() - 1], input);
                               });
	return error;
}

void pretrainProccess(std::vector<int> const& structure, std::vector<Eigen::MatrixXd>& AEweights, std::vector<Eigen::VectorXd>& AEbiases, Eigen::VectorXd input)
{
	//	feedforward proccess
	std::vector<Eigen::VectorXd> outputs;
	outputs.push_back(input);

	for (int i = 0; i < structure.size() - 1; i++)
	{
		Eigen::VectorXd inputs = (outputs[i].transpose() * AEweights[i] + AEbiases[i].transpose());
		Eigen::VectorXd output = activationFunc(inputs);
		outputs.push_back(output);
	}
	//	backpropergation method
	pretrainBP(structure, AEweights, AEbiases, outputs, input);
}

void pretrain(std::vector<int> const& structure, std::ostream& out)
{
	cout << "autoencoder" << endl;
	out << "autoencoder" << endl;
	Eigen::MatrixXd inputData = dataSet.dataSet;
	Eigen::MatrixXd middleData;
	for (int i = 0; i < structure.size() - 2; ++i)
	{
		//init autoencoder
		std::vector<int> AEstructure = {structure[i],structure[i + 1],structure[i]};
		std::vector<Eigen::MatrixXd> AEweights = {weights[i], weights[i].transpose()};
		std::vector<Eigen::VectorXd> AEbiases = {biases[i], Eigen::VectorXd::Zero(AEstructure[2])};

		//for-loop-learing
		double error = 1.0;
		for (int j = 0; j < PRETRAIN_LEARNING_TIME; ++j)
		{
			std::vector<int> ns(inputData.rows());
			iota(ns.begin(), ns.end(), 0);
			shuffle(ns.begin(), ns.end(), std::mt19937());
			for (int n : ns)
			{
				VectorXd input = inputData.row(n);
				pretrainProccess(AEstructure, AEweights, AEbiases, input);
				error = pretrainValidate(AEstructure, AEweights, AEbiases, inputData);
				std::cout << "error: " << error << "\r" << std::flush;
			}
			if (error < PRETRAIN_ERROR_BOTTOM)
			{
				out << i << "," << j << endl;
				break;
			}
		}
		cout << endl << i << endl;
		//pouring middleData
		middleData.resize(inputData.rows(), structure[i + 1]);
		for (int k = 0; k < inputData.rows(); ++k)
		{
			VectorXd inptVctr = AEweights[0].transpose() * inputData.row(k).transpose() + AEbiases[0];
			middleData.row(k) = activationFunc(inptVctr).transpose();
		}

		//move middleData to inputData
		inputData = middleData;
		weights[i] = AEweights[0];
		biases[i] = AEbiases[0];
	}
}
