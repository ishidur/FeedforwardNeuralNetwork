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

#include "config.h"

#include <numeric>
using namespace std;
using namespace Eigen;


vector<MatrixXd> weights;
vector<VectorXd> biases;

void initWeightsAndBiases(vector<int> const& structure, double iniitalVal)
{
	weights.clear();
	biases.clear();
	for (int i = 1; i < structure.size(); ++i)
	{
		weights.push_back(MatrixXd::Random(structure[i - 1], structure[i]) * iniitalVal);
		biases.push_back(VectorXd::Random(structure[i]) * iniitalVal);
	}
}

void Softmaxtest(vector<int> const& structure, ostream& out = cout);
double learnProccess(vector<int> const& structure, int iterator, VectorXd const& input, VectorXd const& teachData, ostream& out = cout);
void pretrain(vector<int> const& structure, ostream& out = cout);

tuple<double, int> singleRun(vector<int> const& structure, double const& initVal, string filename)
{
	initWeightsAndBiases(structure, initVal);
	if (needPretrain)
	{
		//pretraining process
		ofstream preofs(filename + "-ae" + ".csv");
		pretrain(structure, preofs);
		preofs.close();
	}
	//	int a[dataSet.dataSet.rows()] = {0};
	ofstream ofs(filename + ".csv");
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
	ofs << "error" << endl;
	//	std::string progress = "";
	double error = 1.0;

	int s = 0;
	string progress = "";
	int times = LEARNING_TIME;
	vector<int> ns(dataSet.dataSet.rows());
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
		shuffle(ns.begin(), ns.end(), mt19937());
		for (int n : ns)
		{
			s++;
			double status = double((s) * 100.0 / (ns.size() * LEARNING_TIME));
			if (progress.size() < int(status) / 5)
			{
				progress += "#";
			}

			VectorXd input = dataSet.dataSet.row(n);
			VectorXd teach = dataSet.teachSet.row(n);
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
			if (std::isnan(error))
			{
				goto learn_end;
			}
			cout << "progress: " << error << ", " << s << "/ " << (ns.size() * LEARNING_TIME) << " " << progress << "\r" << flush;
		}
	}
learn_end:
	//	ofs << "middleData" << endl;
	//	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	//	{
	//		//	feedforward proccess
	//		Eigen::VectorXd output = dataSet.testDataSet.row(i).transpose();
	//		for (int j = 0; j < structure.size() - 2; j++)
	//		{
	//			Eigen::VectorXd inputs = (output.transpose() * weights[j] + biases[j].transpose());
	//			output = activationFunc(inputs);
	//		}
	//
	//		int sea = output.size();
	//		for (int l = 0; l < sea; ++l)
	//		{
	//			ofs << output[l];
	//			if (l < sea - 1)
	//			{
	//				ofs << ",";
	//			}
	//		}
	//		ofs << endl;
	//	}
	ofs.close();
	if (dataSet.useSoftmax)
	{
		ofstream ofs2(filename + "-test" + ".csv");
		Softmaxtest(structure, ofs2);
		ofs2.close();
	}
	cout << endl;
	return forward_as_tuple(error, times);
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
		string dirName = "data\\";
		ostringstream sout;
		sout << fixed << init_val;
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
			ofs2 << "structures," << layers << endl;
			cout << "structures; " << layers << endl;
			int correct = 0;
			for (int i = 0; i < TRIALS_PER_STRUCTURE; ++i)
			{
				time_t epoch_time;
				epoch_time = time(nullptr);
				ofs2 << "try," << i << endl;
				cout << "try: " << i << endl;
				string fileName = dirName;
				fileName += "result-";
				fileName += to_string(structure.size()) + "-layers-";
				fileName += layers;
				fileName += "-" + to_string(epoch_time);
				ofs2 << "file," << fileName << endl;
				cout << fileName << endl;
				double err;
				int n;
				tie(err, n) = singleRun(structure, init_val, fileName);
				ofs2 << "error," << err << ",,learning time," << n << endl;
				cout << "error; " << err << endl;
				if (err < ERROR_BOTTOM)
				{
					correct++;
				}
			}
			ofs2 << correct << ", /, " << TRIALS_PER_STRUCTURE << ", success" << endl;
			cout << correct << " / " << TRIALS_PER_STRUCTURE << " success" << endl;
		}
		ofs2.close();
	}
	return 0;
}

void Softmaxtest(vector<int> const& structure, ostream& out)
{
	int correct[10] = {0};
	int num[10] = {0};
	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	{
		//	feedforward proccess
		vector<VectorXd> outputs;
		outputs.push_back(dataSet.testDataSet.row(i).transpose());

		for (int j = 0; j < structure.size() - 1; j++)
		{
			VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
			VectorXd output;
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
		vector<double> y;
		y.resize(outputs[structure.size() - 1].size());
		VectorXd::Map(&y[0], outputs[structure.size() - 1].size()) = outputs[structure.size() - 1];
		vector<double>::iterator result = max_element(y.begin(), y.end());
		vector<double> t;
		t.resize(dataSet.testTeachSet.row(i).size());
		VectorXd::Map(&t[0], dataSet.testTeachSet.row(i).size()) = dataSet.testTeachSet.row(i);
		vector<double>::iterator teach = max_element(t.begin(), t.end());
		//		out << "input, " << std::endl;
		//		out << dataSet.testDataSet.row(i) << std::endl;
		if (i == 0)
		{
			out << outputs[structure.size() - 1].transpose() << endl;
		}
		out << "answer, " << "output" << endl;
		out << distance(y.begin(), result) << ", " << distance(t.begin(), teach) << endl;
		cout << distance(y.begin(), result) << ", " << distance(t.begin(), teach) << endl;
		int a = distance(t.begin(), teach);
		num[a]++;
		if (distance(y.begin(), result) == a)
		{
			correct[a]++;
		}
	}
	for (int i = 0; i < 10; ++i)
	{
		out << endl << i << "correct, " << correct[i] << ", /, " << num[i] << endl;
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

double errorFunc(VectorXd const& outData, VectorXd const& teachData)
{
	double error;
	if (dataSet.useSoftmax)
	{
		//		Cross Entropy
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

MatrixXd calcDelta(int layerNo, vector<VectorXd> const& output, MatrixXd const& prevDelta)
{
	VectorXd diff = differential(output[layerNo + 1]);
	MatrixXd delta = (prevDelta * weights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void backpropergation(vector<int> const& structure, vector<VectorXd> const& output, VectorXd const& teachData)
{
	VectorXd diff = outputDifferential(output[structure.size() - 1]);
	MatrixXd delta = (output[structure.size() - 1] - teachData).transpose().array() * diff.transpose().array();

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

double validate(vector<int> const& structure, bool show)
{
	double error = 0.0;
	VectorXd outs = VectorXd::Zero(dataSet.testDataSet.rows());
	mutex mtx;

	Concurrency::parallel_for<int>(0, dataSet.testDataSet.rows(), 1, [&error, &outs, &mtx, structure](int i)
                               {
	                               //	feedforward proccess
	                               vector<VectorXd> outputs;
	                               outputs.push_back(dataSet.testDataSet.row(i).transpose());

	                               for (int j = 0; j < structure.size() - 1; j++)
	                               {
		                               VectorXd inputs = (outputs[j].transpose() * weights[j] + biases[j].transpose());
		                               VectorXd output;
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
	                               VectorXd teach = dataSet.testTeachSet.row(i);
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
	//	return error;
	return error / dataSet.testDataSet.rows();
}

double learnProccess(vector<int> const& structure, int iterator, VectorXd const& input, VectorXd const& teachData, ostream& out)
{
	//	feedforward proccess
	vector<VectorXd> outputs;
	outputs.push_back(input);

	for (int i = 0; i < structure.size() - 1; i++)
	{
		VectorXd inputs = (outputs[i].transpose() * weights[i] + biases[i].transpose());
		VectorXd output;
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
	double error = validate(structure, iterator % 100 == 0);
	if (&out != &cout)
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
		out << error << endl;
	}
	return error;
}

MatrixXd pretrainDelta(vector<MatrixXd>& AEweights, int layerNo, vector<VectorXd> const& output, MatrixXd const& prevDelta)
{
	VectorXd diff = differential(output[layerNo + 1]);
	MatrixXd delta = (prevDelta * AEweights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void pretrainBP(vector<int> const& structure, vector<MatrixXd>& AEweights, vector<VectorXd>& AEbiases, vector<VectorXd> output, VectorXd teachData)
{
	VectorXd diff = differential(output[structure.size() - 1]);
	MatrixXd delta = (output[structure.size() - 1] - teachData).transpose().array() * diff.transpose().array();
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

double pretrainValidate(vector<int> const& structure, vector<MatrixXd>& AEweights, vector<VectorXd>& AEbiases, MatrixXd& inputData)
{
	double error = 0.0;
	VectorXd outs = VectorXd::Zero(inputData.rows());
	mutex mtx;

	Concurrency::parallel_for<int>(0, inputData.rows(), 1, [&error, &outs, &mtx, inputData, structure, AEweights, AEbiases](int i)
                               {
	                               //	feedforward proccess
	                               vector<VectorXd> outputs;
	                               VectorXd input = inputData.row(i);
	                               outputs.push_back(input.transpose());

	                               for (int j = 0; j < structure.size() - 1; j++)
	                               {
		                               VectorXd inputs = (outputs[j].transpose() * AEweights[j] + AEbiases[j].transpose());
		                               VectorXd output;
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
	return error / inputData.rows();
	//	return error;
}

void pretrainProccess(vector<int> const& structure, vector<MatrixXd>& AEweights, vector<VectorXd>& AEbiases, VectorXd input)
{
	//	feedforward proccess
	vector<VectorXd> outputs;
	outputs.push_back(input);

	for (int i = 0; i < structure.size() - 1; i++)
	{
		VectorXd inputs = (outputs[i].transpose() * AEweights[i] + AEbiases[i].transpose());
		VectorXd output = activationFunc(inputs);
		outputs.push_back(output);
	}
	//	backpropergation method
	pretrainBP(structure, AEweights, AEbiases, outputs, input);
}

void pretrain(vector<int> const& structure, ostream& out)
{
	cout << "autoencoder" << endl;
	out << "autoencoder" << endl;
	MatrixXd inputData = dataSet.dataSet;
	MatrixXd middleData;
	for (int k = 0; k < inputData.rows(); ++k)
	{
		out << "inputData" << endl;
		for (int l = 0; l < inputData.cols(); ++l)
		{
			out << inputData(k, l) << ",";
		}
		out << endl;
	}
	for (int i = 0; i < structure.size() - 2; ++i)
	{
		//init autoencoder
		vector<int> AEstructure = {structure[i],structure[i + 1],structure[i]};
		vector<MatrixXd> AEweights = {weights[i], weights[i].transpose()};
		vector<VectorXd> AEbiases = {biases[i], VectorXd::Zero(AEstructure[2])};
		int inputDataSize = inputData.rows();
		//for-loop-learing
		double error = 1.0;
		for (int j = 0; j < PRETRAIN_LEARNING_TIME; ++j)
		{
			vector<int> ns(inputDataSize);
			iota(ns.begin(), ns.end(), 0);
			shuffle(ns.begin(), ns.end(), mt19937());
			for (int n : ns)
			{
				VectorXd input = inputData.row(n);
				pretrainProccess(AEstructure, AEweights, AEbiases, input);
				error = pretrainValidate(AEstructure, AEweights, AEbiases, inputData);
				cout << "error: " << error << "\r" << flush;
			}
			if (error < PRETRAIN_ERROR_BOTTOM)
			{
				out << i << "," << j << endl;
				break;
			}
		}
		cout << endl << i << endl;
		//pouring middleData
		middleData.resize(inputDataSize, structure[i + 1]);
		out << "middleData" << endl;
		for (int k = 0; k < inputDataSize; ++k)
		{
			VectorXd inptVctr = AEweights[0].transpose() * inputData.row(k).transpose() + AEbiases[0];
			middleData.row(k) = activationFunc(inptVctr).transpose();
			int cols = middleData.cols();
			for (int l = 0; l < middleData.cols(); ++l)
			{
				out << middleData(k, l);

				if (l < cols - 1)
				{
					out << ",";
				}
			}
			out << endl;
		}

		//move middleData to inputData
		inputData = middleData;
		weights[i] = AEweights[0];
		biases[i] = AEbiases[0];
	}
}
