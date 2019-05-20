#include "stdafx.h"
#include "NeuralNetwork.h"
#include <numeric>

NeuralNetwork::NeuralNetwork(std::vector<int> const& structure, double inital_val, auto const& dataSet)
{
    this->dataSet = dataSet;
    this->structure = structure;
    this->weights.clear();
	this->biases.clear();
	for (int i = 1; i < structure.size(); ++i)
	{
		this->weights.push_back(Eigen::MatrixXd::Random(structure[i - 1], structure[i]) * inital_val);
		this->biases.push_back(Eigen::VectorXd::Random(structure[i]) * inital_val);
	}
}

/**
 * \brief learn single network
 * \param structure 
 * \param initVal 
 * \param filename 
 * \return learning results
 */
std::tuple<double, int> NeuralNetwork::learn(std::string filename)
{
	//	int a[dataSet.dataSet.rows()] = {0};
    std::ofstream ofs(filename + ".csv");
	ofs << "step,";
	for (int i = 0; i < this->structure.size() - 1; ++i)
	{
		for (int j = 0; j < this->weights[i].rows(); ++j)
		{
			for (int k = 0; k < this->weights[i].cols(); ++k) { ofs << "weight:" << "l:" << i << ":" << j << ":" << k << ", "; }
		}
		for (int j = 0; j < this->biases[i].size(); ++j) { ofs << "bias:" << "l:" << i << ":" << j << ", "; }
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
			if (progress.size() < int(status) / 5) { progress += "#"; }

            Eigen::VectorXd input = dataSet.dataSet.row(n);
            Eigen::VectorXd teach = dataSet.teachSet.row(n);
			if (s % c == 0)
			{
				ofs << i << ",";
				error = this->learn_proccess(input, teach, ofs);
				if (error < ERROR_BOTTOM)
				{
					times = i;
					goto learn_end;
				}
			}
			else { error = this->learn_proccess(input, teach); }
			if (std::isnan(error)) { goto learn_end; }
            std::cout << "progress: " << error << ", " << s << "/ " << (ns.size() * LEARNING_TIME) << " " << progress << "\r" <<
                std::flush;
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
        std::ofstream ofs2(filename + "-test" + ".csv");
		this->softmax_test(ofs2);
		ofs2.close();
	}
    std::cout << std::endl;
	return std::forward_as_tuple(error, times);
}

//TODO: didn't work

void NeuralNetwork::softmax_test(std::ostream& out)
{
	int correct[10] = {0};
	int num[10] = {0};
	for (int i = 0; i < dataSet.testDataSet.rows(); ++i)
	{
		//	feedforward proccess
        std::vector<Eigen::VectorXd> outputs;
		outputs.push_back(dataSet.testDataSet.row(i).transpose());

		for (int j = 0; j < this->structure.size() - 1; j++)
		{
            Eigen::VectorXd inputs = (outputs[j].transpose() * this->weights[j] + this->biases[j].transpose());
            Eigen::VectorXd output;
			if (j == this->structure.size() - 2) { output = outputActivationFunc(inputs); }
			else { output = activationFunc(inputs); }
			outputs.push_back(output);
		}
        std::vector<double> y;
		y.resize(outputs[this->structure.size() - 1].size());
        Eigen::VectorXd::Map(&y[0], outputs[this->structure.size() - 1].size()) = outputs[this->structure.size() - 1];
        std::vector<double>::iterator result = max_element(y.begin(), y.end());
        std::vector<double> t;
		t.resize(dataSet.testTeachSet.row(i).size());
        Eigen::VectorXd::Map(&t[0], dataSet.testTeachSet.row(i).size()) = dataSet.testTeachSet.row(i);
        std::vector<double>::iterator teach = max_element(t.begin(), t.end());
		//		out << "input, " << std::endl;
		//		out << dataSet.testDataSet.row(i) << std::endl;
		if (i == 0)
		{
			out << outputs[this->structure.size() - 1].transpose() << std::endl;
		}
		out << "answer, " << "output" << std::endl;
		out << distance(y.begin(), result) << ", " << distance(t.begin(), teach) << std::endl;
        std::cout << distance(y.begin(), result) << ", " << distance(t.begin(), teach) << std::endl;
		int a = distance(t.begin(), teach);
		num[a]++;
		if (distance(y.begin(), result) == a) { correct[a]++; }
	}
	for (int i = 0; i < 10; ++i) { out << std::endl << i << "correct, " << correct[i] << ", /, " << num[i] << std::endl; }
}

auto squared = [](const double x) { return x * x; };
auto cross = [](const double x) { return log(x); };

double error_func(Eigen::VectorXd const& outData, Eigen::VectorXd const& teachData)
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

Eigen::MatrixXd NeuralNetwork::calc_delta(int layerNo, std::vector<Eigen::VectorXd> const& output, Eigen::MatrixXd const& prevDelta)
{
    Eigen::VectorXd diff = differential(output[layerNo + 1]);
    Eigen::MatrixXd delta = (prevDelta * this->weights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void NeuralNetwork::back_propagation(std::vector<Eigen::VectorXd> const& output, Eigen::VectorXd const& teachData)
{
    Eigen::VectorXd diff = outputDifferential(output[this->structure.size() - 1]);
    Eigen::MatrixXd delta = (output[this->structure.size() - 1] - teachData).transpose().array() * diff.transpose().array();

	this->weights[this->structure.size() - 2] -= LEARNING_RATE * output[this->structure.size() - 2] * delta;
	this->biases[this->structure.size() - 2] -= LEARNING_RATE * delta.transpose();
	for (int i = 3; i <= this->structure.size(); ++i)
	{
		int n = this->structure.size() - i;
		delta = this->calc_delta(n, output, delta);
		this->weights[n] -= LEARNING_RATE * output[n] * delta;
		this->biases[n] -= LEARNING_RATE * delta.transpose();
	}
}

double NeuralNetwork::validate()
{
	double error = 0.0;
    Eigen::VectorXd outs = Eigen::VectorXd::Zero(dataSet.testDataSet.rows());
    std::mutex mtx;

	Concurrency::parallel_for<int>(0, dataSet.testDataSet.rows(), 1, [&error, &outs, &mtx, this](int i)
	{
		//	feedforward proccess
        std::vector<Eigen::VectorXd> outputs;
		outputs.push_back(dataSet.testDataSet.row(i).transpose());

		for (int j = 0; j < this->structure.size() - 1; j++)
		{
            Eigen::VectorXd inputs = (outputs[j].transpose() * this->weights[j] + this->biases[j].transpose());
            Eigen::VectorXd output;
			if (j == this->structure.size() - 2) { output = outputActivationFunc(inputs); }
			else { output = activationFunc(inputs); }
			outputs.push_back(output);
		}
		mtx.lock();
		outs[i] = outputs[this->structure.size() - 1].sum();
		mtx.unlock();
        Eigen::VectorXd teach = dataSet.testTeachSet.row(i);
		error += error_func(outputs[this->structure.size() - 1], teach);
	});

	return error / dataSet.testDataSet.rows();
}

double NeuralNetwork::learn_proccess(Eigen::VectorXd const& input, Eigen::VectorXd const& teachData,
                                     std::ostream& out)
{
	//	feedforward proccess
    std::vector<Eigen::VectorXd> outputs;
	outputs.push_back(input);

	for (int i = 0; i < this->structure.size() - 1; i++)
	{
        Eigen::VectorXd inputs = (outputs[i].transpose() * this->weights[i] + this->biases[i].transpose());
        Eigen::VectorXd output;
		if (i == this->structure.size() - 2) { output = outputActivationFunc(inputs); }
		else { output = activationFunc(inputs); }
		outputs.push_back(output);
	}
	//	backpropagation method
	this->back_propagation(outputs, teachData);


	if (typeid(dataSet) == typeid(MNISTDataSet))
	{
		double error = error_func(outputs[this->structure.size() - 1], teachData);
		return error;
	}
	if (&out == &std::cout)
	{
		double error = error_func(outputs[this->structure.size() - 1], teachData);
		return error;
	}

	double error = this->validate();
	for (int i = 0; i < this->structure.size() - 1; ++i)
	{
		for (int j = 0; j < this->weights[i].rows(); ++j)
		{
			for (int k = 0; k < this->weights[i].cols(); ++k) { out << this->weights[i](j, k) << ", "; }
		}
		for (int j = 0; j < this->biases[i].size(); ++j) { out << this->biases[i][j] << ", "; }
	}
	out << error << std::endl;
	return error;
}

Eigen::MatrixXd pretrainDelta(std::vector<Eigen::MatrixXd>& AEweights, int layerNo, std::vector<Eigen::VectorXd> const& output,
                                             Eigen::MatrixXd const& prevDelta)
{
    Eigen::VectorXd diff = differential(output[layerNo + 1]);
    Eigen::MatrixXd delta = (prevDelta * AEweights[layerNo + 1].transpose()).array() * diff.transpose().array();
	return delta;
}

void pretrainBP(std::vector<int>const & structure, std::vector<Eigen::MatrixXd>& AEweights, std::vector<Eigen::VectorXd>& AEbiases,
                               std::vector<Eigen::VectorXd> output, Eigen::VectorXd teachData)
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

double pretrain_validate(std::vector<int>const & structure, std::vector<Eigen::MatrixXd>& AEweights, std::vector<Eigen::VectorXd>& AEbiases,
                                        Eigen::MatrixXd& inputData)
{
	double error = 0.0;
    Eigen::VectorXd outs = Eigen::VectorXd::Zero(inputData.rows());
    std::mutex mtx;

	Concurrency::parallel_for<int>(0, inputData.rows(), 1,
	                               [&error, &outs, &mtx, inputData, structure, AEweights, AEbiases](int i)
	                               {
		                               //	feedforward proccess
                                       std::vector<Eigen::VectorXd> outputs;
                                       Eigen::VectorXd input = inputData.row(i);
		                               outputs.push_back(input.transpose());

		                               for (int j = 0; j < structure.size() - 1; j++)
		                               {
                                           Eigen::VectorXd inputs = (outputs[j].transpose() * AEweights[j] + AEbiases[j].transpose());
                                           Eigen::VectorXd output;
			                               if (j == structure.size() - 2) { output = outputActivationFunc(inputs); }
			                               else { output = activationFunc(inputs); }
			                               outputs.push_back(output);
		                               }
		                               mtx.lock();
		                               outs[i] = outputs[structure.size() - 1].sum();
		                               mtx.unlock();
		                               error += error_func(outputs[structure.size() - 1], input);
	                               });
	return error / inputData.rows();
	//	return error;
}

void pretrain_proccess(std::vector<int>const & structure, std::vector<Eigen::MatrixXd>& AEweights, std::vector<Eigen::VectorXd>& AEbiases,
                                      Eigen::VectorXd input)
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
	//	backpropagation method
	pretrainBP(structure, AEweights, AEbiases, outputs, input);
}

void NeuralNetwork::pretrain(std::ostream& out)
{
    std::cout << "autoencoder" << std::endl;
	out << "autoencoder" << std::endl;
    Eigen::MatrixXd inputData = dataSet.dataSet;
    Eigen::MatrixXd middleData;
	for (int k = 0; k < inputData.rows(); ++k)
	{
		out << "inputData" << std::endl;
		for (int l = 0; l < inputData.cols(); ++l) { out << inputData(k, l) << ","; }
		out << std::endl;
	}
	for (int i = 0; i < this->structure.size() - 2; ++i)
	{
		//init autoencoder
        std::vector<int> AEstructure = {this->structure[i], this->structure[i + 1], this->structure[i]};
        std::vector<Eigen::MatrixXd> AEweights = {this->weights[i], this->weights[i].transpose()};
        std::vector<Eigen::VectorXd> AEbiases = {this->biases[i], Eigen::VectorXd::Zero(AEstructure[2])};
		int inputDataSize = inputData.rows();
		//for-loop-learing
		double error = 1.0;
		for (int j = 0; j < PRETRAIN_LEARNING_TIME; ++j)
		{
            std::vector<int> ns(inputDataSize);
			iota(ns.begin(), ns.end(), 0);
			shuffle(ns.begin(), ns.end(), std::mt19937());
			for (int n : ns)
			{
                Eigen::VectorXd input = inputData.row(n);
				pretrain_proccess(AEstructure, AEweights, AEbiases, input);
				error = pretrain_validate(AEstructure, AEweights, AEbiases, inputData);
                std::cout << "error: " << error << "\r" << std::flush;
			}
			if (error < PRETRAIN_ERROR_BOTTOM)
			{
				out << i << "," << j << std::endl;
				break;
			}
		}
        std::cout << std::endl << i << std::endl;
		//pouring middleData
		middleData.resize(inputDataSize, this->structure[i + 1]);
		out << "middleData" << std::endl;
		for (int k = 0; k < inputDataSize; ++k)
		{
            Eigen::VectorXd inptVctr = AEweights[0].transpose() * inputData.row(k).transpose() + AEbiases[0];
			middleData.row(k) = activationFunc(inptVctr).transpose();
			int cols = middleData.cols();
			for (int l = 0; l < middleData.cols(); ++l)
			{
				out << middleData(k, l);
				if (l < cols - 1) { out << ","; }
			}
			out << std::endl;
		}

		//move middleData to inputData
		inputData = middleData;
		this->weights[i] = AEweights[0];
		this->biases[i] = AEbiases[0];
	}
}


