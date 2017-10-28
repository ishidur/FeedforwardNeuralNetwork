#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

class MNISTDataSet
{
public:
	bool useSoftmax = true;
	Eigen::MatrixXd dataSet;
	Eigen::MatrixXd teachSet;
	Eigen::MatrixXd testDataSet;
	Eigen::MatrixXd testTeachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	MNISTDataSet();
	void load();
};
