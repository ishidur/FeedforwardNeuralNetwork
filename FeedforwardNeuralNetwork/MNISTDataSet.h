#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

class MNISTDataSet
{
public:
	MatrixXd dataSet;
	MatrixXd teachSet;
	MatrixXd testDataSet;
	MatrixXd testTeachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	MNISTDataSet();
};

