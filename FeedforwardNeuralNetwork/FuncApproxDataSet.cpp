#include "stdafx.h"
#include "FuncApproxDataSet.h"
#include <fstream>

#define TRAIN_DATA_NUM 10
#define TEST_DATA_NUM 20
//y=cos(x/2)sin(8x)

auto func = [](double x)
{
	return sin(2.0 * M_PI * x);
	//	return sin(x / 2.0)*cos(x*2.0);
	//	return cos(x / 2.0) * sin(8.0 * x);
};

FuncApproxDataSet::FuncApproxDataSet() {
    this->useSoftmax = false;
}

void FuncApproxDataSet::load()
{
	this->inputSet.resize(TRAIN_DATA_NUM, 1);
	this->teachSet.resize(TRAIN_DATA_NUM, 1);
	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(TRAIN_DATA_NUM, 0.0, 1.0);
	this->inputSet.col(0) = x;
	this->teachSet.col(0) = x.unaryExpr(func);
	this->testInputSet.resize(TEST_DATA_NUM, 1);
	this->testTeachSet.resize(TEST_DATA_NUM, 1);
	Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(TEST_DATA_NUM, 0.0, 1.0);
	this->testInputSet.col(0) = t;
	this->testTeachSet.col(0) = t.unaryExpr(func);
	std::ofstream ofs("FunctionData.csv");
	ofs << "x, y" << std::endl;
	for (int i = 0; i < TRAIN_DATA_NUM; ++i) { ofs << this->inputSet.row(i) << ", " << this->teachSet.row(i) << std::endl; }
	ofs.close();
}

FILE* fp = _popen("gnuplot", "w");

void FuncApproxDataSet::show()
{
	fprintf(fp, "set multiplot\n");
	fprintf(fp, "set xrange [%f:%f]\n", 0.0, 1.0); // ”ÍˆÍ‚ÌŽw’è
	fprintf(fp, "set yrange [%f:%f]\n", -2.0, 2.0); // ”ÍˆÍ‚ÌŽw’è
	fprintf(fp, "set xlabel \"x\"\n");
	fprintf(fp, "set ylabel \"y\"\n");
	fprintf(fp, "plot '-' with lines linetype 1\n");
	for (int i = 0; i < TEST_DATA_NUM; ++i) { fprintf(fp, "%f\t%f\n", testInputSet.col(0)[i], testTeachSet.col(0)[i]); }
	fprintf(fp, "e\n");
	fprintf(fp, "plot '-' with points pointtype 1\n");
	for (int i = 0; i < TRAIN_DATA_NUM; ++i) { fprintf(fp, "%f\t%f\n", inputSet.col(0)[i], teachSet.col(0)[i]); }
	fprintf(fp, "e\n");
	fprintf(fp, "unset multiplot\n");
	fflush(fp);
}

void FuncApproxDataSet::update(Eigen::VectorXd outputs)
{
	fprintf(fp, "clear\n");
	fprintf(fp, "set multiplot\n");
	fprintf(fp, "plot '-' with lines linetype 1\n");
	for (int i = 0; i < TEST_DATA_NUM; ++i) { fprintf(fp, "%f\t%f\n", testInputSet.col(0)[i], testTeachSet.col(0)[i]); }
	fprintf(fp, "e\n");
	fprintf(fp, "plot '-' with points pointtype 1\n");
	for (int i = 0; i < TEST_DATA_NUM; ++i) { fprintf(fp, "%f\t%f\n", testInputSet.col(0)[i], outputs[i]); }
	fprintf(fp, "e\n");
	fprintf(fp, "unset multiplot\n");
	fflush(fp);
}
