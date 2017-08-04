#include "stdafx.h"
#include "FuncApproxDataSet.h"
#include <fstream>

#define TRAIN_DATA_NUM 100
#define TEST_DATA_NUM 1000
//y=cos(x/2)sin(8x)
FuncApproxDataSet::FuncApproxDataSet()
{
}


auto func = [](double x)
{
	return cos(x);
//	return sin(x / 2.0)*cos(x*2.0);
//	return cos(x / 2.0) * sin(8.0 * x);
};

void FuncApproxDataSet::load()
{
	dataSet.resize(TRAIN_DATA_NUM, 1);
	teachSet.resize(TRAIN_DATA_NUM, 1);
	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(TRAIN_DATA_NUM, -M_PI, M_PI);
	dataSet.col(0) = x;
	teachSet.col(0) = x.unaryExpr(func);
	testDataSet.resize(TEST_DATA_NUM, 1);
	testTeachSet.resize(TEST_DATA_NUM, 1);
	Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(TEST_DATA_NUM, -M_PI, M_PI);
	testDataSet.col(0) = t;
	testTeachSet.col(0) = t.unaryExpr(func);
	std::ofstream ofs("FunctionData.csv");
	ofs << "x, y" << std::endl;
	for (int i = 0; i < TRAIN_DATA_NUM; ++i)
	{
		ofs << dataSet.row(i) << ", " << teachSet.row(i) << std::endl;
	}
	ofs.close();
}

void FuncApproxDataSet::show()
{
	FILE* fp = _popen("gnuplot -persist", "w");
	fprintf(fp, "set multiplot\n");
	fprintf(fp, "set xrange [%f:%f]\n", -M_PI, M_PI);	// 範囲の指定
	fprintf(fp, "set xlabel \"x\"\n");
	fprintf(fp, "set ylabel \"y\"\n");
	fprintf(fp, "plot '-' with lines linetype 1\n");
	for (int i = 0; i < TEST_DATA_NUM; ++i) {
		fprintf(fp, "%f\t%f\n", testDataSet.col(0)[i], testTeachSet.col(0)[i]);
	}
	fprintf(fp, "e\n");
	fprintf(fp, "plot '-' with points pointtype 1\n");
	for (int i = 0; i < TRAIN_DATA_NUM; ++i) {
		fprintf(fp, "%f\t%f\n", dataSet.col(0)[i], teachSet.col(0)[i]);
	}
	fprintf(fp, "e\n");

	fprintf(fp, "set nomultiplot\n"); // マルチプロットモード終了
	fprintf(fp, "exit\n"); // gnuplotの終了
	fflush(fp);
	_pclose(fp);
}
