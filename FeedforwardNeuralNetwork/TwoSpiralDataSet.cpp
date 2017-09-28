#include "stdafx.h"
#include "TwoSpiralDataSet.h"
#include <fstream>

#define POINTS 96
#define LAP 3
#define MINDIAMETER 0.1
#define MAXDIAMETER 1.0

TwoSpiralDataSet::TwoSpiralDataSet()
{
}


void TwoSpiralDataSet::load()
{
	double radiusGrowth = (MAXDIAMETER - MINDIAMETER) / 2.0 / double(POINTS);
	double angleGrowth = 2.0 * M_PI / double(POINTS) * double(LAP);
	int dataNum = POINTS * 2;
	dataSet.resize(dataNum, 2);
	teachSet.resize(dataNum, 1);
	std::ofstream ofs("TwoSpiralData.csv");
	ofs << "x, y, val" << std::endl;
	for (int i = 0; i < POINTS; ++i)
	{
		const double x = (radiusGrowth * i + MINDIAMETER) * cos(angleGrowth * i) + MAXDIAMETER / 2.0;
		const double y = (radiusGrowth * i + MINDIAMETER) * sin(angleGrowth * i) + MAXDIAMETER / 2.0;
		ofs << x << ", " << y << ", " << 1 << std::endl;
		ofs << -x << ", " << -y << ", " << 0 << std::endl;
		dataSet.row(2 * i) << x , y;
		dataSet.row(2 * i + 1) << -x , -y;
		teachSet.row(2 * i) << 1;
		teachSet.row(2 * i + 1) << 0;
	}
	testDataSet = dataSet;
	testTeachSet = teachSet;
}
