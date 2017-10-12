#include "stdafx.h"
#include "TwoSpiralDataSet.h"
#include <fstream>

#define POINTS 96
#define LAP 3
#define MINDIAMETER 0.01
#define MAXDIAMETER 1.0

TwoSpiralDataSet::TwoSpiralDataSet()
{
}

void TwoSpiralDataSet::load()
{
	double radiusGrowth = (MAXDIAMETER / 2.0 - MINDIAMETER) / double(POINTS);
	double angleGrowth = 2.0 * M_PI / double(POINTS) * double(LAP);
	int dataNum = POINTS * 2;
	dataSet.resize(dataNum, 2);
	teachSet.resize(dataNum, 1);
	std::ofstream ofs("TwoSpiralData.csv");
	ofs << "x, y, val" << std::endl;
	for (int i = 0; i < POINTS; ++i)
	{
		const double x = (radiusGrowth * i + MINDIAMETER) * cos(angleGrowth * i);
		const double y = (radiusGrowth * i + MINDIAMETER) * sin(angleGrowth * i);
		ofs << x + MAXDIAMETER / 2.0 << ", " << y + MAXDIAMETER / 2.0 << ", " << 1 << std::endl;
		ofs << -x + MAXDIAMETER / 2.0 << ", " << -y + MAXDIAMETER / 2.0 << ", " << 0 << std::endl;
		dataSet.row(2 * i) << x + MAXDIAMETER / 2.0, y + MAXDIAMETER / 2.0;
		dataSet.row(2 * i + 1) << -x + MAXDIAMETER / 2.0, -y + MAXDIAMETER / 2.0;
		teachSet.row(2 * i) << 1;
		teachSet.row(2 * i + 1) << 0;
	}
	testDataSet = dataSet;
	testTeachSet = teachSet;
}
