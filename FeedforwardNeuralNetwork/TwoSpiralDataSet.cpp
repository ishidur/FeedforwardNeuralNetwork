#include "stdafx.h"
#include "TwoSpiralDataSet.h"


TwoSpiralDataSet::TwoSpiralDataSet()
{
}


void TwoSpiralDataSet::load()
{
	int points = 96;
	int density = 2;
	double maxDiameter = 1.0;
	int dataNum = points * density;
	dataSet.resize(dataNum, 2);
	teachSet.resize(dataNum, 1);
	for (int i = 0; i < density; ++i)
	{
		for (int j = 0; j < points; ++j)
		{
			const double angle = j * M_PI / (16.0 * density);
			const double radius = maxDiameter * (104 * density - i) / (104 * density);
			const double x = radius * cos(angle);
			const double y = radius * sin(angle);
			dataSet.row(i * points + j) << x , y;

			teachSet.row(i * points + j) << i;
		}
	}
	testDataSet = dataSet;
	testTeachSet = teachSet;
}
