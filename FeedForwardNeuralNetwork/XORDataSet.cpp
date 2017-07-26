#include "stdafx.h"
#include "XORDataSet.h"


XORDataSet::XORDataSet()
{
}

void XORDataSet::load()
{
	dataSet.resize(4, 2);
	teachSet.resize(4, 1);
	dataSet << 0, 0,
		0, 1,
		1, 0,
		1, 1;
	teachSet << 0,
		1,
		1,
		0;
	testDataSet = dataSet;
	testTeachSet = teachSet;
}


