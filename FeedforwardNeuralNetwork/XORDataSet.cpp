#include "stdafx.h"
#include "XORDataSet.h"


XORDataSet::XORDataSet() {
    this->useSoftmax = false;
}

void XORDataSet::load()
{
	this->inputSet.resize(4, 2);
	this->teachSet.resize(4, 1);
	this->inputSet << 0, 0,
		0, 1,
		1, 0,
		1, 1;
	this->teachSet << 0,
		1,
		1,
		0;
	this->testInputSet = this->inputSet;
	this->testTeachSet = this->teachSet;
}
