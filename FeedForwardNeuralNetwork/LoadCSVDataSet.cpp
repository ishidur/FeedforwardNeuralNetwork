#include "stdafx.h"
#include "LoadCSVDataSet.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define SET_PATERNS 9999
#define INPUT_DIM 4
#define OUTPUT_DIM 1
#define importFileName "normalizedTrainingSet10000Eps10.0.csv"


LoadCSVDataSet::LoadCSVDataSet()
{
}

void LoadCSVDataSet::load()
{
	dataSet.resize(SET_PATERNS, INPUT_DIM);
	teachSet.resize(SET_PATERNS, OUTPUT_DIM);

	//�t�@�C���̓ǂݍ���
	ifstream ifs(importFileName);
	if (!ifs) {
		cout << "���̓G���[";
	}

	//csv�t�@�C����1�s���ǂݍ���
	string str;
	int lineCount = 0;

	while (getline(ifs, str)) {
		int rowCount = 0;
		string token;
		istringstream stream(str);

		//1�s�̂����A������ƃR���}�𕪊�����
		while (getline(stream, token, ',')) {
			//���ׂĕ�����Ƃ��ēǂݍ��܂�邽��
			//���l�͕ϊ����K�v
			double temp = stof(token); //stof(string str) : string��float�ɕϊ�

			if(rowCount<INPUT_DIM)
			{
				//cout << "I:" << temp;
				//cout << rowCount << endl;
				dataSet(lineCount,rowCount) = temp;
			}

			else
			{
				//cout << "O:" << temp;
				teachSet(lineCount, rowCount-INPUT_DIM) = temp;

			}
			rowCount++;
		}
		//cout << endl;
		//countColum++;
		lineCount++;
	}

	//cout << dataSet << endl;
	//cout << teachSet << endl;
	testDataSet = dataSet;
	testTeachSet = teachSet;
}