#include "stdafx.h"
#include "MNISTDataSet.h"

int reverseInt(int i)
{
	unsigned char c1 = i & 255;
	unsigned char c2 = (i >> 8) & 255;
	unsigned char c3 = (i >> 16) & 255;
	unsigned char c4 = (i >> 24) & 255;

	return (int(c1) << 24) + (int(c2) << 16) + (int(c3) << 8) + c4;
}

MatrixXd readImageFile(string filename)
{
	ifstream ifs(filename.c_str(), ios::in | ios::binary);
	int magic_number = 0;
	int number_of_images = 0;
	int rows = 0;
	int cols = 0;
	ifs.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	ifs.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
	rows = reverseInt(rows);
	ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
	cols = reverseInt(cols);

	MatrixXd images = MatrixXd::Zero(number_of_images, PIXEL);
	cout << magic_number << " " << number_of_images << " " << rows << " " << cols << endl;
	cout << "fetching image data..." << endl;
	string progress = "";

	for (int i = 0; i < number_of_images; i++)
	{
		double status = double(i * 100.0 / (number_of_images - 1));
		if (progress.size() < int(status) / 5)
		{
			progress += "#";
		}
		cout << "progress: " << setw(4) << right << fixed << setprecision(1) << (status) << "% " << progress << "\r" << flush;
		for (int row = 0; row < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				unsigned char temp = 0;
				ifs.read(reinterpret_cast<char*>(&temp), sizeof(temp));
				double a = double(temp / 255.0);
				images(i, rows * row + col) = double(a);
			}
		}
	}
	cout << endl << "Done." << endl;
	return images;
}

MatrixXd readLabelFile(string filename)
{
	ifstream ifs(filename.c_str(),ios::in | ios::binary);
	int magic_number = 0;
	int number_of_images = 0;
	ifs.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	ifs.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);

	MatrixXd _label = MatrixXd::Zero(number_of_images, 10);

	cout << number_of_images << endl;

	cout << "fetching label data..." << endl;
	string progress = "";

	for (int i = 0; i < number_of_images; i++)
	{
		double status = double(i * 100.0 / (number_of_images - 1));
		if (progress.size() < int(status) / 5)
		{
			progress += "#";
		}
		cout << "progress: " << setw(4) << right << fixed << setprecision(1) << (status) << "% " << progress << "\r" << flush;
		unsigned char temp = 0;
		ifs.read(reinterpret_cast<char*>(&temp), sizeof(temp));
		_label(i, int(temp)) = 1.0;
	}
	cout << endl << "Done." << endl;
	return _label;
}

MNISTDataSet::MNISTDataSet()
{
	dataSet = readImageFile(TRAIN_IMAGE_PATH);
	teachSet = readLabelFile(TRAIN_LABEL_PATH);
	testDataSet = readImageFile(TEST_IMAGE_PATH);
	testTeachSet = readLabelFile(TEST_LABEL_PATH);
}
