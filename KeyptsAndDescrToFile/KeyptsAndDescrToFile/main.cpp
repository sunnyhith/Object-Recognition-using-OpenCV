#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <fstream> 

using namespace cv;
using namespace std;

/** @function main */
int main(int argc, char** argv)
{


	//////////////---***DON'T FORGET TO CHANGE "name_of_object" TO THE NAME OF THE ACTUAL OBJECT***---/////////////


	Mat img = imread("C:\\Users\\sunny\\Desktop\\Project\\ObjectImages\\book.jpg", 1);

	if (!img.data)
	{
		printf("Error loading file! Please try again.\n");
		system("pause");
		return -1;
	}

	//Detect the keypoints using SURF Detector
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints;
	detector.detect(img, keypoints);

	//Calculate descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors;
	extractor.compute(img, keypoints, descriptors);

	//Store the keypoints and descriptors into the text file
	cv::FileStorage store("C:\\Users\\sunny\\Desktop\\Project\\KeyptsAndDescriptors\\book.txt", cv::FileStorage::WRITE);
	cv::write(store, "keypoints", keypoints);
	cv::write(store, "descriptors", descriptors);
	store.release();

	cout << "File Created!";
	
	while (1)
	{
		if (waitKey(300) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
			break;
	}

	return 0;
}
