#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;

int main(int argc, char** argv)
{
	Mat object_image = imread("C:\\Users\\sunny\\Desktop\\Images\\book.jpg", 1);

	std::vector<KeyPoint> keypoints_object;

	Mat descriptors_object;

	if (!object_image.data)
	{
		printf("Error loading file! Please try again.\n");
		//cin.get();
		system("pause");
		return -1;
	}

	//Detect the keypoints using SURF Detector
	int minHessian = 500;
	SurfFeatureDetector detector(minHessian);

	detector.detect(object_image, keypoints_object);

	//Calculate the descriptors
	SurfDescriptorExtractor extractor;
	extractor.compute(object_image, keypoints_object, descriptors_object);

	Mat img_matches;
	drawKeypoints(object_image, keypoints_object, img_matches, Scalar::all(-1));

	namedWindow("Keypoints And Descriptors", WINDOW_NORMAL);
	imshow("Keypoints And Descriptors", img_matches);

	waitKey(0);
	return 0;

}