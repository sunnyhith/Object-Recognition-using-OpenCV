
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;

int main(int argc, char** argv)
{
	Mat object_image = imread("C:\\Users\\sunny\\Desktop\\Images\\book.jpg", 1);
	Mat query_image = imread("C:\\Users\\sunny\\Desktop\\Images\\book3.jpg", 1);

	std::vector<KeyPoint> keypoints_object;
	std::vector<KeyPoint> keypoints_query;

	Mat descriptors_object;
	Mat descriptors_query;

	if (!query_image.data || !object_image.data)
	{
		printf("Error loading file! Please try again.\n");
		//cin.get();
		system("pause");
		return -1;
	}

	//Detect the keypoints using SURF Detector
	int minHessian = 2500;
	SurfFeatureDetector detector(minHessian);

	detector.detect(object_image, keypoints_object);
	detector.detect(query_image, keypoints_query);

	//Calculate the descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	extractor.compute(object_image, keypoints_object, descriptors_object);
	extractor.compute(query_image, keypoints_query, descriptors_query);

	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptors_query, descriptors_object, matches);

	Mat img_matches;
	drawMatches(query_image, keypoints_query, object_image, keypoints_object, matches, img_matches);

	namedWindow("Matches", WINDOW_NORMAL);
	imshow("Matches", img_matches);
	
	waitKey(0);
	return 0;

}