#include <stdio.h>
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	std::vector<std::string> object = { "Landscape", "Book", "Rack" };
	std::string loc_of_img = { "C:\\Users\\sunny\\Desktop\\Project\\ObjectImages\\" };
	std::string loc_of_txt = { "C:\\Users\\sunny\\Desktop\\Project\\KeyptsAndDescriptors\\" };
	std::string location_img = "", cur_object = "", location_txt = "";
	int no_of_objects = object.size();
	int i = 0, j = 0, k = 0, size_of_matches = 0;
	const float ratio = 0.8f;
	bool flag;
	
	std::vector<KeyPoint> keypoints_object;
	Mat descriptors_object;
	cv::FileNode n1, n2;
	
	//Loading the query image from the user
	Mat query_image = imread("C:\\Users\\sunny\\Desktop\\Images\\book3.jpg", 1);
	
	//Displays error message if there is a problem in loading the image
	if (!query_image.data)
	{
		printf("Error loading file! Please try again.\n");
		system("pause");
		return -1;
	}

	//Detecting the keypoints using SURF Detector
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_query;
	detector.detect(query_image, keypoints_query);

	//Calculating the descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors_query;
	extractor.compute(query_image, keypoints_query, descriptors_query);
	
	
	//Comparing the keypoints and descriptors of query image and each of the object images to find the best possible match

	for (i = 0 ; i < no_of_objects ; i++)
	{
		//Considering one object at a time
		cur_object = object[i];
		//Taking the location of the object image
		location_img = loc_of_img + cur_object + ".jpg";
		
		Mat object_image = imread(location_img, 1);
		if (!object_image.data)
		{
			printf("Unexpected error has occured! Please try again.\n");
			system("pause");
			return -1;
		}

		//Getting the txt files containing information about keypoints and descriptors of objects
		location_txt = loc_of_txt + cur_object + ".txt";
		cv::FileStorage retrieve(location_txt, cv::FileStorage::READ);

		//Fetching the keypoints values into n1
		n1 = retrieve["keypoints"];
		//Storing the value of n1 in keypoints_object
		cv::read(n1, keypoints_object);
		//Fetching the descriptors values into n2
		n2 = retrieve["descriptors"];
		//Storing the value of n2 in descriptors_object
		cv::read(n2, descriptors_object);

		//Using Brute Force Algorithm for drawing the match lines
		BFMatcher matcher(NORM_L2);
		std::vector<vector< DMatch >> matches;
		//knnMatch is used to find the k nearest neighbours 
		matcher.knnMatch(descriptors_query, descriptors_object, matches, 2);

		vector<cv::DMatch> good_matches;
		size_of_matches = matches.size();

		for (j = 0 ; j < size_of_matches ; ++j)
		{
			if (matches[j][0].distance < ratio * matches[j][1].distance)
			{
				//Updating the good_matches vector whenever the above condition is satisfied
				good_matches.push_back(matches[j][0]);
			}
		}
		
		if (good_matches.size() > (0.35 * size_of_matches))
		{
			flag = true;
			Mat img_matches2;

			//Drawing the matches and storing the result in img_matches2
			drawMatches(query_image, keypoints_query, object_image, keypoints_object, good_matches, img_matches2);

			//Used to display text on an image. Here the image is img_matches2 and the text is object[i]
			putText(img_matches2, object[i], Point(10, 250), FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2);
			cout << "By means of the best matches, the Object Found is : " << cur_object;

			//Creates a new window, named "Matches Found"
			namedWindow("Matches Found", WINDOW_NORMAL);
			//Used to display the img_matches2 image in the "Matches Found" window
			imshow("Matches Found", img_matches2);
			break;
		}

		else flag = false;

		//Used for de-allocating the keypoints and descriptors variables
		descriptors_object.release();
		keypoints_object.clear();

	}

	//Failure check if incase the object is not found
	if ((i == no_of_objects) && (flag == false))
	{
		cout << "Sorry! No Object found in the Dataset\n";
		system("pause");
	}

	waitKey(0);
	return 0;
	
}