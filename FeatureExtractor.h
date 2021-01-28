#include<iostream>
#include<vector>
#include <math.h> 
#include<map>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
 
 
#include <opencv2/opencv.hpp>

using namespace std;


class FeatureExtractor
{
private :
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> descriptor;

public:
	FeatureExtractor(string fname);
	~FeatureExtractor()
	{


	}
	void ExtractFeature(cv::Mat Image, cv::Mat& descriptor);


};
