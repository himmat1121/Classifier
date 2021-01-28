#include"FeatureExtractor.h"

FeatureExtractor::FeatureExtractor(string fname)
{
	if (fname == "Orb")
	{

		 detector = cv::ORB::create(700);
		 descriptor = cv::ORB::create(700);
		 
	}

}


void FeatureExtractor::ExtractFeature(cv::Mat Image, cv::Mat& desc)
{
	std::vector<cv::KeyPoint> kps;

	detector->detect(Image, kps);
	descriptor->compute(Image, kps, desc);

}
