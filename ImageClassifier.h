#include"FeatureExtractor.h"
#include"KMeans.h"
class ImageClassifier
{
private:
	FeatureExtractor *fExtractor;
	int FeatureVectorDimenstion;
	std::map<int, std::vector<int*>>  BagOfWordsData;
	void FeatureExtractionOfImageUsingSift(std::map<int, std::vector<cv::Mat>>classToImageMap, vector<float*> &descriptors, std::map<int, std::map<int, vector<float*>> >&classToSiftVectorMap);
	void BagOfWordFromSiftDescriptor(std::map<int, std::map<int, vector<float*>>>&classToSiftVectorMap, std::map<int, std::vector<int*>>& BagOfWordsData);
	int NearestNeighbour(int*TestImageBow);
public:
	KMeans *kmeans;
	ImageClassifier()
	{


	}
	ImageClassifier(int nCenters);
	~ImageClassifier();
	void ReadImagesfromDirectory(std::map<int, std::vector<cv::String>>&classToImagePathImage, std::map<int, std::vector<cv::Mat>>&classToImageMap);
	void TrainClassifier(std::map<int, std::vector<cv::Mat>>&classToImageMap);
	std::map<int, std::vector<int*>> getModelData()
	{
		return BagOfWordsData;
	}
	int Predict(cv::Mat Image);

};