#include"ImageClassifier.h"
#include"Distance.h"
ImageClassifier::ImageClassifier(int nCenters) 
{
	fExtractor = new FeatureExtractor("Orb");
	kmeans = new KMeans(32,nCenters);
	FeatureVectorDimenstion = 32;
 
}

ImageClassifier::~ImageClassifier()
{

	delete fExtractor;
	delete kmeans;
	for (int i = 0; i < BagOfWordsData.size(); i++)
	{

		for (int j = 0; j < BagOfWordsData[i].size(); j++)
		{
			delete[]  BagOfWordsData[i][j];
		}
		BagOfWordsData[i].clear();
	}
}



void ImageClassifier::ReadImagesfromDirectory(std::map<int, std::vector<cv::String>>&classToImagePathImage, std::map<int, std::vector<cv::Mat>>&classToImageMap)
{


	for (int j = 0; j < classToImagePathImage.size(); j++)
	{
		std::vector<cv::Mat>ImagesMat;
		for (int i = 0; i < classToImagePathImage[j].size(); i++)
		{

			cv::Mat img = cv::imread(classToImagePathImage[j][i], 0);
			cv::resize(img, img, cv::Size(720,540));
			ImagesMat.push_back(img);

		}
		classToImageMap[j] = ImagesMat;
	}
}


void ImageClassifier::FeatureExtractionOfImageUsingSift(std::map<int, std::vector<cv::Mat>>classToImageMap, vector<float*> &descriptors, std::map<int, std::map<int, vector<float*>> >&classToSiftVectorMap)
{
 

	for (int i = 0; i < classToImageMap.size(); i++)
	{
		std::vector<cv::Mat> ImagesOfithClass = classToImageMap[i];
		cv::Mat descriptorsameClassImage;
		std::map<int, vector<float*>> DescriptorsOfithClass;
		for (int j = 0; j < ImagesOfithClass.size(); j++)
		{

			cv::Mat desc;
			vector<float*> descVector;
			fExtractor->ExtractFeature(ImagesOfithClass[j], desc);
			descriptorsameClassImage.push_back(desc);
			for (int xt = 0; xt < desc.rows; xt++)
			{
				float *DataPtr = new float[desc.cols];
				for (int xt1 = 0; xt1 < desc.cols; xt1++)
					DataPtr[xt1] = desc.row(xt).data[xt1];
				descVector.push_back(DataPtr);

			}

			DescriptorsOfithClass[j] = descVector;
		}
		classToSiftVectorMap[i] = DescriptorsOfithClass;
		
		for (int xt = 0; xt < descriptorsameClassImage.rows; xt++)
		{
			float *DataPtr = new float[descriptorsameClassImage.cols];
			for (int xt1 = 0; xt1 < descriptorsameClassImage.cols; xt1++)
				DataPtr[xt1] = descriptorsameClassImage.row(xt).data[xt1];
			descriptors.push_back(DataPtr);

		}
	}
}


void ImageClassifier::BagOfWordFromSiftDescriptor(std::map<int, std::map<int, vector<float*>>>&classToSiftVectorMap,  std::map<int, std::vector<int*>>& BagOfWordsData)
{


	for (int i = 0; i < classToSiftVectorMap.size(); i++)
	{

		std::map<int, vector<float*>> ithClassDescriptor = classToSiftVectorMap[i];
		std::vector<int*> BowDataPerclass;
		for (int j = 0; j < ithClassDescriptor.size(); j++)
		{
			int *hist = new int[kmeans->getKVal()];
			memset(hist, 0, sizeof(int)*kmeans->getKVal());
			 vector<float*> DescOfImage = classToSiftVectorMap[i][j];
			for (int k = 0; k < DescOfImage.size(); k++)
			{
				int index = kmeans->predict(DescOfImage[k]);
				hist[index]++;

			}
			BowDataPerclass.push_back(hist);

		}
		BagOfWordsData[i] = BowDataPerclass;
	}

}

int ImageClassifier::NearestNeighbour(int*TestImageBow )
{
	float Distance = INT_MAX;
	int classID = -1;
	for (int i = 0; i < BagOfWordsData.size(); i++)
	{
		for (int j = 0; j < BagOfWordsData[i].size(); j++)
		{
			float dist = euclideanDistance<int>(TestImageBow, BagOfWordsData[i][j], kmeans->getKVal());
			if (dist < Distance)
			{
				 Distance = dist;
				classID = i;
			}
		}

	}

	return classID;
}

void ImageClassifier::TrainClassifier(std::map<int, std::vector<cv::Mat>>&classToImageMap )
{
	vector<float*> descriptors;
	std::map<int, std::map<int, vector<float*>> >classToSiftVectorMap;
	FeatureExtractionOfImageUsingSift(classToImageMap, descriptors, classToSiftVectorMap);
	kmeans->Initialize(descriptors);
	kmeans->train(descriptors, 300);
	BagOfWordFromSiftDescriptor(classToSiftVectorMap, BagOfWordsData);

	for (int i = 0; i < descriptors.size(); i++)
	{
		delete[] descriptors[i];
	}

	for (int i = 0; i < classToSiftVectorMap.size(); i++)
	{

		for (int j = 0; j < classToSiftVectorMap[i].size(); j++)
		{
			for (int k = 0; k < classToSiftVectorMap[i][j].size(); k++)
				delete[]  classToSiftVectorMap[i][j][k];

			classToSiftVectorMap[i][j].clear();
		}
		
	}
	descriptors.clear();
}


int ImageClassifier::Predict(cv::Mat Image)
{
	vector<float*> descriptors;
	std::vector<cv::Mat> ImageList;
	std::map<int, std::vector<cv::Mat>> classToImageMapTest;
	std::map<int, std::vector<int*>> BagOfWordsDataTest;
	ImageList.push_back(Image);
	classToImageMapTest[0] = ImageList;
	std::map<int, std::map<int, vector<float*>> >classToSiftVectorMap;
	 FeatureExtractionOfImageUsingSift(classToImageMapTest, descriptors , classToSiftVectorMap);
	BagOfWordFromSiftDescriptor(classToSiftVectorMap, BagOfWordsDataTest);
	int cID =  NearestNeighbour(BagOfWordsDataTest[0][0]);

	for (int i = 0; i < classToSiftVectorMap.size(); i++)
	{

		for (int j = 0; j < classToSiftVectorMap[i].size(); j++)
		{
			for (int k = 0; k < classToSiftVectorMap[i][j].size(); k++)
				delete[]  classToSiftVectorMap[i][j][k];

			classToSiftVectorMap[i][j].clear();
		}

	}

	for (int i = 0; i < descriptors.size(); i++)
	{
		delete[] descriptors[i];
	}
	descriptors.clear();

	for (int i = 0; i < BagOfWordsDataTest.size(); i++)
	{

		for (int j = 0; j < BagOfWordsDataTest[i].size(); j++)
		{
			delete[]  BagOfWordsDataTest[i][j];
		}
		BagOfWordsDataTest[i].clear();
	}

	return cID;
}


