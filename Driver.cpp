#include"ImageClassifier.h"
#include <fstream>
using namespace cv;
using namespace std;


map <int,string>ClassId_ClassNameMap;

void getDirectoryImages(std::map<int, std::vector<cv::String>>&classToImagePath , std::map<int, std::vector<string>>&classToImagetype, string path)
{
	vector<cv::String> listofImageCar, listofImageCat, listofImageFlower;
	cv::glob(path+"/car/*.jpg", listofImageCar, false);
	cv::glob(path + "/cat/*.jpg", listofImageCat, false);
	cv::glob(path + "/flower/*.jpg", listofImageFlower, false);

	classToImagePath [0] = listofImageCar;
	classToImagePath [1] = listofImageCat;
	classToImagePath [2] = listofImageFlower;
	vector<string> listOfCarImageName(listofImageCar.size(), "Car");
	vector<string> listOfCatImageName(listofImageCat.size(), "Cat");
	vector<string> listOfFlowerImageName(listofImageFlower.size(), "Flower");
	classToImagetype[0] = listOfCarImageName;
	classToImagetype[1] = listOfCatImageName;
	classToImagetype[2] = listOfFlowerImageName;
}

string SplitFilename(const std::string& str)
{
 
	unsigned found = str.find_last_of("/\\");
 
	return str.substr(found + 1) ;
}

int main(int argc, char *argv[])
{

	
	if (argc < 3)
	{
		cout << "please give correct path of training and testing images\n\n";
		return 0;
	}
	const int nCenters = 180;

	ClassId_ClassNameMap[0]="Car";
	ClassId_ClassNameMap[1]="Cat";
	ClassId_ClassNameMap[2] ="Flower";
	ofstream ResultFile;
	ResultFile.open("Result.txt");
	//,"C:\\Users\\himmat\\Downloads\\object_dataset\\Test"

	std::map<int, std::vector<cv::String>>classToImagePathImageTrain, classToImagePathImageTest;
	std::map<int, std::vector<cv::Mat>>classToImageMapTrain, classToImageMapTest;
	std::map<int, std::vector<string>>classToImageTypeTrain, classToImageTypeTest;
	ImageClassifier *Cobj = new ImageClassifier(nCenters);
	getDirectoryImages(classToImagePathImageTrain, classToImageTypeTrain,argv[1]);
	getDirectoryImages(classToImagePathImageTest, classToImageTypeTest,argv[2]);

	

	Cobj->ReadImagesfromDirectory(classToImagePathImageTrain, classToImageMapTrain);
	Cobj->ReadImagesfromDirectory(classToImagePathImageTest, classToImageMapTest);


	Cobj->TrainClassifier(classToImageMapTrain);

	map<int, vector<int>>ClassID_PredictedClass;

	for (int i = 0; i < classToImageMapTest.size(); i++)
	{
		vector<int> predictedClass;

		for (int j = 0; j < classToImageMapTest[i].size(); j++)
		{
			int ClassID = Cobj->Predict(classToImageMapTest[i][j]);
			predictedClass.push_back(ClassID);
		}
		ClassID_PredictedClass[i] = predictedClass;
	}
		
	ResultFile << " K-Means Centroids \n\n";
	vector<float*>Centroids = Cobj->kmeans->getCentroids();
	for (int i = 0; i < Centroids.size(); i++)
	{
		for (int j = 0; j < Cobj->kmeans->getDataDimension(); j++)
			ResultFile << Centroids[i][j] << " ";
		ResultFile << endl;
	}



	ResultFile << "\n\n\nResult of Classification of Images \n\n";


	ResultFile << "Image Name \t\t Class True  \t\t \t Class Predicted \n";
	for (int i = 0; i < classToImageMapTest.size(); i++)
	{

		for (int j = 0; j < classToImageMapTest[i].size(); j++)
		{
			ResultFile << SplitFilename(classToImagePathImageTrain[i][j]) << "\t\t" << classToImageTypeTest[i][j] << "\t\t" << ClassId_ClassNameMap[ClassID_PredictedClass[i][j]] << endl;

		}

	}

	ResultFile << "\n\n End \n\n";
	ResultFile.close();

	delete Cobj;
}
 
