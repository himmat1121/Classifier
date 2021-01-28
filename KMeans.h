#include<iostream>
#include<vector>
#include <math.h> 
#include<map>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <cstdint> 
using namespace std;
class KMeans 
{
private:
	int  dataDimension;
	int  K;
	vector<float*>centroid;
public:


	KMeans(int dim, int nCluster);
	~KMeans();

	// forgy method
	void Initialize(vector<float*> data);

	int predict(float* data);
	int getKVal()
	{
		return K;

	}

	int getDataDimension()
	{

		return dataDimension;
	}

	vector<float*>getCentroids()
	{

		return centroid;
	}

	void train(vector<float*> data, int nIteration=100);
};
