#include <math.h>
#include "KMeans.h"
#include"Distance.h"
#include<string.h>
 

KMeans::KMeans(int dim, int nCluster)
{
	this->dataDimension = dim;
	this->K = nCluster;

	centroid.resize(this->K);

	for (int i = 0; i < nCluster; i++) 
	{
		centroid[i] = new float[dim];
	}
}
KMeans::~KMeans() 
{
	for (int i = 0; i < K; i++) 
	{
		delete[] centroid[i];
	}
	 centroid.clear();
}

void KMeans::Initialize(vector<float*> data)
{
	int number_sample = data.size() / K;
	for (int i = 0; i < K; i++) 
	{
		for (int j = 0; j < dataDimension; j++) 
		{
			double sum = 0;

			for (int k = i * number_sample; k < (i + 1) * number_sample; k++) 
			{
				sum += data[k][j];
			}
			centroid[i][j] = sum / number_sample;
		}
	}
}

int KMeans::predict(float* data) 
{
	int centerIndex = -1;;

	float distance = std::numeric_limits<std::int32_t>::max();

	for (int j = 0; j < K; j++)
	{

		float dist = euclideanDistance<float>(data, centroid[j], dataDimension);

		if (dist < distance)
		{
			centerIndex = j;
			distance = dist;
		}
	}
	return centerIndex;
}

void KMeans::train(vector<float*> data,int nIteration)
{
	

	// Sum up and count points for each cluster.
	vector<float*> new_means(K);
	for (int i = 0; i < K; i++)
	{

		new_means[i] = new float[dataDimension];
		memset(new_means[i], 0, sizeof(float) * dataDimension);
	}

	std::vector<size_t> assignments(data.size());
	for (size_t iteration = 0; iteration < nIteration; ++iteration)
	{
		// Find assignments.
		for (size_t point = 0; point < data.size(); ++point)
		{
			double best_distance = std::numeric_limits<double>::max();
			size_t best_cluster = 0;
			for (size_t cluster = 0; cluster < K; ++cluster)
			{
				const double distance = euclideanDistance<float>(data[point], centroid[cluster], dataDimension);
				if (distance < best_distance)
				{
					best_distance = distance;
					best_cluster = cluster;
				}
			}
			assignments[point] = best_cluster;
		}

		for (int i = 0; i < K; i++)
		{
			memset(new_means[i], 0, sizeof(float) * dataDimension);
		}

		std::vector<size_t> counts(K, 0);
		for (size_t point = 0; point < data.size(); ++point)
		{
			const auto cluster = assignments[point];
			for (int i = 0; i < 32; i++)
			{
				new_means[cluster][i] += data[point][i];

			}
			counts[cluster] += 1;
		}

		// Divide sums by counts to get new centroids.
		for (size_t cluster = 0; cluster < K; ++cluster)
		{
			// Turn 0/0 into 0/1 to avoid zero division.
			const auto count = std::max<size_t>(1, counts[cluster]);
			for (int i = 0; i < 32; i++)
			{
				centroid[cluster][i] = new_means[cluster][i] / count;

			}
		}
	}


	for (int i = 0; i < K; i++)
	{
		delete[] new_means[i];
	}
	new_means.clear();

}
