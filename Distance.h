#include<iostream>

template <typename T>
float euclideanDistance(T *d1, T*d2, int lenOfDesc)
{
	float Sum = 0;
	for (int i = 0; i < lenOfDesc; i++)
		Sum += (abs(d1[i] - d2[i])*abs(d1[i] - d2[i]));

	return sqrtf(Sum);

}
