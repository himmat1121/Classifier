 I have used following major techniques to solve given problem.
1. ORB (Oriented FAST and Rotated BRIEF) : I have used ORB to extract feature (descriptor) from images. I have used opencv ORB algorithm.
2. K-Means Clustering : This is used to cluster all descriptor of training images.
3. Bag Of Words: This is used to create histogram of image descriptors using K-Means centroids beacuse every image has different # of 	  descriptors.
4. Nearest Neighbour : this is used to find nearest neighbour of test images from trained images using BOWs.



How to Run the code.
1. updated training and testing data path in build.sh file .
2. then simply run "build.sh" file on linux platform by using command: "sh build.sh".



for training and testing, I have spilted data in  ratio 90:10 respectively.i have got accuracy around 80%(dependeds on kmeans centroid initialization).
