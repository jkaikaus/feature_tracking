// Some basic code to track points in a video using OpenCV.
// Look at this function for ideas: calcOpticalFlowPyrLK()
// Replace the C-interface functions with C++ functions in the current tracking code.
// Integrate the updated code into our reflection tracking code.

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;

Mat maskingPoints(Mat img, std::vector<Point2f> vector, int sidelength)
{
	//make mask to 'block off' existing feature points
	Mat gray;	
	Size s = img.size(); //obtain image dimensions
	int rows = s.height;
	int cols = s.width;
	Mat mask = Mat::ones(rows, cols, CV_8UC1)*255; //make mask based off of image dimensions
	for (int i =0; i < vector.size(); i++) //plot feature points onto mask
	{
		mask.at<int>(vector[i]) = 0;
	}
	return mask;
}

//Load images, get correct image names, then display image
int main(int argc, char** argv)
{
	
	Mat img, gray_prev, gray;
	std::vector<Point2f> features[2],original; //features[0] is previous, features[1] is current, original is very first features obtained
	char name[500];

	while (scanf("%s", name) != EOF){
		
		//goodFeaturesToTrack variables
		int max_count = 200; //maximum number of features to track
		double qlevel = .01; //quality of features increases as qlevel decreases
		double minDist = 10; //minimum distance between points

		//calcOpticalFLowPyrLK variables
		VideoCapture cap;
    		TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    		Size subPixWinSize(10,10), winSize(31,31);
		std::vector<uchar> status;
        	std::vector<float> err;

		Mat mask; //mask returned by maskingPoints
		std::vector<Point2f> temp; //temp vector use to obtain more features
		size_t k =0; //variable needed to resize vectors

		img = imread(name);
		printf("%s\n", name);
		cvtColor(img, gray, COLOR_BGR2GRAY); //convert image to grayscale to use in gFTT and cOFPLK

		// Obtain very first set of features
		if (original.empty())
		{
			goodFeaturesToTrack(gray,original, max_count, qlevel, minDist,Mat(), 3, 0, 0.04);
			features[0] = original;
			
		}

		//Add on to features vector as frames increases.
		if (features[0].size()< max_count)
		{
			mask = maskingPoints(gray, features[0], 1);
			goodFeaturesToTrack(gray,temp, max_count, qlevel, minDist,mask, 3, 0, 0.04);
			for (int j =0; j < temp.size(); j++)
			{
				features[0].push_back(temp[j]);
			}			
		}

		
		if (!features[0].empty())
		{
			features[1].clear();
			if(gray_prev.empty() )
			{
				gray.copyTo(gray_prev);
			}

			calcOpticalFlowPyrLK(gray_prev, gray, features[0], features[1], status, err, winSize, 3, termcrit, 0, 0.001);
			
			for(size_t i = 0; i<features[1].size(); i++)
			{
				
				if( !status[i] )
				{
                    			continue;
				}
				features[1][k++] = features[1][i];//keep track of totale number of features
				circle(img, features[1][i], 5, Scalar(0, 0, 255), -1);
				//circle(img, features[0][i], 5, Scalar(0, 255, 0), -1);	
				//line(img, features[0][i], features[1][i],Scalar(0, 255, 0));		
			
			}
		features[1].resize(k); //resize to total number of features (no blank spaces)
		}

		imshow("img", img);
		waitKey(1); //image displayed till key is pressed

		std::swap(features[1], features[0]); //move current features to previous
        	swap(gray_prev, gray); //move the current image to previous
	}

	return 0;
}
		


	
