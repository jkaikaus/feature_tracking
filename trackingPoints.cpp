// Write some basic code to track points in a video using OpenCV.
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

//Load images, get correct image names, then display image
int main(int argc, char** argv)
{
	Mat img, gray_prev, gray;
	std::vector<Point2f> features[2],original; //features[0] is previous, features[1] is current, original is very first features chosen
	int max_count = 300;
	char name[500];
	double qlevel = .01;
	double minDist = 10;
	std::vector<uchar> status;
        std::vector<float> err;
	VideoCapture cap;
    	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    	Size subPixWinSize(10,10), winSize(31,31);
	
	while (scanf("%s", name) != EOF){
		img = imread(name);
		printf("%s\n", name);
		cvtColor(img, gray, COLOR_BGR2GRAY);

		// Obtain very first set of features
		if (original.empty())
		{
			goodFeaturesToTrack(gray,original, max_count, qlevel, minDist,Mat(), 3, 0, 0.04);
			features[0] = original;
			
		}

		
		
		/*if (features[0].size()< max_count) //add to statement in order to get more points
		{
			goodFeaturesToTrack(gray,features[0], max_count, qlevel, minDist,Mat(), 3, 0, 0.04);
		}*/

		
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
				circle(img, features[0][i], 5, Scalar(0, 255, 0), -1);	
				circle(img, features[1][i], 5, Scalar(0, 0, 255), -1);
				line(img, features[0][i], features[1][i],Scalar(0, 255, 0));
			
			}
			
		}

		imshow("img", img);
		waitKey(1); //image displayed till key is pressed
		
		features[0]=features[1];
		gray_prev = gray.clone();
	}

	return 0;
}
		


	
