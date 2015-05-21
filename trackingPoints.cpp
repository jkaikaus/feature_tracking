#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;

Mat maskingPoints(Mat img, std::vector<Point2f> vector)
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



int main(int argc, char** argv)
{
	
	Mat img, gray_prev, gray;
	std::vector<int> tags;
	std::vector<Point2f> features_prev, features, original; //features_prev is previous, features is current, original is very first features obtained
	char name[500];
	char str[500];
	size_t next_tag = 1;
	while (scanf("%s", name) != EOF){
		
		//goodFeaturesToTrack variables
		const int max_count = 100; //maximum number of features to track
		double qlevel = .01; //quality of features increases as qlevel decreases
		double minDist = 10; //minimum distance between points

		//calcOpticalFLowPyrLK variables
		VideoCapture cap;
    		TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    		Size subPixWinSize(10,10), winSize(31,31);
		

		Mat mask; //mask returned by maskingPoints
		std::vector<Point2f> temp; //temp vector use to obtain more features
		size_t k =0; //variable needed to resize vectors
		
		
		img = imread(name);
		//printf("%s\n", name);
		cvtColor(img, gray, COLOR_BGR2GRAY); //convert image to grayscale for use in gFTT and cOFPLK

		// Obtain very first set of features
		if (original.empty())
		{
			goodFeaturesToTrack(gray,original, max_count, qlevel, minDist,Mat(), 3, 0, 0.04);
			features_prev = original;
			for(int z = 0; z <features_prev.size(); z++)
			{
				tags.push_back(next_tag++);
			}
			
		}
		
		if (!features_prev.empty())
		{
			//features.clear();
			std::vector<uchar> status;
        		std::vector<float> err;
			if(gray_prev.empty())
			{
				gray.copyTo(gray_prev);
			}
			
			calcOpticalFlowPyrLK(gray_prev, gray, features_prev, features, status, err, winSize, 3, termcrit, 0, 0.001);
			size_t j = 0;
			std::vector<int>::iterator it = tags.begin();
			std::vector<Point2f>::iterator its = features.begin();
			std::vector<Point2f>::iterator itss = features_prev.begin();
			while(it != tags.end())
			{
				if(!status[j])
				{					
					it=tags.erase(it);
					its=features.erase(its);
					itss=features_prev.erase(itss);
					j++;
                   			continue;	
				} else {
					++it;
					++its;
					++itss;
					++j;
				}
				
			}
			
			//homography
			std::vector<Point2f> features_est;
			
			Mat homography = findHomography(features_prev, features, CV_RANSAC );
			perspectiveTransform(features_prev,features_est, homography);
			
			//removing outlier points
			int count = 0;
			while(its != features.end())
			{
				double error = pow(pow(features[count].x - features_est[count].x, 2) + pow(features[count].y - features_est[count].y, 2), 0.5);
				if(error>5)
				{					
					it=tags.erase(it);
					its=features.erase(its);
                   			continue;	
				} else {
					++it;
					++its;
					++count;
				}
				
			}


			for(size_t i = 0; i<features.size(); i++)
			{	
			
				features[k++] = features[i];  //keep track of total number of features
				circle(img, features[i], 5, Scalar(0, 0, 255), -1);
				sprintf(str, "%d", tags[i]);
				putText(img, str, features[i], FONT_HERSHEY_SCRIPT_SIMPLEX, .5,  Scalar::all(0)); //labels on each point
				printf("%d,%f,%f\n", tags[i], features[i].x, features[i].y);
				
			}
		


			features.resize(k); //resize to total number of features (no blank spaces)
			tags.resize(k);

		
		}

		if (features.size()< (size_t)max_count)
		{
						
			const int max_count_2 = 100;
			mask = maskingPoints(gray, features);
			goodFeaturesToTrack(gray,temp, max_count_2, qlevel, minDist,mask, 3, 0, 0.04);
			for (size_t j =0; j < temp.size(); j++)
			{
				features.push_back(temp[j]);
				tags.push_back(next_tag++);
			}			
		}

		imshow("img", img);
		waitKey(1); //image displayed till key is pressed
		features_prev.clear();
		std::swap(features, features_prev); //move current features to previous
        	swap(gray_prev, gray); //move the current image to previous
	}

	return 0;
}
