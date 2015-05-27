/*
 =====================================================================================
 *
 *       Author:  Jamshed Kaikaus, kaikaus2@illinois.edu
 *
 *       Description: Feature Tracking code
 *
 *       Version:  1.0
 *       Created:  05/27/2015
 *
 =====================================================================================
 */

#include <math.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <string>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

Mat maskingPoints(Mat img, std::vector<Point2f> vector, int size)
{
	//make mask to 'block off' existing feature points
	Mat gray;	
	Size s = img.size(); //obtain image dimensions
	int rows = s.height;
	int cols = s.width;
	Mat mask = Mat::ones(rows, cols, CV_8UC1)*255; //make mask based off of image dimensions
	int dim = size *2;
	for (int i =0; i < vector.size(); i++) //plot feature points onto mask
	{
		Rect pixels(vector[i].x-size, vector[i].y-size, dim, dim); 
		rectangle(mask, pixels, Scalar::all(0),-1,8, 0);
	}
	return mask;
}

void help(std::string program_name)
{
	std::cout << "Usage: " << program_name << " [options] [error_value] " << std::endl;
}

int main(int argc, char** argv)
{
	int error_val; //best value is between 9 and 16, ~12.
	if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'f') //'f' is flag to use findHomography
	{
		error_val = argv[1][0];
	} else if (argc > 1 && (argv[1][1] != 'f')) {
		help(argv[0]);
		exit(EXIT_FAILURE);
	} else {
		error_val = 0;
	}

	Mat img, gray_prev, gray;
	std::vector<int> tags;
	std::vector<Point2f> features_prev, original; //features_prev is previous, features is current, original is very first features obtained
	char name[500];
	char str[500];
	size_t next_tag = 1;
	const int max_count = 100; //maximum number of features to track
	const double qlevel = .01; //quality of features increases as qlevel decreases
	const double minDist = 15; //minimum distance between points

	while (scanf("%s", name) != EOF){
		
		//goodFeaturesToTrack variables
		std::vector<Point2f> features;
		
		//calcOpticalFLowPyrLK variables
		VideoCapture cap;
    	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    	Size subPixWinSize(10,10), winSize(31,31);
		
		Mat mask; //mask returned by maskingPoints
		std::vector<Point2f> temp; //temp vector use to obtain more features
		size_t k =0; //variable needed to resize vectors
		
		img = imread(name);
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
				}
				++it;
				++its;
				++itss;
				++j;
			}

			
			//homography
			if (error_val != 0)
			{	
				std::vector<Point2f> features_est, featprev_Vec, features_Vec;
				for(int a =0; a < max_count; a++)
		   		{
		        	featprev_Vec.push_back(Point2f( features_prev[a].x, features_prev[a].y ));
		        	features_Vec.push_back(Point2f( features[a].x, features[a].y ));
		    	}
				Mat homography = findHomography(featprev_Vec, features_Vec, CV_RANSAC );
				if (!homography.empty())
				{
				
					perspectiveTransform(featprev_Vec,features_est, homography);
					//removing outlier points
					size_t count = 0;
					double error;
					std::vector<int>::iterator t = tags.begin();
					std::vector<Point2f>::iterator ts = features.begin();
					std::vector<Point2f>::iterator tss = features_prev.begin();
					while(ts != features.end())
					{
						error = pow(features_Vec[count].x - features_est[count].x, 2) + pow(features_Vec[count].y - features_est[count].y, 2);
						if(error>error_val)
						{
							t=tags.erase(t);
							ts=features.erase(ts);
							tss= features_prev.erase(tss);
							count++;
							continue;
						}
						++t;
						++ts;
						++tss;
						++count;
					}	
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
			mask = maskingPoints(gray, features, 15);
			goodFeaturesToTrack(gray,temp, max_count_2, qlevel, minDist,mask, 3, 0, 0.04);
			for (size_t j =0; j < temp.size(); j++)
			{
				features.push_back(temp[j]);
				tags.push_back(next_tag++);
			}			
		}
		namedWindow("Image Window", WINDOW_NORMAL );
		imshow("Image Window", img);
		waitKey(1); //image displayed till key is pressed
		features_prev.clear();
		std::swap(features, features_prev); //move current features to previous
        std::swap(gray_prev, gray); //move the current image to previous
	}

	return 0;
}
