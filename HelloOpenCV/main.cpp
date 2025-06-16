#include "opencv2/opencv.hpp"

using namespace cv;

void main() 
{
	Mat img = imread("../data/windows.jpg");
	Mat img2(480, 640, CV_8UC1, Scalar(128));
	Mat img3(Size(640, 480), CV_8UC3, Scalar(0,0,255));

	imshow("img", img);
	imshow("img2", img2);
	imshow("img3", img3);
	waitKey();
}