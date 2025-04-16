#include "opencv2/opencv.hpp"

using namespace cv;

void main() 
{
	Mat original = imread("../data/windows.jpg");
	Mat original_gray, mini, mini_gray;
	cvtColor(original, original_gray, COLOR_BGR2GRAY);
	resize(original, mini, original.size()/2);
	cvtColor(mini, mini_gray, COLOR_BGR2GRAY);

	imshow("original", original);
	imshow("original_gray", original_gray);
	imshow("mini", mini);
	imshow("mini_gray", mini_gray);
	waitKey();
}