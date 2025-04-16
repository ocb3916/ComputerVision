#include "opencv2/opencv.hpp"

using namespace cv;

void main() 
{
	Mat original = imread("../data/windows.jpg");
	Mat ryan = imread("../data/ryan.bmp");
	Mat area = original(Rect(100, 120, ryan.cols, ryan.rows));
	ryan.copyTo(area);

	imshow("result", original);
	waitKey();
}