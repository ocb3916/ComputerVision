#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void main() 
{
	Mat original = imread("../data/windows.jpg");
	Mat ryan = imread("../data/ryan.png", IMREAD_UNCHANGED);
	Mat area = original(Rect(100, 120, ryan.cols, ryan.rows));

	vector<Mat> channels;
	split(ryan, channels);
	Mat	mask = channels[3];
	channels.pop_back();
	merge(channels, ryan);

	ryan.copyTo(area, mask);

	imshow("ryan", ryan);
	imshow("mask", mask);
	imshow("result", original);
	waitKey();
}