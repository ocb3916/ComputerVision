#include "opencv2/opencv.hpp"
#include "fstream"

using namespace cv;
using namespace std;

const std::string videoHome = "../data/Practice4/video/";
const std::string videoExt = "mp4";
const std::string annHome = "../data/Practice4/annotation/";
const std::string annSuffix = "_gt_points";
const std::string annExt = "txt";

void main(int argc, char** argv)
{
	const std::string videoName = argv[1];
	std::fstream fin(annHome + videoName + annSuffix + "." + annExt);
	std::vector<Point2f> points(4);
	VideoCapture video(videoHome + videoName + "." + videoExt);
	Mat img;
	while (video.read(img))
	{
		for (int p = 0; p < 4; ++p) fin >> points[p].x >> points[p].y; 
		//draw line here
		for (int p = 0; p < 4; ++p) {
			if (p == 3) line(img, points[p], points[0], Scalar(0, 255, 0), 3);
			else
			line(img, points[p], points[p + 1], Scalar(0, 255, 0), 3);
		}
		imshow("img", img);
		waitKey();
	}
	video.release();
	fin.close();
}