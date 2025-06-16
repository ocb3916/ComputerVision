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
	
	//create SIFT here
	Ptr<Feature2D> sift = SIFT::create();
	
	Mat img;
	while (video.read(img))
	{
		//detect SIFT keypoints here
		std::vector<KeyPoint> kps;
		sift->detect(img, kps);
		for (int p = 0; p < 4; ++p) fin >> points[p].x >> points[p].y;
		//draw lines here
		for (int p = 0; p < 4; ++p) {
			if (p == 3) line(img, points[p], points[0], Scalar(0, 255, 0), 3);
			else
				line(img, points[p], points[p + 1], Scalar(0, 255, 0), 3);
		}
		//draw SIFT keypoints here
		drawKeypoints(img, kps, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("img", img);
		waitKey();
	}
	
	video.release();
	fin.close();
}
