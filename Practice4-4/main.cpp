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
	Mat mask = Mat::zeros(video.get(CAP_PROP_FRAME_HEIGHT), video.get(CAP_PROP_FRAME_WIDTH), CV_8UC1);
	
	//create SIFT here
	Ptr<Feature2D> sift = SIFT::create();
	
	Mat img;
	video >> img; // read the first frame
	video.release();
	for (int p = 0; p < 4; ++p) fin >> points[p].x >> points[p].y;
	std::vector<Point> poly;
	for (int p = 0; p < 4; ++p) {
		poly.push_back(Point(cvRound(points[p].x), cvRound(points[p].y)));
	}
	std::vector<std::vector<Point>> set_of_poly = { poly };
	fillPoly(mask, set_of_poly, Scalar(255));
	std::vector<KeyPoint> kps;
	Mat featureVector;
	sift->detectAndCompute(img, mask, kps, featureVector, false);
	drawKeypoints(img, kps, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cout << "Keypoints: " << kps.size() << endl;
	cout << "Feature Vector: " << featureVector.size() << endl;

	imshow("img", img);
	imshow("mask", mask);
	waitKey();
	fin.close();
}
