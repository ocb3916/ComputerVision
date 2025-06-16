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
		const std::string videoName = "V30_7";
		std::fstream fin(annHome + videoName + annSuffix + "." + annExt);
		std::vector<Point2f> points(4);
		VideoCapture video(videoHome + videoName + "." + videoExt);
		Mat mask = Mat::zeros(video.get(CAP_PROP_FRAME_HEIGHT), video.get(CAP_PROP_FRAME_WIDTH), CV_8UC1);
	
		//create SIFT here
		Ptr<Feature2D> sift = SIFT::create();
		Ptr<BFMatcher> bfmatcher = BFMatcher::create(NORM_L2, true);
	
		Mat initImg;
		video >> initImg; // read the first frame
		for (int p = 0; p < 4; ++p) fin >> points[p].x >> points[p].y;
		std::vector<Point> poly;
		for (int p = 0; p < 4; ++p) {
			poly.push_back(Point(cvRound(points[p].x), cvRound(points[p].y)));
		}
		std::vector<std::vector<Point>> set_of_poly = { poly };
		fillPoly(mask, set_of_poly, Scalar(255));
		std::vector<KeyPoint> initKps;
		Mat initFeatureVector;
		sift->detectAndCompute(initImg, mask, initKps, initFeatureVector, false);

		Mat testImg;
		while (video.read(testImg))
		{
			//detect SIFT keypoints here
			std::vector<KeyPoint> testKps;
			Mat testFeatureVector;
			vector<DMatch> matches;
			Mat matchImg;
			sift->detectAndCompute(testImg, noArray(), testKps, testFeatureVector, false);
			bfmatcher->match(initFeatureVector, testFeatureVector, matches);
			std::vector<Point2f> initPTs, testPTs;
			for (int i = 0; i < matches.size(); i++) {
				initPTs.push_back(initKps[matches[i].queryIdx].pt);
				testPTs.push_back(testKps[matches[i].trainIdx].pt);
			}
			Mat inlier;
			Mat H = findHomography(initPTs, testPTs, RANSAC, 3.0, inlier);
			int nlinlier = countNonZero(inlier);
			std::vector<Point2f> gtPTs(4);
			for (int p = 0; p < 4; ++p) fin >> gtPTs[p].x >> gtPTs[p].y;
			line(testImg, gtPTs[0], gtPTs[1], Scalar(255, 0, 0), 3);
			line(testImg, gtPTs[1], gtPTs[2], Scalar(255, 0, 0), 3);
			line(testImg, gtPTs[2], gtPTs[3], Scalar(255, 0, 0), 3);
			line(testImg, gtPTs[3], gtPTs[0], Scalar(255, 0, 0), 3);
			if (nlinlier >= 100) {
				std::cout << "recognition success: " << nlinlier << std::endl;
				std::vector<Point2f> transPTs;
				perspectiveTransform(points, transPTs, H);
				line(testImg, transPTs[0], transPTs[1], Scalar(0, 255, 0), 3);
				line(testImg, transPTs[1], transPTs[2], Scalar(0, 255, 0), 3);
				line(testImg, transPTs[2], transPTs[3], Scalar(0, 255, 0), 3);
				line(testImg, transPTs[3], transPTs[0], Scalar(0, 255, 0), 3);
			}
			else {
				std::cout << "recognition failed: " << nlinlier << std::endl;
			}

			imshow("testImg", testImg);
			waitKey();
		}

		waitKey();
		fin.close();
		video.release();
	}
