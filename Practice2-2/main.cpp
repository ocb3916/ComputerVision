#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const string dataPath = "../data/Practice2/";
const string imgName = "lenna.bmp";

Mat calcHist(const Mat& input) {
	Mat output(1, 256, CV_32FC1, Scalar(0));
	//2중 for loop
	for (int i = 0; i < input.rows; i++) { // 이미지 행
		for (int j = 0; j < input.cols; j++) { // 이미지 열
			output.at<float>(0, input.at<uchar>(i, j))++;
		}
	}

	return output;
}

Mat getGrayHistImage(const Mat& hist) {
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(256, 1));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC1, Scalar(255));
	for (int i = 0; i < 256; i++) {
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(0, i) * 100.0 / histMax)), Scalar(0));
	}

	return imgHist;
}

void main() 
{
	Mat img = imread(dataPath + imgName, IMREAD_GRAYSCALE);
	Mat hist = calcHist(img);
	Mat hist_img = getGrayHistImage(hist);

	imshow("img", img);
	imshow("hist_img", hist_img);
	waitKey();
}