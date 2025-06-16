#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const string dataPath = "../data/Practice2/";
const string cascadeName = "haarcascade_frontalface_alt.xml";
const string imgName = "friends.jpg";
const string imgName2 = "oceans.jpg";

CascadeClassifier faceDetector(dataPath + cascadeName);

const std::vector<Rect> FaceDetection(cv::Mat &input) {
	std::vector<Rect> faces;
	faceDetector.detectMultiScale(input, faces, 1.1, 3, 0, Size(20, 20), Size(100, 100));
	return faces;
}

void main() 
{
	Mat img = imread(dataPath + imgName);
	Mat img2 = imread(dataPath + imgName2);
	std::vector<Rect> faces = FaceDetection(img);
	std::vector<Rect> faces2 = FaceDetection(img2);
	//faceDetector.detectMultiScale(img, faces, 1.1, 3, 0, Size(20, 20), Size(100, 100));
	for (int i = 0; i< faces.size(); i++) 
		rectangle(img, faces[i], Scalar(0,255,0), 2, 8, 0);
	for (int i = 0; i < faces2.size(); i++)
		rectangle(img2, faces2[i], Scalar(0, 255, 0), 2, 8, 0);
	
	imshow("img", img);
	imshow("img2", img2);
	waitKey();
}