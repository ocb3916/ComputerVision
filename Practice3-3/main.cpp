#include "opencv2/opencv.hpp"
#include "fstream"

using namespace cv;
using namespace std;

const string dataPath = "../data/Practice3/";
const string cascadeName = "haarcascade_frontalface_alt.xml";
const string galleryName = "Meta_Gallery.txt";

CascadeClassifier faceDetector(dataPath + cascadeName);

Mat LBPEncoding(Mat& input, int neighbor, int radius) {
	Mat mLBPCodes = Mat::zeros(input.rows - 2 * radius, input.cols - 2 * radius, CV_8U);
	for (int n = 0; n < neighbor; ++n) {
		float x = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbor)));
		float y = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbor)));
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		float ty = y - fy;
		float tx = x - fx;
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;
		for (int i = radius; i < input.rows - radius; ++i) {
			for (int j = radius; j < input.cols - radius; ++j) {
				uchar f = w1 * input.at<uchar>(i + fy, j + fx) + w2 * input.at<uchar>(i + fy, j + cx) +
					w3 * input.at<uchar>(i + cy, j + fx) + w4 * input.at<uchar>(i + cy, j + cx);
				mLBPCodes.at<uchar>(i - radius, j - radius) += (f > input.at<uchar>(i, j)) << n;
			}
		}
	}
	return mLBPCodes;
}

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

const std::vector<Rect> FaceDetection(cv::Mat& input) {
	std::vector<Rect> faces;
	faceDetector.detectMultiScale(input, faces, 1.1, 3, 0, Size(20, 20));
	return faces;
}

void main()
{
	ifstream infile(dataPath + galleryName);
	string line;
	while (getline(infile, line)) {
		Mat img = imread(dataPath + line, IMREAD_GRAYSCALE);
		std::vector<Rect> faces = FaceDetection(img); // std::vector<Rect> 변수 정의. 얼굴 검출.
		int numFaces = faces.size();
		cout << numFaces << endl; // 얼굴이 몇 개 검출되었는지 확인
		if (faces.empty()) 		// 검출되지 않으면 continue
			continue;
		Rect faceRoi = faces[0]; // 얼굴이 여러개 검출되면 첫번째 것이 정답이라고 가정
		rectangle(img, faceRoi, Scalar(255, 255, 255), 2, 8, 0); // 얼굴 영역에 사각형 표시
		// 얼굴 영역으로만 LBPencoding 후 getGrayHistImage
		Mat face = img(faceRoi);
		Mat LBPImage = LBPEncoding(face, 8, 3);
		Mat calchist = calcHist(LBPImage);
		Mat histimg = getGrayHistImage(calchist);

		imshow("face", face);
		imshow("LBPImage", LBPImage);
		imshow("histimg", histimg);
		waitKey();
	}
	infile.close();
}