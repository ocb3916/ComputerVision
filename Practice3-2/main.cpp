#include "opencv2/opencv.hpp"
#include "fstream"

using namespace cv;
using namespace std;

const string dataPath = "../data/Practice3/";
const string cascadeName = "haarcascade_frontalface_alt.xml";
const string galleryName = "Meta_Gallery.txt";

CascadeClassifier faceDetector(dataPath + cascadeName);

Mat calcHist(const Mat& input) {
	Mat output(1, 256, CV_32FC1, Scalar(0));
	//2중 for loop
	for (int i = 0; i < input.rows; i++) { // 이미지 행
		for (int j = 0; j < input.cols; j++) { // 이미지 열
			output.at<float>(0, input.at<uchar>(i, j))++; // 해당 픽셀의 intensity에 해당하는 histogram의 값을 증가시킴
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

// 1-NN nclassification
Mat Classification(Mat& mGallery, Mat& mGalleryLabels, Mat& mProbe) {
	// mProbe 15 x 256, mGallery 45 x 256, return 15 x 1
	Mat mPrediction;
	for (int i = 0; i < mProbe.rows; i++) {
		Mat probeVector = mProbe.row(i); // i 행 벡터 추출 -> i x 256
		int idx;
		double minDist = DBL_MAX;
		for (int j = 0; j < mGallery.rows; j++) {
			Mat galleryVector = mGallery.row(j);
			double dist = norm(probeVector, galleryVector, NORM_L2SQR);
			if (dist < minDist) {
				idx = j;
				minDist = dist;
			}
		}
		mPrediction.push_back(Mat(1, 1, CV_32S, mGalleryLabels.at<int>(idx)));
	}

	return mPrediction; // 15 x 1
}

float GetAccuracy(Mat& prediction, Mat& gt) { // ground truth (정답)
	double hit = 0;
	for (int i = 0; i < prediction.rows; i++)
		if (prediction.at<int>(i) == gt.at<int>(i)) hit++;

	return hit / prediction.rows * 100;
}

void main() {
	ifstream infile(dataPath + galleryName);
	string line;
	Mat mGallery, mProbe;
	Mat mGalleryLabels, mProbeLabels;
	while (getline(infile, line)) {
		string filePath(dataPath + line);
		Mat rawlmg = imread(filePath, IMREAD_GRAYSCALE); // 배경 포함된 얼굴 이미지
		vector<Rect> detectedRegions = FaceDetection(rawlmg); // 얼굴 검출
		if (!detectedRegions.size()) continue; // 얼굴 없는 경우 다음 시행으로
		Mat facelmg = rawlmg(detectedRegions[0]); // 얼굴 영역만 추출
		Mat featureVector = calcHist(facelmg); // 얼굴 영역에 대한 feature vector 생성
		mGallery.push_back(featureVector); // 45 장의 feature vector, 45행 256열 Mat
		// 이미지의 id 추출 "Gallery/1_1.img"에서 1만 추출.
		mGalleryLabels.push_back(Mat(1, 1, CV_32S, Scalar((int)line[line.find("/") + 1] - '0')));
	}
	infile.close();
	infile.open(dataPath + "Meta_Probe.txt");
	while (getline(infile, line)) {
		std::string filePath(dataPath + line);
		Mat rawlmg = imread(filePath, IMREAD_GRAYSCALE);
		vector<Rect> detectedRegions = FaceDetection(rawlmg);
		if (!detectedRegions.size()) continue;
		Mat facelmg = rawlmg(detectedRegions[0]);
		Mat featureVector = calcHist(facelmg);
		mProbe.push_back(featureVector);
		mProbeLabels.push_back(Mat(1, 1, CV_32S, Scalar((int)line[line.find("/") + 1] - '0')));
	}
	infile.close();
	std::cout << mGallery.size() << endl; // 45 x 256, feature vector 모음. feature space
	std::cout << mGalleryLabels.size() << endl; // 45 x 1, 45개 이미지의 id모음.
	std:: cout << mProbe.size() << endl; // 15 x 256, test 이미지에 대한 feacture vector 모음.
	std:: cout << mProbeLabels.size() << endl; // 15 x 1, test 이미지 15개에 대한 id 모음.
	Mat mPredictions = Classification(mGallery, mGalleryLabels, mProbe); // 45 x 1
	std::cout << mPredictions << endl; // 15 x 1, 모델이 예측한 정답
	std::cout << "Accuracy == " << GetAccuracy(mPredictions, mProbeLabels) << '%' << endl; // 추론, 정답 -> 정확도 계산
}