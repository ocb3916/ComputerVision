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
	//2�� for loop
	for (int i = 0; i < input.rows; i++) { // �̹��� ��
		for (int j = 0; j < input.cols; j++) { // �̹��� ��
			output.at<float>(0, input.at<uchar>(i, j))++; // �ش� �ȼ��� intensity�� �ش��ϴ� histogram�� ���� ������Ŵ
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
		Mat probeVector = mProbe.row(i); // i �� ���� ���� -> i x 256
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

float GetAccuracy(Mat& prediction, Mat& gt) { // ground truth (����)
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
		Mat rawlmg = imread(filePath, IMREAD_GRAYSCALE); // ��� ���Ե� �� �̹���
		vector<Rect> detectedRegions = FaceDetection(rawlmg); // �� ����
		if (!detectedRegions.size()) continue; // �� ���� ��� ���� ��������
		Mat facelmg = rawlmg(detectedRegions[0]); // �� ������ ����
		Mat featureVector = calcHist(facelmg); // �� ������ ���� feature vector ����
		mGallery.push_back(featureVector); // 45 ���� feature vector, 45�� 256�� Mat
		// �̹����� id ���� "Gallery/1_1.img"���� 1�� ����.
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
	std::cout << mGallery.size() << endl; // 45 x 256, feature vector ����. feature space
	std::cout << mGalleryLabels.size() << endl; // 45 x 1, 45�� �̹����� id����.
	std:: cout << mProbe.size() << endl; // 15 x 256, test �̹����� ���� feacture vector ����.
	std:: cout << mProbeLabels.size() << endl; // 15 x 1, test �̹��� 15���� ���� id ����.
	Mat mPredictions = Classification(mGallery, mGalleryLabels, mProbe); // 45 x 1
	std::cout << mPredictions << endl; // 15 x 1, ���� ������ ����
	std::cout << "Accuracy == " << GetAccuracy(mPredictions, mProbeLabels) << '%' << endl; // �߷�, ���� -> ��Ȯ�� ���
}