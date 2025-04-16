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
	//2�� for loop
	for (int i = 0; i < input.rows; i++) { // �̹��� ��
		for (int j = 0; j < input.cols; j++) { // �̹��� ��
			output.at<float>(0, input.at<uchar>(i, j))++;
		}
	}

	return output;
}

Mat calchistWithGrid(Mat& input, int grid_x, int grid_y)
{
	int cell_size_x = input.cols / grid_x;
	int cell_size_y = input.rows / grid_y;
	Mat result;
	for (int i = 0; i < grid_y; i++) {
		for (int j = 0; j < grid_x; j++) {
			Mat cell = input(Rect(j * cell_size_x, i * cell_size_y, cell_size_x, cell_size_y)); // �� ���� ROI ����
			Mat hist = calcHist(cell); // �� ���� ������׷� ���
			result.push_back(hist);
		}
	}
	return result.reshape(0, 1); // 1 x (grid_x * grid_y) ��ķ� ��ȯ
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

// 1-NN classification
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
	return mPrediction;
}

float GetAccuracy(const Mat& mPredictions, const Mat& mProbeLabels) {
	int correct = 0;
	for (int i = 0; i < mPredictions.rows; i++) {
		if (mPredictions.at<int>(i) == mProbeLabels.at<int>(i)) {
			correct++;
		}
	}
	return (float)correct / mPredictions.rows * 100.0f;
}

void main()
{
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
		Mat LBPCodes = LBPEncoding(facelmg, 8, 3); // LBP feature vector ����
		Mat featureVector = calchistWithGrid(LBPCodes, 8, 10); // LBP feature vector�� grid�� ������ histogram ���
		mGallery.push_back(featureVector); // feature vector ����
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
		Mat LBPCodes = LBPEncoding(facelmg, 8, 3);
		Mat featureVector = calchistWithGrid(LBPCodes, 8, 10);
		mProbe.push_back(featureVector);
		mProbeLabels.push_back(Mat(1, 1, CV_32S, Scalar((int)line[line.find("/") + 1] - '0')));
	}
	infile.close();
	std::cout << mGallery.size() << endl; // 45 x 2^8*8*10, feature vector ����. feature space
	std::cout << mGalleryLabels.size() << endl; // 45 x 1, 45�� �̹����� id����.
	std::cout << mProbe.size() << endl; // 15 x 2^8*8*10, test �̹����� ���� feacture vector ����.
	std::cout << mProbeLabels.size() << endl; // 15 x 1, test �̹��� 15���� ���� id ����.
	Mat mPredictions = Classification(mGallery, mGalleryLabels, mProbe);
	std::cout << mPredictions << endl; // 15 x 1, ���� ������ ����
	std::cout << "Accuracy == " << GetAccuracy(mPredictions, mProbeLabels) << '%' << endl; // �߷�, ���� -> ��Ȯ�� ���
}