	#include "opencv2/opencv.hpp"
	#include "fstream"

	using namespace cv;
	using namespace std;

	const string dataPath = "../data/Practice5/";
	const string databaseMetaFile = "meta_database.txt";

	void main() {
		Mat visualWords;
		// FileStorage를 이용하여 ../data/Practice5/VWs.xml에서 visual words를 읽어옴
		FileStorage fs(dataPath + "VWs.xml", FileStorage::READ);
		fs["VWs"] >> visualWords;
		fs.release();
		
		int nVWs = visualWords.rows;

		Ptr<Feature2D> sift = SIFT::create();
		Ptr<BFMatcher> vwMatcher = BFMatcher::create(NORM_L2SQR);
		vwMatcher->add(vector<Mat>(1, visualWords)); // Matcher에 visual words를 추가
		
		// Matcher 학습
		vwMatcher->train();

		Mat imageVectors;
		fstream fin(dataPath + databaseMetaFile);
		string line;
		while (getline(fin, line)) { 
			// 모든 이미지에서 feature vector를 추출해서 VWs와 매칭
			string filePath(dataPath + line);
			Mat img = imread(filePath);
			vector<KeyPoint> kps;
			Mat featureVector;
			sift->detectAndCompute(img, noArray(), kps, featureVector);
			vector<DMatch> matches;
			vwMatcher->match(featureVector, matches);

			Mat imageVector = Mat::zeros(1, nVWs, CV_32F); // 1 x nVWs
			// 히스토그램 계산
			for (int i = 0; i < matches.size(); i++) {
				imageVector.at<float>(matches[i].trainIdx)++; // 매칭된 visual word의 인덱스에 1을 더함
			}
			// feature 수로 나누기
			imageVector /= featureVector.rows; // featureVector.rows는 이미지에서 추출된 keypoint의 수 1 x k(1000)
			// imageVectors에 누적
			imageVectors.push_back(imageVector); // 1000 x k(1000)
		}
		fin.close();
		fs.open(dataPath + "BoVWs.xml", FileStorage::WRITE);
		fs << "BoVWs" << imageVectors;
		fs.release();
	}
