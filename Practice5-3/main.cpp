	#include "opencv2/opencv.hpp"
	#include "fstream"

	using namespace cv;
	using namespace std;

	const string dataPath = "../data/Practice5/";
	const string queryMetaFile = "meta_query.txt";

	void main() {
		Mat visualWords, imageVectors;
		// FileStorage를 이용하여 ../data/Practice5/VWs.xml에서 visual words를 읽어옴
		FileStorage fs(dataPath + "VWs.xml", FileStorage::READ);
		fs["VWs"] >> visualWords;
		fs.release();

		fs.open(dataPath + "BoVWs.xml", FileStorage::READ);
		fs["BoVWs"] >> imageVectors;
		fs.release();

		int nVWs = visualWords.rows;

		Ptr<Feature2D> sift = SIFT::create();
		Ptr<BFMatcher> vwMatcher = BFMatcher::create(NORM_L2SQR);
		vwMatcher->add(vector<Mat>(1, visualWords)); // Matcher에 visual words를 추가
		
		// Matcher 학습
		vwMatcher->train();

		Ptr<BFMatcher> imgMatcher = BFMatcher::create(NORM_L2SQR);
		imgMatcher->add(vector<Mat>(1, imageVectors)); // BoVWs를 Matcher에 추가
		imgMatcher->train();

		fstream fin(dataPath + queryMetaFile);
		string line;
		int objIdx = 0, totalCorrect = 0; // 이미지 인덱스와 총 정답 수 초기화 
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
			vector<vector<DMatch>> knnMatches; 
			imgMatcher->knnMatch(imageVector, knnMatches, 4);
			int nCorrect = 0;
			for (int i = 0; i < knnMatches[0].size(); i++)
				if (knnMatches[0][i].trainIdx / 4 == objIdx) nCorrect++;
			objIdx++;
			totalCorrect += nCorrect;
			cout << "image: " << objIdx << ": " << nCorrect << endl;
		}
		cout << "average: " << totalCorrect / (double)objIdx << endl;
	}
