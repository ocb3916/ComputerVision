	#include "opencv2/opencv.hpp"
	#include "fstream"

	using namespace cv;
	using namespace std;

	const string dataPath = "../data/Practice5/";
	const string trainMetaFile = "meta_train.txt";
	const int nVWs = 1000; // number of visual words

	void main() {
		Ptr<Feature2D> sift = SIFT::create();
		Mat totalFeatureVectors;
		fstream fin(dataPath + trainMetaFile);
		string line;
		while (getline(fin, line)) { // 모든 이미지에서 feature vector를 추출 totalFeatureVectors에 저장
			string filePath(dataPath + line);
			Mat img = imread(filePath);
			vector<KeyPoint> kps;
			Mat featureVector;
			sift->detectAndCompute(img, noArray(), kps, featureVector);
			totalFeatureVectors.push_back(featureVector);
		}
		fin.close();
		Mat labels, visualWords; // k x 128
		// K-means clustering을 이용하여 visual words 생성
		kmeans(totalFeatureVectors, nVWs, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 0),
			1, KMEANS_PP_CENTERS, visualWords);

		// FileStorage 이용하여 ../data/Practice5/VWs.xml에 저장
		FileStorage fs(dataPath + "VWs.xml", FileStorage::WRITE);
		fs << "VWs" << visualWords;
		fs.release();
	}
