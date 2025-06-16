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
		while (getline(fin, line)) { // ��� �̹������� feature vector�� ���� totalFeatureVectors�� ����
			string filePath(dataPath + line);
			Mat img = imread(filePath);
			vector<KeyPoint> kps;
			Mat featureVector;
			sift->detectAndCompute(img, noArray(), kps, featureVector);
			totalFeatureVectors.push_back(featureVector);
		}
		fin.close();
		Mat labels, visualWords; // k x 128
		// K-means clustering�� �̿��Ͽ� visual words ����
		kmeans(totalFeatureVectors, nVWs, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 0),
			1, KMEANS_PP_CENTERS, visualWords);

		// FileStorage �̿��Ͽ� ../data/Practice5/VWs.xml�� ����
		FileStorage fs(dataPath + "VWs.xml", FileStorage::WRITE);
		fs << "VWs" << visualWords;
		fs.release();
	}
