	#include "opencv2/opencv.hpp"
	#include "fstream"

	using namespace cv;
	using namespace std;

	const string dataPath = "../data/Practice5/";
	const string databaseMetaFile = "meta_database.txt";

	void main() {
		Mat visualWords;
		// FileStorage�� �̿��Ͽ� ../data/Practice5/VWs.xml���� visual words�� �о��
		FileStorage fs(dataPath + "VWs.xml", FileStorage::READ);
		fs["VWs"] >> visualWords;
		fs.release();
		
		int nVWs = visualWords.rows;

		Ptr<Feature2D> sift = SIFT::create();
		Ptr<BFMatcher> vwMatcher = BFMatcher::create(NORM_L2SQR);
		vwMatcher->add(vector<Mat>(1, visualWords)); // Matcher�� visual words�� �߰�
		
		// Matcher �н�
		vwMatcher->train();

		Mat imageVectors;
		fstream fin(dataPath + databaseMetaFile);
		string line;
		while (getline(fin, line)) { 
			// ��� �̹������� feature vector�� �����ؼ� VWs�� ��Ī
			string filePath(dataPath + line);
			Mat img = imread(filePath);
			vector<KeyPoint> kps;
			Mat featureVector;
			sift->detectAndCompute(img, noArray(), kps, featureVector);
			vector<DMatch> matches;
			vwMatcher->match(featureVector, matches);

			Mat imageVector = Mat::zeros(1, nVWs, CV_32F); // 1 x nVWs
			// ������׷� ���
			for (int i = 0; i < matches.size(); i++) {
				imageVector.at<float>(matches[i].trainIdx)++; // ��Ī�� visual word�� �ε����� 1�� ����
			}
			// feature ���� ������
			imageVector /= featureVector.rows; // featureVector.rows�� �̹������� ����� keypoint�� �� 1 x k(1000)
			// imageVectors�� ����
			imageVectors.push_back(imageVector); // 1000 x k(1000)
		}
		fin.close();
		fs.open(dataPath + "BoVWs.xml", FileStorage::WRITE);
		fs << "BoVWs" << imageVectors;
		fs.release();
	}
