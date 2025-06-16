	#include "opencv2/opencv.hpp"
	#include "fstream"

	using namespace cv;
	using namespace std;

	const string dataPath = "../data/Practice5/";
	const string queryMetaFile = "meta_query.txt";

	void main() {
		Mat visualWords, imageVectors;
		// FileStorage�� �̿��Ͽ� ../data/Practice5/VWs.xml���� visual words�� �о��
		FileStorage fs(dataPath + "VWs.xml", FileStorage::READ);
		fs["VWs"] >> visualWords;
		fs.release();

		fs.open(dataPath + "BoVWs.xml", FileStorage::READ);
		fs["BoVWs"] >> imageVectors;
		fs.release();

		int nVWs = visualWords.rows;

		Ptr<Feature2D> sift = SIFT::create();
		Ptr<BFMatcher> vwMatcher = BFMatcher::create(NORM_L2SQR);
		vwMatcher->add(vector<Mat>(1, visualWords)); // Matcher�� visual words�� �߰�
		
		// Matcher �н�
		vwMatcher->train();

		Ptr<BFMatcher> imgMatcher = BFMatcher::create(NORM_L2SQR);
		imgMatcher->add(vector<Mat>(1, imageVectors)); // BoVWs�� Matcher�� �߰�
		imgMatcher->train();

		fstream fin(dataPath + queryMetaFile);
		string line;
		int objIdx = 0, totalCorrect = 0; // �̹��� �ε����� �� ���� �� �ʱ�ȭ 
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
