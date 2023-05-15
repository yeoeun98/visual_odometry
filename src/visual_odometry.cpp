#include "visual_odometry.h"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	// 이미지 변수
	Mat img_1;
	Mat img_2;
	Mat gray_1;
	Mat gray_2;
	Mat feature_1;
	Mat feature_2;
	Mat match_img;

	// key points 벡터
	vector<KeyPoint> keyPoint_1;
	vector<KeyPoint> keyPoint_2;

	//descriptor 행렬
	Mat desc_1;
	Mat desc_2;

	// 매칭된 특징점
	vector<DMatch> matches;


	// 특징 검출기
	Ptr<cv::Feature2D> orb = cv::ORB::create(100);
	Ptr<cv::Feature2D> sift = cv::SIFT::create(500);
	Ptr<cv::Feature2D> brisk = cv::BRISK::create();

	// 특징 매칭
	Ptr<DescriptorMatcher> matcher = BFMatcher::create();

	img_1 = imread("../data/00/image_2/000000.png", cv::IMREAD_ANYCOLOR);
	img_2 = imread("../data/00/image_2/000000.png", cv::IMREAD_ANYCOLOR);
	
	// 컬러 이미지 -> 흑백 이미지 변환(특징검출엔 흑백 이미지만 사용)
	cvtColor(img_1, gray_1, COLOR_BGR2GRAY);
	cvtColor(img_2, gray_2, COLOR_BGR2GRAY);

	// feature 생성
	orb->detectAndCompute(gray_1, Mat(), keyPoint_1, desc_1);
	orb->detectAndCompute(gray_2, Mat(), keyPoint_2, desc_2);

	matcher->match(desc_1, desc_2, matches);


	imshow("img", gray_1);

	drawMatches(gray_1, keyPoint_1, gray_2, keyPoint_2, matches, match_img);
	drawKeypoints(gray_1, keyPoint_1, feature_1);
	imshow("feature", feature_1);
	imshow("match", match_img);
	waitKey(0);

	cout<<"Hello"<<endl;

	return 0;
}
