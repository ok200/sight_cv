#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
//for surf http://y-takeda.tumblr.com/post/41255703041/opencv-sift-surf
#include <opencv2\nonfree\features2d.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include <cv.h>
//#include <highgui.h>

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
	Mat colorImage;
	Mat grayImage;
	vector<KeyPoint> kp_vec;
	vector<float> desc_vec;

	//CvCapture* capture;
	VideoCapture capture(0);
	// カメラを初期化
	//if ((capture = cvCreateCameraCapture(0)) == NULL) {
	//    cerr << "cannot find camera" << endl;
	//    return -1;
	//}
	if(capture.isOpened()==NULL){
		cerr << "cannot find camera" << endl;
		return 0;
	}

	// ウィンドウを生成
	//cvNamedWindow("SURF");
	namedWindow("SURF",CV_WINDOW_AUTOSIZE);

	//IplImage* captureImage = cvQueryFrame(capture);
	//Mat colorImage;
	//capture.read(colorImage);
	while (true) {
		capture >> colorImage;
		//imshow("SURF", colorImage);

		// (2)convert Color Image to Grayscale for Feature Extraction
		cvtColor(colorImage, grayImage, CV_BGR2GRAY);

		// (3)initialize SURF class
		SURF calc_surf = SURF(500,2,2,true);

		// (4)extract SURF     
		calc_surf(grayImage, Mat(), kp_vec, desc_vec);

		// (5)draw keypoints
		cout << "Image Keypoints: " << kp_vec.size() << endl;
		vector<KeyPoint>::iterator it = kp_vec.begin(), it_end = kp_vec.end();
		for(; it!=it_end; ++it) {
			circle(colorImage, Point(it->pt.x, it->pt.y), 
				saturate_cast<int>(it->size*0.25), Scalar(255,255,0));
		}
		imshow("SURF", colorImage);

		// ESCキーが押されたらループを抜ける
		int key = cvWaitKey(30);
		if (key == 27) {
			break;
		}
	}

	return 0;
}