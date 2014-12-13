#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
//for surf http://y-takeda.tumblr.com/post/41255703041/opencv-sift-surf
#include <opencv2/nonfree/features2d.hpp>
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
    // ÉJÉÅÉâÇèâä˙âª
    //if ((capture = cvCreateCameraCapture(0)) == NULL) {
    //    cerr << "cannot find camera" << endl;
    //    return -1;
    //}
    if(capture.isOpened()==NULL){
        cerr << "cannot find camera" << endl;
        return 0;
    }
    
    namedWindow("SURF",CV_WINDOW_AUTOSIZE);
    
    // PCAの準備
    // すでに読み込んだ点の数
    int num_keypoints = 0;
    // フレーム数
    int num_frames = 0;
    
    // 特徴量
    const int DIM=128; // 特徴量は128次元
    const int SAMPLES=3000; // 1000サンプル
    const int RDIM=3; // 圧縮後は3次元にする
    cv::Mat pca_src(SAMPLES, DIM, CV_32FC1);
    cv::Mat pca_result(DIM, DIM, CV_32FC1); // DIMとSAMPLESのうち小さい方
    
    // 3000 特徴点分のデータが溜まるまで待つ
    while (true) {
        num_frames += 1;
        capture >> colorImage;
        //imshow("SURF", colorImage);
        cvtColor(colorImage, grayImage, CV_BGR2GRAY);
        SURF calc_surf = SURF(1000,2,2,true);
        calc_surf(grayImage, Mat(), kp_vec, desc_vec);
        vector<KeyPoint>::iterator it = kp_vec.begin(), it_end = kp_vec.end();
        int j = 0;
        for(; it!=it_end; ++it) {
            circle(colorImage, Point(it->pt.x, it->pt.y),
                   saturate_cast<int>(it->size*0.25), Scalar(255,255,0));
            if(num_keypoints < SAMPLES){
                for(int i = 0; i < DIM; i++){
                    ((float*)pca_src.data)[num_keypoints * pca_src.cols + i] = desc_vec[i + j * DIM];
                }
                num_keypoints += 1;
            }
            j += 1;
        }
        if(num_keypoints >= SAMPLES){
            break;
        }
        imshow("SURF", colorImage);
        int key = cvWaitKey(30);
        if (key == 27) {
            break;
        }
    }
    
    cv::PCA pca(pca_src, cv::Mat(), CV_PCA_DATA_AS_ROW, 0);
    pca_result=pca.project(pca_src);
    
    while (true) {
        capture >> colorImage;
        cvtColor(colorImage, grayImage, CV_BGR2GRAY);
        SURF calc_surf = SURF(2000,2,2,true);
        
        calc_surf(grayImage, Mat(), kp_vec, desc_vec);
        cout << "Image Keypoints: " << kp_vec.size() << endl;
        vector<KeyPoint>::iterator it = kp_vec.begin(), it_end = kp_vec.end();
        int j = 0;
        for(; it!=it_end; ++it) {
            cv::Mat m1(1, DIM, CV_32FC1);
            // m1 Mat に移す
            for(int i = 0; i < DIM; i++){
                ((float*)m1.data)[i] = desc_vec[j * DIM + i];
            }
            cv::Mat result(1, DIM, CV_32FC1);
            // プロジェクション
            result = pca.project(m1);
            cout << fixed << ((float*)result.data)[0] << endl;
            
            circle(colorImage, Point(it->pt.x, it->pt.y),
                   saturate_cast<int>(it->size*0.25), Scalar(
                                                             floor(((float*)result.data)[0] * 128 + 128),
                                                             floor(((float*)result.data)[1] * 128 + 128),
                                                             floor(((float*)result.data)[2] * 128 + 128)
                                                             ));
            // ここで ((float*)result.data)[0] でPCA射影結果の 0 番目にアクセスできる (値域は -1 〜 1)
            j += 1;
        }
        imshow("SURF", colorImage);
        int key = cvWaitKey(30);
        if (key == 27) {
            break;
        }
    }
    
    
    return 0;
}