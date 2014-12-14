// ok200.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

//stdafx.h is needed to capture video using opencv
#include "stdafx.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <windows.h>
#include <concrt.h>
#include <vector>


#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
//for surf http://y-takeda.tumblr.com/post/41255703041/opencv-sift-surf
#include <opencv2\nonfree\features2d.hpp>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
//for soc http://www.naturalsoftware.jp/blog/7371
#include "osc/OscOutboundPacketStream.h"
#include "ip/UdpSocket.h"
//for video
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define ADDRESS "192.168.100.101"
#define PORT 12000
//#define ADDRESS "127.0.0.1"
//#define PORT 7000
#define OUTPUT_BUFFER_SIZE 10000//588//16384
#define SENTPOINTS_NUM 10
#define SEND_DURATION 200

// response comparison, for list sorting
bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
{
  if (first.size > second.size) return true;
  else return false;
}

using namespace std;
using namespace cv;


//http://iroirous.blog.fc2.com/blog-entry-110.html
//int _tmain(int argc, char* argv[])
int main(int argc, char* argv[])
{
	//arguments:cv
	Mat colorImage;
	Mat colorImage_beforeresize;
	Mat grayImage;
	vector<KeyPoint> kp_vec;

	vector<float> desc_vec;
	//arguments:osc
	UdpTransmitSocket transmitSocket( IpEndpointName( ADDRESS, PORT ) );
	//arguments:video
	bool isvideo=false;
	VideoCapture video;
	char imagename[30];
	//arguments:time
	SYSTEMTIME time;
	int lastframe_msec=0;
	int frameduration=0;

	if(argc>1){
		strcpy(imagename,  argv[1]);}
	else{
		strcpy(imagename, "fushimi.avi");
	}
	cout<<imagename<<endl;
	// カメラを初期化
	VideoCapture capture(0);
	if(capture.isOpened()==NULL){
		cerr << "cannot find camera. Load video" << endl;
		isvideo=true;
		//動画の読み込み

        video.open(imagename);
		if(!video.isOpened()) return -1;

	}

	// ウィンドウを生成
	//cvNamedWindow("SURF");
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
		if(isvideo){
			video >> colorImage_beforeresize;
			//フレームが空か、ボタンが押された時か一周したときに出る。
			if(colorImage_beforeresize.empty() || waitKey(30) >= 0 || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
				return 0;
			}
		}else{
        capture >> colorImage_beforeresize;
		}
		//resize
		resize(colorImage_beforeresize, colorImage, cv::Size(320, 240),INTER_LINEAR);
        
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
    

	//sending

    while (true) {
		GetSystemTime(&time);
		lastframe_msec=1000*time.wSecond+time.wMilliseconds;
		if(isvideo){
			video >> colorImage_beforeresize;
			//フレームが空か、ボタンが押された時か一周したときに出る。
			if(colorImage_beforeresize.empty() || waitKey(30) >= 0 || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
				return 0;
			}
		}else{
        capture >> colorImage_beforeresize;
			if(capture.isOpened()==NULL){
				cerr << "camera error"<<endl;
				cvWaitKey(0);
			}
		}
		cout<<"debug1"<<endl;
		//resize
		resize(colorImage_beforeresize, colorImage, cv::Size(320, 240),INTER_LINEAR);
        cvtColor(colorImage, grayImage, CV_BGR2GRAY);
        SURF calc_surf = SURF(2000,2,2,true);
        
        calc_surf(grayImage, Mat(), kp_vec, desc_vec);
		//sort according to size
		vector<KeyPoint> kp_vec_sorted=kp_vec;
		sort(kp_vec_sorted.begin(),kp_vec_sorted.end(),compare_response);
		int threshould=0;

		if((int)kp_vec_sorted.size()>9){
			threshould=kp_vec_sorted.at(9).size;
			//cout<<"original"<<kp_vec.at(0).size<<" "<<kp_vec.at(1).size<<" "<<kp_vec.at(2).size<<endl;
			//cout<<kp_vec_sorted.at(0).size<<" "<<kp_vec_sorted.at(1).size<<" "<<kp_vec_sorted.at(2).size<<endl;
		}


		cout<<"debug2"<<endl;

        vector<KeyPoint>::iterator it = kp_vec.begin(), it_end = kp_vec.end();
        int j = 0;

		//write down descriptor:header
		char buffer[OUTPUT_BUFFER_SIZE];
		//int bufsize=36+28+(int)kp_vec.size()*30;//36+28: defaultcharacter 30: average buf per a query
		//char* buffer=new char[ bufsize];
		osc::OutboundPacketStream p( buffer, OUTPUT_BUFFER_SIZE );
		//osc::OutboundPacketStream p( buffer,  bufsize );


		p << osc::BeginBundleImmediate
			<< osc::BeginMessage( "/frame" ) 
			<< min((int)kp_vec.size(),(int)SENTPOINTS_NUM);


		int counter=0;
        for(; it!=it_end; ++it) {
            cv::Mat m1(1, DIM, CV_32FC1);
            // m1 Mat に移す
            for(int i = 0; i < DIM; i++){
                ((float*)m1.data)[i] = desc_vec[j * DIM + i];
            }
            cv::Mat result(1, DIM, CV_32FC1);
            // プロジェクション
            result = pca.project(m1);
            //cout << fixed << ((float*)result.data)[0] << endl;
            
            circle(colorImage, Point(it->pt.x, it->pt.y),
                   saturate_cast<int>(it->size*0.25), Scalar(
                                                             floor(((float*)result.data)[0] * 128 + 128),
                                                             floor(((float*)result.data)[1] * 128 + 128),
                                                             floor(((float*)result.data)[2] * 128 + 128)
                                                             ));
            // ここで ((float*)result.data)[0]) でPCA射影結果の 0 番目にアクセスできる (値域は -1 〜 1)

			//write down descriptor:data
			if((counter<min((int)kp_vec.size(),(int)SENTPOINTS_NUM))&&(it->size>=threshould)){
			//intensity values
			p <<it->size;
			//feature values
			p << ((float*)result.data)[0]
				<< ((float*)result.data)[1]
				<< ((float*)result.data)[2]
				<< ((float*)result.data)[3]
				<< ((float*)result.data)[4]
				<< ((float*)result.data)[5];
			//rgb values
			cv::Mat3b dotImg = colorImage;
			cv::Vec3b bgr = dotImg(cv::Point(it->pt.x,it->pt.y));

			//point vlaues
			p<< (float)it->pt.x<<(float)it->pt.y; 
			counter+=1;
			//cv::Vec3b bgr = colorImage.at<cv::Vec3b>((int)it->pt.x,(int)it->pt.y);
			p << (float)bgr[2]<<(float)bgr[1]<<(float)bgr[0];
			//p <<bgr[0];


			}
            j += 1;
        }
        imshow("SURF", colorImage);

		cout<<"debug3"<<endl;
		//write down descriptor:ternminal
		p << osc::EndMessage
			<< osc::EndBundle;
		transmitSocket.Send( p.Data(), p.Size() );
		//cout<<(p.Size()-36-28)/(int)kp_vec.size()<<endl;
		//cout<<p.Size()<<endl;

		//time control
		GetSystemTime(&time);
		frameduration=1000*time.wSecond+time.wMilliseconds-lastframe_msec;
		if(frameduration<SEND_DURATION){
			cout<<SEND_DURATION-frameduration<<endl;
			if(SEND_DURATION-frameduration>200){//たまにめちゃ大きな値が入ることがある
				Sleep(200);
			}else{
				Sleep(SEND_DURATION-frameduration);
			}
		}

        int key = cvWaitKey(1);
        if (key == 27) {
            break;
        }
		cout<<1000*time.wSecond+time.wMilliseconds<<endl;
    }
    
    
    return 0;
}


