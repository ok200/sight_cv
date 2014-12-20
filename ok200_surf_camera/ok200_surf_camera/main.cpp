//win以外でコンパイルするときは次の行をコメントアウトしてください
//#include "stdafx.h"


//opencv library:1.  add include directory (property setting) 2. add library directory (property setting)
#include <opencv2/opencv.hpp>
//for surf http://y-takeda.tumblr.com/post/41255703041/opencv-sift-surf
#include <opencv2/nonfree/features2d.hpp>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
//for soc http://www.naturalsoftware.jp/blog/7371
//osc library:1.  add include directory (property setting) 2. add library directory (additional library directory in linker setting)
//3: add ws2_32.lib, winmm.lib, oscpack.lib to additional dependent file (additional library directory in linker setting)
#include <oscpack/osc/OscOutboundPacketStream.h>
#include <oscpack/ip/UdpSocket.h>
//for video
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//for face tracking
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <string>
//algorithm is needed to use min()
#include <algorithm>
#include <vector>


#ifdef TARGET_OS_MAC
#include <unistd.h>
#include <sys/time.h>
#elif defined __linux__
// Linux Includes Here
#error Can't be compiled on Linux yet
#elif defined WIN32 || defined _WIN64
// Windows Includes Here
//stdafx.h is needed to capture video using opencv
//windows is needed to use time function
#include <windows.h>
//windows is needed to use sort function of vector
#include <concrt.h>

#include <opencv2/opencv_lib.hpp>
#endif
 
using namespace std;
using namespace cv;

//information about get-time
//http://brian.pontarelli.com/2009/01/05/getting-the-current-system-time-in-milliseconds-with-c/

#define ADDRESS "192.168.100.101"
#define PORT 12000
//#define ADDRESS "127.0.0.1"
//#define PORT 7000
#define OUTPUT_BUFFER_SIZE 10000//588//16384
#define SENTPOINTS_NUM 10
#define SEND_DURATION 200
#define LOADED_FILE_FPS 30
//#define WRITEVIDEO
#define FACEDETECT
//#define YELLOWDETECT

// 1: WRITE TO FILE, 2: READ FROM FILE
#define PCA_CALIBRATE 1
#define PCA_PATH "pca_result.xml"

// response comparison, for list sorting
bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
{
  if (first.size > second.size) return true;
  else return false;
}

long getmillisec(){
    long millis;
#ifdef TARGET_OS_MAC
#include <sys/time.h>
timeval time;
gettimeofday(&time, NULL);
millis = (time.tv_sec * 1000) + (time.tv_usec / 1000);
#elif defined __linux__
// Linux Includes Here
#error Can't be compiled on Linux yet
#elif defined WIN32 || defined _WIN64
SYSTEMTIME time;
GetSystemTime(&time);
millis = (time.wSecond * 1000) + time.wMilliseconds;
// Windows Includes Here
#endif
    return millis;
}

void msleep(int msec){
#ifdef TARGET_OS_MAC
    
#elif defined __linux__
#error Can't be compiled on Linux yet
#elif defined WIN32 || defined _WIN64
    Sleep(msec);
#endif
    
}




//face detector(global variable)
 CascadeClassifier face_cascade;

 Mat detectAndDisplay( Mat frame )
{
  vector<Rect> faces;
  Mat frame_gray;

  cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
  cout<<faces.size()<<endl;
  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
  }
return frame;
 }


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
	//char imagename[30];
	string imagename;
	//arguments:time
	long lastframe_msec=0;
	long frameduration=0;
	int skip_frame=(int)(LOADED_FILE_FPS/(int)(1000/200));

	//arguments:file output
	string output_path;
	double outputFPS=30;
	//サイズがそろってないと書き込めない
	Size outputSize = Size(640,480);//cv::Size(1920,1080);
	//VideoWriter writer(output_path, CV_FOURCC_DEFAULT, outputFPS, outputSize);
	VideoWriter writer;
	//VideoWriter writer(output_path, CV_FOURCC('I', '4', '2', '0'), outputFPS, outputSize);
	if(argc>1){
		//strcpy(imagename,  argv[1]);
	imagename=string(argv[1]);
	}
	else{
		//strcpy(imagename, "hoge.avi");
		//strcpy(imagename, "redgate.avi");
		imagename=string("/Users/ryohei/gitrepos/ok200_cv/moviematrials/avi/tonnel.avi");
	}
	output_path=imagename+"_feature.avi";
	cout<<imagename<<endl;
#ifdef WRITEVIDEO
	writer.open(output_path, CV_FOURCC('M','J','P','G'), outputFPS, outputSize);
#endif

	// カメラを初期化
	VideoCapture capture(0);
	if(!capture.isOpened()){
		cerr << "cannot find camera. Load video" << endl;
		isvideo=true;
		//動画の読み込み

        video.open(imagename);
		if(!video.isOpened()) return -1;

	}

	// ウィンドウを生成
	namedWindow("SURF",CV_WINDOW_AUTOSIZE);
	   
    // PCAの準備
    // すでに読み込んだ点の数
    int num_keypoints = 0;
    // フレーム数
    int num_frames = 0;
    
    // 特徴量
    const int DIM=128; // 特徴量は128次元
#ifdef WRITEVIDEO
    const int SAMPLES=30;//3000; // 1000サンプル
#else
	 const int SAMPLES=3000;
#endif
    const int RDIM=3; // 圧縮後は3次元にする
    cv::Mat pca_src(SAMPLES, DIM, CV_32FC1);
    cv::Mat pca_result(DIM, DIM, CV_32FC1); // DIMとSAMPLESのうち小さい方


#ifdef FACEDETECT
	//if(!face_cascade.load("haarcascade_upperbody.xml")){
	if(!face_cascade.load("/Users/ryohei/gitrepos/ok200_cv/haarcascade_frontalface_alt.xml")){
		cout<<"cascade error"<<endl;
		return -1;
	}

#endif

    cv::PCA pca;
    
    // 3000 特徴点分のデータが溜まるまで待つ
    if(PCA_CALIBRATE){

    while (true) {
        num_frames += 1;
		if(isvideo){
#ifdef WRITEVIDEO
			//書き込むときは毎フレームとる
			video >> colorImage_beforeresize;
#else
			//5フレーム飛ばして6フレーム目を取る（5Hzで再生していて、もとの動画が30FPS）
			for(int frame=1;frame<skip_frame;frame++){
			video >> colorImage_beforeresize;
			}
#endif
		//フレームが空か、ボタンが押された時か一周したときに出る。
			if(colorImage_beforeresize.empty() || waitKey(30) >= 0 || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
				return 0;
			}

		}else{
			capture >> colorImage_beforeresize;
			if(!capture.isOpened()){
				cerr << "camera error"<<endl;
				cvWaitKey(0);
			}
		}

		//resize
#ifdef WRITEVIDEO
		colorImage=colorImage_beforeresize;
#else
		resize(colorImage_beforeresize, colorImage, cv::Size(320, 240),INTER_LINEAR);
#endif
        cv::cvtColor(colorImage, grayImage, CV_BGR2GRAY);
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
        imshow("SURF", colorImage);
#ifdef WRITEVIDEO
			//書き込む
			writer.write(colorImage);
#endif

        if(num_keypoints >= SAMPLES){
            break;
        }
        int key = cvWaitKey(30);
        if (key == 27) {
            break;
        }
    }
    pca = PCA(pca_src, cv::Mat(), CV_PCA_DATA_AS_ROW, 0);
    pca_result=pca.project(pca_src);
        char filename[] = PCA_PATH;	// file name
        // Open File Storage
        cv::FileStorage	cvfs(filename,CV_STORAGE_WRITE);
        cv::WriteStructContext ws(cvfs, "mat_array", CV_NODE_SEQ);	// create node
        cv::write(cvfs, "", pca.eigenvectors);
    }else{
        pca = PCA(pca_src, cv::Mat(), CV_PCA_DATA_AS_ROW, 0);
        char filename[] = PCA_PATH;	// file name
        cv::FileStorage cvfs(filename,CV_STORAGE_READ);
        
        // (3)read data from file storage
        cv::FileNode node(cvfs.fs, NULL); // Get Top Node
        cv::FileNode fn = node[string("mat_array")];
        
        for(int i = 0; i < fn.size(); i++){
            cv::read(fn[i], pca.eigenvectors);
        }
    }



    while (true) {
		lastframe_msec=getmillisec();
		if(isvideo){
#ifdef WRITEVIDEO
			//書き込むときは毎フレームとる
			video >> colorImage_beforeresize;
#else
			//5フレーム飛ばして6フレーム目を取る（5Hzで再生していて、もとの動画が30FPS）
			for(int frame=1;frame<skip_frame;frame++){
			video >> colorImage_beforeresize;
			}
#endif
			//フレームが空か、ボタンが押された時か一周したときに出る。
			if(colorImage_beforeresize.empty() || waitKey(30) >= 0 || video.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
				return 0;
			}
		}else{
        capture >> colorImage_beforeresize;
			if(!capture.isOpened()){
				cerr << "camera error"<<endl;
				cvWaitKey(0);
			}
		}
#ifdef WRITEVIDEO
		colorImage=colorImage_beforeresize;
#else
		resize(colorImage_beforeresize, colorImage, cv::Size(320, 240),INTER_LINEAR);
#endif
        cv::cvtColor(colorImage, grayImage, CV_BGR2GRAY);
        //SURF calc_surf = SURF(2000,2,2,true);
		SURF calc_surf = SURF(1000,2,2,true);
        //SURF calc_surf = SURF(600,2,2,true);
        calc_surf(grayImage, Mat(), kp_vec, desc_vec);
		//sort according to size
		vector<KeyPoint> kp_vec_sorted=kp_vec;
		std::sort(kp_vec_sorted.begin(),kp_vec_sorted.end(),compare_response);
		int threshould=0;

		if((int)kp_vec_sorted.size()>9){
			threshould=kp_vec_sorted.at(9).size;
			//cout<<"original"<<kp_vec.at(0).size<<" "<<kp_vec.at(1).size<<" "<<kp_vec.at(2).size<<endl;
			//cout<<kp_vec_sorted.at(0).size<<" "<<kp_vec_sorted.at(1).size<<" "<<kp_vec_sorted.at(2).size<<endl;
		}


		//cout<<"debug2"<<endl;

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
			//default
            /*circle(colorImage, Point(it->pt.x, it->pt.y),
                   saturate_cast<int>(it->size*0.25), Scalar(
                                                             floor(((float*)result.data)[0] * 128 + 128),
                                                             floor(((float*)result.data)[1] * 128 + 128),
                                                             floor(((float*)result.data)[2] * 128 + 128)
                                                             ));*/
			//if((rand()/(float)RAND_MAX)<0.1)
			cv::Mat3b dotImg = colorImage;
			cv::Vec3b bgr = dotImg(cv::Point(it->pt.x,it->pt.y));

#ifdef YELLOWDETECT
			//ヒューリスティックな設定（点字ブロック）
			if(bgr[2]>150&&(float)bgr[2]/bgr[1]<1.2&&(float)bgr[2]/bgr[1]>0.8){
#endif
			//circle: bgr
				circle(colorImage, Point(it->pt.x, it->pt.y),
					   saturate_cast<int>(it->size*0.25), Scalar(
																 bgr[0],
																 bgr[1],
																 bgr[2]
																 ),-1,CV_AA);
			//補色
			int cint=min(min(bgr[0],bgr[1]),bgr[2])+max(max(bgr[0],bgr[1]),bgr[2]);
				circle(colorImage, Point(it->pt.x, it->pt.y),
					   saturate_cast<int>(it->size*0.25), Scalar(
																 cint-bgr[0],
																 cint-bgr[1],
																 cint-bgr[2]
																 ),2);		
#ifdef YELLOWDETECT
			}
#endif
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
		

#ifdef FACEDETECT
		colorImage=detectAndDisplay(colorImage);
#endif
        imshow("SURF", colorImage);
		#ifdef WRITEVIDEO
			//書き込む
			writer.write(colorImage);
		#endif

		//cout<<"debug3"<<endl;
		//write down descriptor:ternminal
		p << osc::EndMessage
			<< osc::EndBundle;
		transmitSocket.Send( p.Data(), p.Size() );
		//cout<<(p.Size()-36-28)/(int)kp_vec.size()<<endl;
		//cout<<p.Size()<<endl;

		//time control
		frameduration=getmillisec()-lastframe_msec;
		if(frameduration<SEND_DURATION){
			//cout<<SEND_DURATION-frameduration<<endl;
			if(SEND_DURATION-frameduration>200){//たまにめちゃ大きな値が入ることがある intの値があふれることが原因? lastframe_msecとframedurationをlong型に変更した
				msleep(200);
			}else{
				msleep((unsigned int)SEND_DURATION-frameduration);
			}
		}

        int key = cvWaitKey(1);
        if (key == 27) {
            break;
        }
		//cout<<1000*time.wSecond+time.wMilliseconds<<endl;
    }
    
    writer.release();
	video.release();
	capture.release();
	return 0;
}


