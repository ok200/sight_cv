#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
//for surf http://y-takeda.tumblr.com/post/41255703041/opencv-sift-surf
#include <opencv2\nonfree\features2d.hpp>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
//#include <cv.h>
//#include <highgui.h>
 
using namespace std;
using namespace cv;
 
 
int main(int argc, char *argv[])
{
  // (1)load Color Image
  const char *imagename = argc > 1 ? argv[1] : "hoge.jpg";
  Mat colorImage = imread(imagename,1);
  if(colorImage.empty())
    return -1;
 
  // (2)convert Color Image to Grayscale for Feature Extraction
  Mat grayImage;
  cvtColor(colorImage, grayImage, CV_BGR2GRAY);
 
  // (3)initialize SURF class
  SURF calc_surf = SURF(250,4,2,true);
 
  // (4)extract SURF
  vector<KeyPoint> kp_vec;
  vector<float> desc_vec;     
  calc_surf(grayImage, Mat(), kp_vec, desc_vec);
 
  // (5)draw keypoints
  cout << "Image Keypoints: " << kp_vec.size() << endl;
#if 1
  vector<KeyPoint>::iterator it = kp_vec.begin(), it_end = kp_vec.end();
  for(; it!=it_end; ++it) {
    circle(colorImage, Point(it->pt.x, it->pt.y), 
       saturate_cast<int>(it->size*0.25), Scalar(255,255,0));
  }
#else
  for(int i = 0; i < kp_vec.size(); i++) {
    KeyPoint* point = &(kp_vec[i]);
    Point center;  // Key Point's Center
    int radius;      // Radius of Key Point
    center.x = cvRound(point->pt.x);
    center.y = cvRound(point->pt.y);
    radius = cvRound(point->size*0.25);
    circle(colorImage, center, radius, Scalar(255,255,0), 1, 8, 0);
  }
#endif
 
  namedWindow("SURF",CV_WINDOW_AUTOSIZE);
  imshow("SURF", colorImage);
  waitKey(0);
 
  return 0;
}