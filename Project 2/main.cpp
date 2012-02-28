#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

ostream& operator<<(ostream& out, const Scalar& s){
  CvScalar cvs = CvScalar(s);
  for(int i = 0; i < 4; i++)
    out << cvs.val[i] << ", ";

  return out;
}

void intensity_normalize(const Mat& matrix, Mat& normalized) {
  Scalar mean, stddev;
  Mat stddev_mat;

  meanStdDev(matrix, mean, stddev);
  stddev_mat = Mat(matrix.size(), matrix.type(), stddev);
  normalized = (matrix - mean) / stddev_mat;
}

double NCC(const Mat& f, const Mat& g){
  Mat f_hat, g_hat; 

  intensity_normalize(f, f_hat);
  intensity_normalize(g, g_hat);

  return f_hat.dot(g_hat);
}

Mat match_epipolar_lines(const Mat& f, const Mat& g, const int half_patch_height=4){
  Mat result(f.size(), CV_8U, Scalar(0));
  Mat temp;

  int patch_height = half_patch_height*2+1;
  double largest_score;
  int largest_index;
  int search_tolerance = 64;

  namedWindow("matchresult", CV_WINDOW_AUTOSIZE);

  for(int i = 0; i < f.rows-patch_height; i++) {
    cout << "Computing row: " << i << endl;
    for(int j = 0; j < f.cols-patch_height; j++) {
      Mat f_patch = f(Range(i, i + patch_height), Range(j, j + patch_height));
      bool run_once = false;
      for(int k = (j >= search_tolerance? j-search_tolerance : 0); 
          k < j+search_tolerance && k < (g.cols-patch_height); k++) {
        Mat g_patch = g(Range(i, i + patch_height), Range(k, k + patch_height));
        double score = NCC(f_patch, g_patch);
        if(score > largest_score || !run_once){
          run_once = true;
          largest_score = score;
          largest_index = k;
        }
      }
      result.at<unsigned char>(i, j) = (unsigned char)abs(largest_index - j)*3;
    }

    result.convertTo(temp, CV_8U);
    imshow("matchresult", temp);
    waitKey(1);
  }
  return result;
}


int main(int argc, char *argv[])
{
  if(argc != 3) {
    cout << "Usage: " << argv[0] << " <left image> <right image>" << endl;
    exit(0);
  }

  // load the images
  Mat left_image_color  = imread(argv[1]);
  Mat right_image_color = imread(argv[2]);

  // Make sure the files opened
  if(!left_image_color.data || !right_image_color.data) {
    cout << "Could not load one of the image files." << endl;
    exit(0);
  }

  Mat left_image_gray, right_image_gray;
  cvtColor(left_image_color, left_image_gray, CV_BGR2GRAY);
  cvtColor(right_image_color, right_image_gray, CV_BGR2GRAY);

  Mat left_image, right_image;
  GaussianBlur(left_image_gray, left_image, Size(3, 3), 30, 30);
  GaussianBlur(right_image_gray, right_image, Size(3, 3), 30, 30);

  // Make them 32 bit signed...
  left_image.convertTo(left_image, CV_32SC1);
  right_image.convertTo(right_image, CV_32SC1);

  int height    = left_image.rows;
  int width     = left_image.cols;
  int channels  = left_image.channels();

  cout << "Height: " << height << endl
       << "Width: " << width << endl
       << "Channels: " << channels << endl;    

  Mat match = match_epipolar_lines(left_image, right_image, 5);

  // show the image
  namedWindow("matchWin", CV_WINDOW_AUTOSIZE);
  match.convertTo(match, CV_8U);
  imshow("matchWin", match);

  // wait for a key
  waitKey(0);

  return 0;
}
