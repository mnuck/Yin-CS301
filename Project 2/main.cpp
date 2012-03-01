#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

#define DONE                0
#define MATCH               1
#define OCCLUDED_FROM_LEFT  2
#define OCCLUDED_FROM_RIGHT 3

#define TOLERANCE            64

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

void compute_cost_matrix(const Mat& dsi, Mat& costs, Mat& paths, 
                         const float occlusion_cost=100){
  costs = Mat(dsi.size(), CV_32FC1, Scalar(INFINITY));
  paths = Mat(dsi.size(), CV_8UC1, Scalar(NAN));

  if(dsi.rows != dsi.cols) {
    throw "What? DSI isn't square.";
  }

  // Fill in borders
  for(int i = 0; i < dsi.cols; i++){
    costs.at<float>(i, 0) = i * occlusion_cost;
    paths.at<unsigned char>(i, 0) = OCCLUDED_FROM_RIGHT;

    costs.at<float>(0, i) = i * occlusion_cost;
    paths.at<unsigned char>(0, i) = OCCLUDED_FROM_LEFT;
  }

  paths.at<unsigned char>(0, 0) = DONE;

  float match_cost, occ_left_cost, occ_right_cost;
  for(int y = 1; y < dsi.rows; y++){
    for(int x = 1; x < dsi.cols; x++){
      match_cost     = costs.at<float>(x-1, y-1) + dsi.at<float>(x, y);
      occ_left_cost  = costs.at<float>(x, y-1) + occlusion_cost;
      occ_right_cost = costs.at<float>(x-1, y) + occlusion_cost;
      if(match_cost > occ_left_cost){
        if(occ_left_cost > occ_right_cost){
          // occ_right smallest
          costs.at<float>(x, y) = occ_right_cost;
          paths.at<unsigned char>(x, y) = OCCLUDED_FROM_RIGHT;
        }
        else{
          // occ_left smallest
          costs.at<float>(x, y) = occ_left_cost;
          paths.at<unsigned char>(x, y) = OCCLUDED_FROM_LEFT;
        }
      }
      else{
        if(match_cost > occ_right_cost){
          // occ_right smallest
          costs.at<float>(x, y) = occ_right_cost;
          paths.at<unsigned char>(x, y) = OCCLUDED_FROM_RIGHT;
        }
        else{
          // match_cost smallest
          costs.at<float>(x, y) = match_cost;
          paths.at<unsigned char>(x, y) = MATCH;
        }
      }
    } 
  }
}

double NCC(const Mat& f, const Mat& g){
  return 1-f.dot(g);
}

void DSI_method(const Mat& left, const Mat& right, 
                Mat& dest, const int half_patch_height=4){
  int patch_height = half_patch_height*2+1;

  dest = Mat(left.rows, right.rows, CV_32SC1, Scalar(0));

  // for(int i = 0; i < left.rows-patch_height; i++) {
  for(int i = 0; i < 1; i++) {
    Mat l_row, r_row; 
    Mat dsi(left.cols, right.cols, CV_32FC1, Scalar(INFINITY));

    intensity_normalize(left.rowRange(i, i+patch_height), l_row);
    intensity_normalize(right.rowRange(i, i+patch_height), r_row);

    for(int j = 0; j < left.cols-patch_height; j++) {
      Mat l_patch = l_row.colRange(j, j + patch_height);
      for(int k = (j >= TOLERANCE? j-TOLERANCE : 0); k < j; k++) {
        Mat r_patch = r_row.colRange(k, k + patch_height);
        dsi.at<float>(k, j) = NCC(l_patch, r_patch);
      }
    }

    double min, max;
    minMaxLoc(dsi, &min, &max);
    cout << "Computed row: " << i 
         << " Max score: " << max
         << " Min score: " << min << endl;

    Mat costs, paths, temp(dsi.size(), CV_8UC3);
    compute_cost_matrix(dsi, costs, paths);

    cout << "Non-zero: " << countNonZero(paths) << endl;
    cout << "Zero: " << paths.cols*paths.rows-countNonZero(paths) << endl;

    for (int y = 0; y < dsi.rows; y++){
      for (int x = 0; x < dsi.cols; x++){
        Point pt = Point(x, y);
        unsigned char val = paths.at<unsigned char>(pt);
        switch (val){
        case 0:
          rectangle(temp, pt, pt, Scalar(255, 255, 0));
          break;
        case 1:
          rectangle(temp, pt, pt, Scalar(0, 100, 0));
          break;
        case 2:
          rectangle(temp, pt, pt, Scalar(0, 0, 255));
          break;
        case 3:
          rectangle(temp, pt, pt, Scalar(255, 0, 0));
          break;
        default:
          break;
        }
        if (x == y)
          rectangle(temp, pt, pt, Scalar(0, 0, 0));
      }
    }

    namedWindow("paths", CV_WINDOW_AUTOSIZE);

    bool done = false;
    int x=paths.cols-patch_height-1, y=paths.rows-patch_height-1;
    while(!done){
      Point pt = Point(x, y);
      unsigned char val = paths.at<unsigned char>(pt);
      switch (val){
      case 0:
        cout << "DONE!" << endl;
        done = true;
        break;
      case 1:
        x--; y--;
        break;
      case 2:
        x--;
        break;
      case 3:
        y--;
        break;
      default:
        cout << "OH NO!" << endl;
        break;
      }
      rectangle(temp, pt, pt, Scalar(0, 255, 255));
      rectangle(costs, pt, pt, Scalar(0, 255, 255));
      imshow("paths", temp);
    }


    temp = dsi;
    temp.convertTo(temp, CV_8U);
    namedWindow("dsi", CV_WINDOW_AUTOSIZE);
    imshow("dsi", temp);

    minMaxLoc(costs, &min, &max);
    cout << "min " << min << " max " << max << endl;
    temp = 255*(costs+abs(min))/(30000+abs(min));
    temp.convertTo(temp, CV_8U);
    namedWindow("costs", CV_WINDOW_AUTOSIZE);
    imshow("costs", temp);
    

    cout << "Waiting for key press" << endl;
    waitKey(0);
  }  
}

Mat match_epipolar_lines(const Mat& f, const Mat& g, const int half_patch_height=4){
  Mat result(f.size(), CV_8U, Scalar(0));
  Mat temp;

  int patch_height = half_patch_height*2+1;
  double largest_score;
  int largest_index;

  namedWindow("matchresult", CV_WINDOW_AUTOSIZE);

  for(int i = 0; i < f.rows-patch_height; i++) {
    cout << "Computing row: " << i << endl;
    for(int j = 0; j < f.cols-patch_height; j++) {
      Mat f_patch = f(Range(i, i + patch_height), Range(j, j + patch_height));
      bool run_once = false;
      for(int k = (j >= TOLERANCE? j-TOLERANCE : 0); 
          k < j+TOLERANCE && k < (g.cols-patch_height); k++) {
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

  // Mat match = match_epipolar_lines(left_image, right_image, 5);

  Mat dsi;
  DSI_method(left_image, right_image, dsi);

  // // show the image
  // namedWindow("dsiWin", CV_WINDOW_AUTOSIZE);
  // dsi.convertTo(dsi, CV_8U);
  // imshow("dsiWin", dsi);

  // // wait for a key
  // cout << "Waiting for key press" << endl;
  // waitKey(0);

  return 0;
}
