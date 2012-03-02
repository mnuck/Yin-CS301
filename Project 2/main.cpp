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

#define OCCLUSION_COST      -10
#define HALF_PATCH_HEIGHT   2
#define TOLERANCE           64

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
                         const float occlusion_cost){
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

double NCC(const Mat& f_hat, const Mat& g_hat){
  return -f_hat.dot(g_hat);
}

void DSI_method(const Mat& left, const Mat& right, 
                Mat& dest, const int half_patch_height,
                const float occlusion_cost){
  int patch_height = half_patch_height*2+1;

  dest = Mat(left.size(), CV_8U, Scalar(0));

  namedWindow("pathWin", CV_WINDOW_AUTOSIZE);
  namedWindow("costWin", CV_WINDOW_AUTOSIZE);
  namedWindow("accWin", CV_WINDOW_AUTOSIZE);

  for(int i = 0; i < left.rows-patch_height; i++) {
    Mat l_row, r_row, l_patch, r_patch, l_patch_norm, r_patch_norm; 
    Mat dsi(left.cols, right.cols, CV_32FC1, Scalar(INFINITY));

    Mat l_row_norm, r_row_norm;
    l_row = left.rowRange(i, i+patch_height);
    r_row = right.rowRange(i, i+patch_height);

    for(int j = 0; j < left.cols-patch_height; j++) {
      l_patch = l_row.colRange(j, j + patch_height);
      intensity_normalize(l_patch, l_patch_norm);
      for(int k = (j >= TOLERANCE? j-TOLERANCE : 0); k < j; k++) {
        r_patch = r_row.colRange(k, k + patch_height);
        intensity_normalize(r_patch, r_patch_norm);
        dsi.at<float>(k, j) = NCC(l_patch_norm, r_patch_norm);
      }
    }

    cout << "Computed row: " << i << endl;

    Mat costs, paths;
    compute_cost_matrix(dsi, costs, paths, occlusion_cost);

    // Draw Paths
    Mat temp(paths.size(), CV_8UC3);
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

    bool done = false;
    int x=paths.cols-patch_height-1, y=paths.rows-patch_height-1;
    while(!done){
      Point pt = Point(x, y);
      unsigned char val = paths.at<unsigned char>(pt);

      // Add path to image
      rectangle(temp, pt, pt, Scalar(0, 255, 255));

      switch (val){
      case DONE:
        done = true;
        break;
      case MATCH:
        dest.at<unsigned char>(i, x) = (unsigned char)abs((y - x)*255/64);
        x--; y--;
        break;
      case OCCLUDED_FROM_LEFT:
        x--;
        break;
      case OCCLUDED_FROM_RIGHT:
        y--;
        break;
      default:
        cout << "OH NO!" << endl;
        break;
      }
    }

    // Fill in blanks.
    unsigned char prev = 0;
    for(int x = 0; x < dest.cols; x++){
      unsigned char current = dest.at<unsigned char>(i, x);
      if(current == 0){
        dest.at<unsigned char>(i, x) = prev;
      }
      else {
        prev = current;
      }
    }

    // Show path image
    double min, max;
    Mat temp2, temp3;
    minMaxLoc(costs, &min, &max);
    temp2 = (abs(min)+costs)/25000*255;

    temp2.convertTo(temp2, CV_8U);
    dest.convertTo(temp3, CV_8U);
    imshow("pathWin", temp);
    imshow("costWin", temp2);
    imshow("accWin", temp3);
    waitKey(1);
  }  
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

  Mat left_image=left_image_gray, right_image=right_image_gray;

  // Make them 32 bit signed...
  left_image.convertTo(left_image, CV_32SC1);
  right_image.convertTo(right_image, CV_32SC1);

  int height    = left_image.rows;
  int width     = left_image.cols;
  int channels  = left_image.channels();

  cout << "Height: " << height << endl
       << "Width: " << width << endl
       << "Channels: " << channels << endl;    

  Mat dsi;
  DSI_method(left_image, right_image, dsi, HALF_PATCH_HEIGHT, OCCLUSION_COST);

  // show the image
  namedWindow("dsiWin", CV_WINDOW_AUTOSIZE);
  dsi.convertTo(dsi, CV_8U);
  imshow("dsiWin", dsi);

  // wait for a key
  cout << "Waiting for key press" << endl;
  waitKey(0);

  return 0;
}
