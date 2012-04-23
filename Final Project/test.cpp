#include <iostream>
#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

#define PATCH_SIZE 10

typedef struct
{
  Mat color;
  Mat gray;
  vector<Point2f> features;
} Frame;

typedef struct
{
  Point2f a;
  Point2f b;
} PointPair;

double dist(const Point2f& a, const Point2f& b)
{
  return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

bool patch_in_bounds(const Mat& matrix, const Point2f& pt)
{
  int x = pt.x, y = pt.y, rows = matrix.rows, cols = matrix.cols;
  return (x >= PATCH_SIZE) && (x < cols-PATCH_SIZE) && 
         (y >= PATCH_SIZE) && (y < rows-PATCH_SIZE);
}

double NCC_score(const Mat& orig_a, const Mat& orig_b)
{
  Scalar mean, stddev;
  Mat stddev_mat, norm_a, norm_b;

  meanStdDev(orig_a, mean, stddev);
  stddev_mat = Mat(orig_a.size(), orig_a.type(), stddev);
  norm_a = (orig_a - mean) / stddev_mat;

  meanStdDev(orig_b, mean, stddev);
  stddev_mat = Mat(orig_b.size(), orig_b.type(), stddev);
  norm_b = (orig_b - mean) / stddev_mat;
  
  return 1-norm_a.dot(norm_b);
}

Mat get_patch(const Mat& matrix, const Point2f pt)
{
  return matrix(Range(pt.y-PATCH_SIZE, pt.y+PATCH_SIZE), 
                Range(pt.x-PATCH_SIZE, pt.x+PATCH_SIZE));
}

PointPair find_best_match(const Frame& a, const Frame& b)
{
  double best_match = INFINITY;
  PointPair closest;
  for(size_t i = 0; i < a.features.size(); i++)
  {
    Point2f a_f = a.features[i];
    if (patch_in_bounds(a.gray, a_f))
    {
      Mat a_patch = get_patch(a.gray, a_f);
      for(size_t j = 0; j < b.features.size(); j++)
      {
        Point2f b_f = b.features[i];
        if(patch_in_bounds(b.gray, b_f))
        {
          Mat b_patch = get_patch(b.gray, b_f);
          double score = NCC_score(a_patch, b_patch);
          if(abs(score-1) < best_match)
          {
            best_match = score;
            closest.a = a_f;
            closest.b = b_f;
          }
        }
      }
    }
  }
  return closest;
}

int main(int, char**)
{
  VideoCapture cap("moon_test.avi"); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  Mat result, previous_frame, previous_gray, tmp;
  Frame previous;

  namedWindow("result", 1);
  namedWindow("display", 1);
  namedWindow("color", 1);
  namedWindow("gray", 1);

  for(int frame_number = 0; ;frame_number++)
  {
    int waitTime = 30;
    Frame current;
    cap >> current.color;

    Mat display = current.color.clone();

    cvtColor(current.color, current.gray, CV_BGR2GRAY);
    GaussianBlur(current.gray, current.gray, Size(9, 9), 2, 2);
    goodFeaturesToTrack(current.gray, current.features, 10, .01, 10);
    
    if(frame_number > 0)
    {
      for(size_t i = 0; i < current.features.size(); i++)
      {
        Point2f feature_point = current.features[i];
        circle(display, feature_point, 3, Scalar(0,255,0), -1, 8, 0);
      }

      PointPair most_similar = find_best_match(current, previous);
      circle(display, most_similar.a, 3, Scalar(0,0,255), -1, 8, 0);
      circle(display, most_similar.b, 3, Scalar(255,0,0), -1, 8, 0);
      cout << "a: (" << most_similar.a.x << ", " << most_similar.a.y << ") "
           << "b: (" << most_similar.b.x << ", " << most_similar.b.y << ") " << endl;

      if(dist(most_similar.a, most_similar.b) < 10)
      {
        tmp = result;
        int row_diff = abs(most_similar.a.y - most_similar.b.y);
        int col_diff = abs(most_similar.a.x - most_similar.b.x);
        result = Mat::zeros(tmp.rows + row_diff, tmp.cols + col_diff, tmp.type());

        result(Range(0, tmp.rows), Range(0, tmp.cols)) += tmp;

        result(Range(tmp.rows, tmp.rows+row_diff),
               Range(col_diff, col_diff+current.color.cols)) +=
          current.color(Range(current.color.rows - row_diff, current.color.rows), 
                        Range::all());
      }
      else
      {
        cout << "Distance was greater than 10" << endl;
        waitTime = 0;
      }
    }
    else if(frame_number == 0)
    {
      result = current.color.clone();
    }

    previous = current;

    imshow("result", result);
    imshow("display", display);
    imshow("color", current.color);
    imshow("gray", current.gray);
    
    int key = waitKey(0);
    if (key == 27)
      break;
  }


  return 0;
}
