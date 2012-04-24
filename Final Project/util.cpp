#include <iostream>
#include "cv.h"
#include "highgui.h"

#define PATCH_SIZE 20

using namespace std;
using namespace cv;

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

void drawCirclesOnFeaturePoints(Mat& img, const vector<Point2f>& features)
{
  for(size_t i = 0; i < features.size(); i++)
  {
    circle(img, features[i], 3, Scalar(0,255,0), -1, 8, 0);
  }
}

void drawRectangleForRanges(Mat& img, const Range row_range, const Range col_range,
  const Scalar& color)
{
  Point upper(col_range.start, row_range.start);
  Point lower(col_range.end, row_range.end);
  rectangle(img, upper, lower, color);
}

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
