#include <iostream>
#include <sstream>
#include <stdexcept>
#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

#define PATCH_SIZE 10
#define TOO_CLOSE 4
#define WAIT_TIME 5

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

void drawFeaturePoints(Mat& image, vector<Point2f> features, Scalar color)
{
  for(size_t i = 0; i < features.size(); i++)
  {
    Point2f feature_point = features[i];
    circle(image, feature_point, 3, color, -1, 8, 0);
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
    for(size_t j = 0; j < b.features.size(); j++)
    {
      Point2f b_f = b.features[i];
      double score = dist(a_f, b_f);
      if(score < best_match && score > TOO_CLOSE &&
         a_f.x >= b_f.x && a_f.y >= b_f.y)
      {
        best_match = score;
        closest.a = a_f;
        closest.b = b_f;
      }
    }
  }
  return closest;
}

Mat stitch_movie(const char* filename)
{
  VideoCapture cap(filename);
  if(!cap.isOpened())
  {
    stringstream ss;
    ss << "Cannot open file: " << filename;
    throw runtime_error(ss.str());
  }

  Mat result, previous_frame, previous_gray, old_result;
  Frame previous;

  namedWindow("result", 1);
  namedWindow("previous", 1);
  namedWindow("old_result", 1);
  namedWindow("display", 1);
  namedWindow("color", 1);
  namedWindow("gray", 1);

  for(int frame_number = 0; ;frame_number++)
  {
    Frame current;
    Mat display;
    
    cap >> current.color;
    
    if(current.color.empty())
    {
      return result;
    }
    current.color = current.color(Range(1, current.color.rows-1), 
                                  Range(1, current.color.cols-1));

    display = current.color.clone();
      
    cvtColor(current.color, current.gray, CV_BGR2GRAY);
    GaussianBlur(current.gray, current.gray, Size(5, 5), 4, 4);
    goodFeaturesToTrack(current.gray, current.features, 20, .01, 20);

    if(frame_number > 0)
    {
      drawFeaturePoints(display, current.features, Scalar(0, 255, 0));
      drawFeaturePoints(display, previous.features, Scalar(255, 255, 0));

      PointPair most_similar = find_best_match(current, previous);
      circle(display, most_similar.a, 3, Scalar(0,0,255), -1, 8, 0);
      circle(display, most_similar.b, 3, Scalar(255,0,0), -1, 8, 0);
      cout << "a: " << most_similar.a << " b:" << most_similar.b << endl;

      int distance = dist(most_similar.a, most_similar.b);
      if(distance < 15 && distance > 0)
      {
        old_result = result;
        int row_diff = abs(most_similar.a.y - most_similar.b.y);
        int col_diff = abs(most_similar.a.x - most_similar.b.x);
        result = Mat::zeros(old_result.rows + row_diff, old_result.cols + col_diff, old_result.type());

        Range current_row_range, current_col_range,
          previous_row_range, previous_col_range;
        if (most_similar.a.y > most_similar.b.y)
        {
          cout << "most_similar.a.y > most_similar.b.y" << endl;
          current_row_range = Range(0, current.color.rows);
          previous_row_range = Range(row_diff, old_result.rows + row_diff); 
        }
        else
        {
          cout << "most_similar.a.y <= most_similar.b.y" << endl;
          current_row_range = Range(row_diff, row_diff+current.color.rows);
          previous_row_range = Range(0, old_result.rows); 
        }

        if (most_similar.a.x > most_similar.b.x)
        {
          cout << "most_similar.a.x > most_similar.b.x" << endl;
          current_col_range = Range(0, current.color.cols);
          previous_col_range = Range(col_diff, old_result.cols + col_diff); 
        }
        else
        {
          cout << "most_similar.a.x <= most_similar.b.x" << endl;
          current_col_range = Range(col_diff, col_diff+current.color.cols);
          previous_col_range = Range(0, old_result.cols); 
        }

        result(previous_row_range, previous_col_range) += old_result;
        result(current_row_range, current_col_range) -= 
          result(current_row_range, current_col_range);
        result(current_row_range, current_col_range) += current.color;
        
        drawRectangleForRanges(old_result, previous_row_range, 
                               previous_col_range, Scalar(255, 0, 0));
        drawRectangleForRanges(old_result, current_row_range, 
                               current_col_range, Scalar(0, 0, 255));

        previous = current;
      }
      else
      {
        cout << "Distance was greater than 10" << endl;
        previous = current;
      }
    }
    else if(frame_number == 0)
    {
      result = current.color.clone();
      
      old_result = result;
      previous = current;
    }

    
    imshow("result", result);
    imshow("old_result", old_result);
    imshow("display", display);
    imshow("color", current.color);
    imshow("gray", current.gray);
    
    int key = waitKey(WAIT_TIME);
    if (key == 27)
      break;
  }

  return result;
}

int main(int , char** )
{
  string files[11] = {"moon_stripe3.avi", "moon_stripe8.avi", "moon_stripe1.avi", 
                      "moon_stripe4.avi", "moon_stripe9.avi", "moon_stripe10.avi", 
                      "moon_stripe5.avi", "moon_stripe11.avi", "moon_stripe6.avi",
                      "moon_stripe2.avi", "moon_stripe7.avi"};
  
  vector<Mat> stitched_swipes;
  for(size_t i = 0; i < 11; i++)
  {
    Mat image = stitch_movie(files[i].c_str());
    imwrite(files[i]+".png", image);
  }

  return 0;
}
