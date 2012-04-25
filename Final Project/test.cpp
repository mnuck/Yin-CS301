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
      if(score < best_match && a_f.x > b_f.x && a_f.y > b_f.y)
      {
        best_match = score;
        closest.a = a_f;
        closest.b = b_f;
      }
    }
  }
  return closest;
}

int main(int, char**)
{
  VideoCapture cap("moon_stripe2.avi");
  if(!cap.isOpened())
    return -1;

  Mat result, previous_frame, previous_gray, tmp;
  Frame previous;

  namedWindow("result", 1);
  namedWindow("previous", 1);
  namedWindow("tmp", 1);
  namedWindow("display", 1);
  namedWindow("color", 1);
  namedWindow("gray", 1);

  for(int frame_number = 0; ;frame_number++)
  {
    cout << "13" << endl;
    int waitTime = 30;
    Frame current;
    Mat display;
    cout << "14" << endl;
    cap >> current.color;
    cout << "15" << endl;
    if(current.color.empty())
    {
      return 0;
    }
    current.color = current.color(Range(1, current.color.rows-1), 
                                  Range(1, current.color.cols-1));

    cout << "1" << endl;

    display = current.color.clone();
      
    cvtColor(current.color, current.gray, CV_BGR2GRAY);
    GaussianBlur(current.gray, current.gray, Size(7, 7), 2, 2);
    goodFeaturesToTrack(current.gray, current.features, 10, .1, 20);

    cout << "2" << endl;
    
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
      cout << "a: " << most_similar.a << " b:" << most_similar.b << endl;

      cout << "3" << endl;

      int distance = dist(most_similar.a, most_similar.b);
      if(distance < 15 && distance > 0)
      {
        tmp = result;
        int row_diff = abs(most_similar.a.y - most_similar.b.y);
        int col_diff = abs(most_similar.a.x - most_similar.b.x);
        result = Mat::zeros(tmp.rows + row_diff, tmp.cols + col_diff, tmp.type());

        cout << "4" << endl;

        Range current_row_range, current_col_range,
          previous_row_range, previous_col_range;
        if (most_similar.a.y > most_similar.b.y)
        {
          cout << "most_similar.a.y > most_similar.b.y" << endl;
          current_row_range = Range(0, current.color.rows);
          previous_row_range = Range(row_diff, tmp.rows + row_diff); 
        }
        else
        {
          cout << "most_similar.a.y <= most_similar.b.y" << endl;
          current_row_range = Range(row_diff, row_diff+current.color.rows);
          previous_row_range = Range(0, tmp.rows); 
        }

        cout << "5" << endl;        

        if (most_similar.a.x > most_similar.b.x)
        {
          cout << "most_similar.a.x > most_similar.b.x" << endl;
          current_col_range = Range(0, current.color.cols);
          previous_col_range = Range(col_diff, tmp.cols + col_diff); 
        }
        else
        {
          cout << "most_similar.a.x <= most_similar.b.x" << endl;
          current_col_range = Range(col_diff, col_diff+current.color.cols);
          previous_col_range = Range(0, tmp.cols); 
        }

        cout << "6" << endl;

        result(previous_row_range, previous_col_range) += tmp;
        result(current_row_range, current_col_range) -= 
          result(current_row_range, current_col_range);
        result(current_row_range, current_col_range) += current.color;
        
        cout << "7" << endl;

        drawRectangleForRanges(tmp, previous_row_range, 
                               previous_col_range, Scalar(255, 0, 0));
        drawRectangleForRanges(tmp, current_row_range, 
                               current_col_range, Scalar(0, 0, 255));

        cout << "8" << endl;

        previous = current;
      }
      else
      {
        cout << "Distance was greater than 10" << endl;
      }
    }
    else if(frame_number == 0)
    {
      cout << "9" << endl;
      result = current.color.clone();
      cout << "10" << endl;
      tmp = result;
      previous = current;
    }

    cout << "11" << endl;
    imshow("result", result);
    imshow("tmp", tmp);
    imshow("display", display);
    imshow("color", current.color);
    imshow("gray", current.gray);
    
    cout << "12" << endl;

    int key = waitKey(30);
    if (key == 27)
      break;
  }


  return 0;
}
