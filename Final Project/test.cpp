#include <iostream>
#include "cv.h"
#include "highgui.h"
#include "util.cpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
  VideoCapture cap("moon_stripe1.avi"); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  Frame results;
  SiftFeatureDetector detector;

  namedWindow("result", 1);
  namedWindow("display", 1);
  namedWindow("color", 1);
  namedWindow("gray", 1);

  for(int frame_number = 0; ;frame_number++)
  {
    int waitTime = 0;
    Frame current;
    cap >> current.color;
    
    // Slice one row and one column from each side
    current.color = current.color(Range(1, current.color.rows-1), 
                                  Range(1, current.color.cols-1));

    cout << "Frame number: " << frame_number << endl;

    // Find features for current frame
    cvtColor(current.color, current.gray, CV_BGR2GRAY);
    detector.detect(current.color, current.features); 
    
    if(frame_number > 0)
    {
      // Find features in the accumulated image
      detector.detect(results.color, results.features); 

      PointPair most_similar = find_best_match(current, results);

      Mat display = results.color.clone();
      drawCirclesOnFeaturePoints(display, results.features);

      // circle(display, most_similar.a, 3, Scalar(255,0,0), -1, 8, 0);
      // circle(display, most_similar.b, 3, Scalar(0,0,255), -1, 8, 0);
      cout << "most_similar.a: " << most_similar.a
           << "most_similar.b: " << most_similar.a << endl;

      imshow("display", display);

      if(dist(most_similar.a, most_similar.b) < 10)
      {
        Mat old_results = results.color.clone();
        int row_diff = abs(most_similar.a.y - most_similar.b.y);
        int col_diff = abs(most_similar.a.x - most_similar.b.x);
        results.color = Mat::zeros(old_results.rows + row_diff, 
                                  old_results.cols + col_diff, 
                                  old_results.type());

        Range current_row_range, current_col_range, previous_row_range, previous_col_range;
        if (most_similar.a.y > most_similar.b.y)
        {
          cout << "most_similar.a.y > most_similar.b.y" << endl;
          current_row_range = Range(0, current.color.rows);
          previous_row_range = Range(row_diff, old_results.rows + row_diff); 
        }
        else
        {
          cout << "most_similar.a.y <= most_similar.b.y" << endl;
          current_row_range = Range(row_diff, row_diff+current.color.rows);
          previous_row_range = Range(0, old_results.rows); 
        }

        if (most_similar.a.x > most_similar.b.x)
        {
          cout << "most_similar.a.x > most_similar.b.x" << endl;
          current_col_range = Range(0, current.color.cols);
          previous_col_range = Range(col_diff, old_results.cols + col_diff); 
        }
        else
        {
          cout << "most_similar.a.x <= most_similar.b.x" << endl;
          current_col_range = Range(col_diff, col_diff+current.color.cols);
          previous_col_range = Range(0, old_results.cols); 
        }

        results.color(previous_row_range, previous_col_range) += old_results;
        results.color(current_row_range, current_col_range) -= results.color(current_row_range, current_col_range);
        results.color(current_row_range, current_col_range) += current.color;
        
        drawRectangleForRanges(old_results, previous_row_range, previous_col_range, Scalar(255, 0, 0));
        drawRectangleForRanges(old_results, current_row_range, current_col_range, Scalar(0, 0, 255));
      }
      else
      {
        cout << "Distance was greater than 10" << endl;
      }
    }
    else if(frame_number == 0)
    {
      results.color = current.color.clone();
    }

    imshow("results", results.color);
    imshow("color", current.color);
    imshow("gray", current.gray);
    
    int key = waitKey(waitTime);
    if (key == 27)
      break;
  }


  return 0;
}
