#include <stdlib.h>
#include <iostream>
#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
  if(argc != 3) {
    cout << "Usage: " << argv[0] << " <left image> <right image>" << endl;
    exit(0);
  }

  // load the images
  Mat left_image  = imread(argv[1]);
  Mat right_image = imread(argv[2]);

  // Make sure the files opened
  if(!left_image.data || !right_image.data) {
    cout << "Could not load one of the image files." << endl;
    exit(0);
  }

  // create a window
  namedWindow("mainWin", CV_WINDOW_AUTOSIZE); 

  int height    = left_image.rows;
  int width     = left_image.cols;
  int channels  = left_image.channels();

  cout << "Height: " << height << endl
       << "Width: " << width << endl
       << "Channels: " << channels << endl;

  Mat inverter(left_image.size(), left_image.type(), Scalar(255, 255, 255));

  left_image = inverter - left_image;

  // show the image
  imshow("mainWin", left_image);

  // wait for a key
  waitKey(0);

  return 0;
}
