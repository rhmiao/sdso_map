#ifndef __DECOMPHOMOGRAPHY_H
#define __DECOMPHOMOGRAPHY_H

#include "opencv2/opencv.hpp"
using namespace cv;

int decomposeHomographyMat(InputArray _H,
                       InputArray _K,
                       OutputArrayOfArrays _rotations,
                       OutputArrayOfArrays _translations,
                       OutputArrayOfArrays _normals);

void Triangulate(const cv::Point2f &p1,const cv::Point2f &p2, const cv::Mat P1,const cv::Mat P2, cv::Mat &x3D);
int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::Point2f> &vKeys1, const std::vector<cv::Point2f> &vKeys2,
                       const cv::Mat &K, float th2, float &parallax);

#endif
