#ifndef __DRAWMATCHES_H
#define __DRAWMATCHES_H

#include <opencv2/opencv.hpp>
using namespace cv;

void drawMatches( const Mat& img1, const std::vector<cv::Point2f>& keypoints1,
                  const Mat& img2, const std::vector<cv::Point2f>& keypoints2,
                   Mat& outImg,
                  const Scalar& matchColor, const Scalar& singlePointColor, int flags );

void draw_point(cv::Mat& img, cv::Point2f fp, cv::Scalar color);
void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color);
void draw_voronoi(cv::Mat& img, cv::Subdiv2D& subdiv);
void locate_point( cv::Mat& img, cv::Subdiv2D& subdiv, cv::Point2f fp, cv::Scalar active_color );

#endif
