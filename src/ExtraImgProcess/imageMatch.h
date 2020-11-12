#ifndef __IMAGEMATCH_H
#define __IMAGEMATCH_H

#include "util/NumType.h"
#include <opencv2/opencv.hpp>
#if CV_VERSION_MAJOR == 3
#include "opencv2/xfeatures2d.hpp"
#else
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/ocl.hpp"
#include <opencv2/nonfree/features2d.hpp>
#endif

#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"

class ImageMatch{
public:
    ImageMatch()
    {
#if CV_VERSION_MAJOR == 3
        detector=cv::FastFeatureDetector ::create();
        detectorGFTT= cv::GFTTDetector::create();//using this detector because it is good for affine
        computeDescriptors=cv::xfeatures2d::SiftDescriptorExtractor::create();
#else
        detector=new cv::OrbFeatureDetector;
        detectorGFTT=new cv::GFTTDetector;//using this detector because it is good for affine
        computeDescriptors=new cv::SiftDescriptorExtractor;
#endif

        descriptorMatcher=new cv::BFMatcher(cv::NORM_L2,false);//new cv::FlannBasedMatcher;

        cK=cv::Mat::zeros(3,3,CV_32F);
        cD=cv::Mat::zeros(1,5,CV_32F);

        cK.at<float>(0,0) = dso::fxG[dso::pyrLevelsUsed-1];
        cK.at<float>(1,1)= dso::fyG[dso::pyrLevelsUsed-1];
        cK.at<float>(0,2)= dso::cxG[dso::pyrLevelsUsed-1];
        cK.at<float>(1,2)= dso::cyG[dso::pyrLevelsUsed-1];
        cK.at<float>(2,2)= 1;

        ifx=1.0/cK.at<float>(0,0);
        ify=1.0/cK.at<float>(1,1);

        depth=cv::Mat::zeros(dso::wG[0],dso::hG[0],CV_32F);

    }
    ~ImageMatch()
    {
#if CV_VERSION_MAJOR == 2
        delete detector;
        delete detectorGFTT;
        delete computeDescriptors;
#endif
        delete descriptorMatcher;
        std::vector<cv::DMatch>().swap(filteredMatches);
        cK.release();
        cD.release();
        depth.release();
    }

    #if CV_VERSION_MAJOR == 3
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::FeatureDetector> detectorGFTT;
    cv::Ptr<cv::DescriptorExtractor> computeDescriptors;
    #else
    cv::FeatureDetector *detector;
    cv::FeatureDetector *detectorGFTT;
    cv::DescriptorExtractor *computeDescriptors;
    #endif
    cv::DescriptorMatcher *descriptorMatcher;
    std::vector<cv::DMatch > filteredMatches;
    cv::Mat cK;
    cv::Mat cD;
    cv::Mat rvec;
    cv::Mat tvec;
    float ifx;
    float ify;
    bool isBlur;

    cv::Subdiv2D subdiv;
    cv::Mat depth;

    int ratioTest(std::vector<std::vector<cv::DMatch> >  &matches );

    void crossCheckMatching( const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& filteredMatches12, int knn=2);

    cv::Mat match(const cv::Mat& descriptors1, const cv::Mat& descriptors2
                  ,std::vector<cv::KeyPoint>& KeyPointsRef,std::vector<cv::KeyPoint>& KeyPointsNew,
               std::vector<cv::Point2f>& pointsRefFix ,std::vector<cv::Point2f>& pointsNewFix );

    cv::Mat match(const cv::Mat& descriptors1, const cv::Mat& descriptors2,std::vector<cv::KeyPoint>& KeyPointsRef,std::vector<cv::KeyPoint>& KeyPointsNew,
               std::vector<cv::KeyPoint>& pointsRefFix ,std::vector<cv::KeyPoint>& pointsNewFix );

    cv::Mat match(const cv::Mat& descriptors1, const cv::Mat& descriptors2,std::vector<cv::KeyPoint>& KeyPointsRef,std::vector<cv::KeyPoint>& KeyPointsNew);

    bool matchForLoopclose(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2,int wl,int hl,int lvl,std::vector<std::vector<float>> idepthPairVec);

    void calcFeature(dso::FrameHessian* Frame,int wl,int hl,int lvl);

    void calcHWithFeature(dso::FrameHessian* firstFrame, dso::FrameHessian* newFrame,int wl,int hl,int lvl,dso::Mat33 K_Eigen,dso::SE3 &refToNew,bool doMatch);

    int trackingWithOFStereo(dso::FrameHessian* Frame1,int wl,int hl,int lvl,
                                          std::vector<cv::KeyPoint> &KeyPointsRef,std::vector<cv::Point3f> &KeyPoints3d,float &idepthScale,cv::Mat idepthMat=cv::Mat(),int idpthflag=0);
    dso::SE3 trackingWithOF(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2,int wl,int hl,int lvl,
                            std::vector<cv::KeyPoint> &KeyPointsRef,std::vector<cv::Point3f> &KeyPoints3d,Eigen::Vector2f &relAff,dso::SE3 init_pose,int &flag);

    void getTriangleVetexInDelaunay(cv::Subdiv2D& subdiv, cv::Point2f fp,std::vector<cv::Point2f> &output);

    float getInterpDepthInDelaunay(Eigen::Vector3f target,std::vector<Eigen::Vector3f> triangle);

    void getScaleDepthFromStereo(dso::FrameHessian* Frame,int wl,int hl,int lvl);

    float traceOnRight(dso::PointHessian* ph,int lvl,dso::Vec2f hostToFrame_affine);

    void calcLinePara(std::vector<cv::Point2f> pts, float &a, float &b, float &c, float &res);

    bool getSample(std::vector<int> set, std::vector<int> &sset);

    void fitLineRANSAC(std::vector<cv::Point2f> ptSet, float &a, float &b, float &c, std::vector<bool> &inlierFlag);

    void calcCornersInSelectPts(dso::FrameHessian* Frame,std::vector<dso::ImmaturePoint*> phs,int lvl);

    void calcCornersInSelectPts(dso::FrameHessian* Frame,std::vector<dso::PointHessian*> phs,int lvl);

};


#endif
