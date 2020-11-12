#ifndef __MAP_H
#define __MAP_H

#include <iostream>
#include <string>
#include <sstream>
#include "FullSystem/HessianBlocks.h"


// OpenCV
#include <opencv2/opencv.hpp>
#if CV_VERSION_MAJOR == 3
#include "opencv2/xfeatures2d.hpp"
#else
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/ocl.hpp"
#include <opencv2/nonfree/features2d.hpp>
#endif

#include "BoostArchiver.h"
// for map file io
#include <fstream>
namespace dso{


typedef std::vector<std::vector<FrameHessian*>> vvFrameHessian;
typedef std::vector<FrameHessian*> vFrameHessian;
class FrameHessian;

class MapPool{

public:
    vvFrameHessian loadRecKeyFrameVVec;
    vFrameHessian currentRecKeyFrameVec;
    FrameHessian* newComeInFrame;
    cv::Mat K;
    int maxGroupID;
    std::vector<int> frameSize;

private:
    // serialize is recommended to be private
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version);
};

class Map{
private:


     std::string mapfile;
     bool is_save_map;

public:

    MapPool myMap;
    Map(std::string saveDir = "/home/mrh/Record/dsoMap/map01")
    {
        mapfile=saveDir;
        myMap.maxGroupID = -1;
    }

    ~Map(){

    }
    void addKeyFrame(FrameHessian* recordFrame);
    void addConnect(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2);
    void delConnect(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2);
    void updateConnect(std::vector<int> updateList,int minGroupId);
    void mergeKeyFrame();
    bool findId(std::set<int> visited,connId id,int base);
    vConnId searchPath(connId startId,connId endId);
    void mergeKeyFrameLoopInLoadRecFrames(int groupID,int subLoopID);
    void mergeKeyFrameLoopInCurRecFrames(int subLoopID);
    void eraseKeyFrame(int index);
    void addPoints();
    void erasePoints();
    void SaveMap();
    bool LoadMap();
    void setDir(std::string dir);

    bool loopclose(vFrameHessian frameHessians,int delay);
    void mapTest();


};

}
#endif
