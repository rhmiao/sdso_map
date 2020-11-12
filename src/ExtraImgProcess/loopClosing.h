#ifndef __LOOPCLOSING_H
#define __LOOPCLOSING_H
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"

#include <iostream>
#include <string>
// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include "imageMatch.h"


class LoopClosing{
private:
    DBoW3::Vocabulary voc;
    DBoW3::Database db,dbRecord;
    const DBoW3::EntryId noLoopID = -1;

public:
    int  relocLoopID;
    int loopDetectedType;
    DBoW3::QueryResults lastRet;
    DBoW3::QueryResults lastRetRecord;
    ImageMatch _imageMatch;
    dso::Vec3 lastTranslation;
    dso::Vec3 lastTranslationRecord;
    Eigen::Matrix<double,6,1> loopRelativePose;
    std::vector<std::vector<float>> idepthPairVec;

    LoopClosing(std::string settingFile = "/home/mrh/catkin_ws/src/cgdso/config/setting.xml"){
            std::cout << "Use DBoW file: " << dso::vocPath << std::endl << "Loading ..." << std::endl;
            voc.load(dso::vocPath);
            db.setVocabulary(voc);
            dbRecord.setVocabulary(voc);
            loopDetectedType = -1;
            std::cout << "Load finished." << std::endl;
    }

    ~LoopClosing(){
    }

    bool loopclose(dso::FrameHessian* curentFrameHessian,std::vector<dso::FrameHessian*> frameHessians,int wl,int hl,int lvl);
    bool loopclose(dso::FrameHessian* curentFrameHessian,std::vector<std::vector<dso::FrameHessian*>> recordFrameHessians,int wl,int hl,int lvl);
    void addFrameInDBoWDb(dso::FrameHessian* curentFrameHessian,bool isRecord=false);
    void loopOptimize(dso::FrameHessian* newloopFrameHessian,dso::FrameHessian* loopFrame,std::vector<dso::FrameHessian*> recordFrameHessians,std::vector<dso::FrameHessian*> currentFrameHessians,Eigen::Matrix<double,6,1> _loopRelativePose);
    void loopOptimizeWithLoadMap(dso::FrameHessian* newloopFrameHessian,dso::FrameHessian* loopFrame,std::vector<dso::FrameHessian*> recordFrameHessians,std::vector<dso::FrameHessian*> currentFrameHessians,Eigen::Matrix<double,6,1> _loopRelativePose);

};


#endif
