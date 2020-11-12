#ifndef _RANK1FACTORIZATIONOPT__H
#define _RANK1FACTORIZATIONOPT__H


#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "util/settings.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace dso
{
class Rank1FactorizationOpt{
public:
    inline Rank1FactorizationOpt(){

    }

    inline ~Rank1FactorizationOpt(){

    }

    Vec3 getVijFromRT(SE3 pose, dso::Vec3 p3d);
    Vec2 getPjFromRT(SE3 pose, dso::Vec3 p3d,cv::Mat imgi,cv::Mat imgj);
    void addFrame(SE3 pose,std::vector<Vec2> Vp2d);
    void addFrame(SE3 pose,std::vector<Vec3> Vp3d);
    void optimize();

    std::vector<Vec3> vPos;
    std::vector<double> vInvDepth;

private:
    std::vector<std::vector<Vec3>> M;

};
}




#endif
