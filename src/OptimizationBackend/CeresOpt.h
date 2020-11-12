#ifndef _CERESOPT__H
#define _CERESOPT__H


#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "glog/logging.h"
#include "util/utility.h"
#include "util/tic_toc.h"
#include "FullSystem/HessianBlocks.h"
#include <ctime>
#include <cstdlib>
#include <chrono>

namespace dso
{
int example();
class DSOCeresOpt
{
public:
    DSOCeresOpt();
    ~DSOCeresOpt();
    void optimization(std::vector<FrameHessian*> fhs);
    double para_Pose[10][7];
    double inv_d[100000][1];

};


class DSOProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 1>
{
public:
    DSOProjectionFactor(  dso::PointHessian*  _pts_i,  dso::FrameHessian*  _i_cam, dso::FrameHessian*  _j_cam);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    dso::PointHessian* pts_i;
    dso::FrameHessian* i_cam;
    dso::FrameHessian* j_cam;
    static Eigen::Matrix2d sqrt_info;
    static Eigen::Matrix3d K;
    static double sum_t;
};
}

#endif
