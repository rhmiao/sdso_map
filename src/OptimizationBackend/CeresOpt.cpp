#include "CeresOpt.h"
#include "FullSystem/FullSystem.h"
#include "util/settings.h"
#include "util/globalCalib.h"

using namespace ceres;
using namespace std;

namespace dso
{

Eigen::Matrix2d DSOProjectionFactor::sqrt_info;
double DSOProjectionFactor::sum_t;
Eigen::Matrix3d DSOProjectionFactor::K;

DSOCeresOpt::DSOCeresOpt()
{
    DSOProjectionFactor::sqrt_info = (fxG[0] / 1.5 * Eigen::Matrix2f::Identity()).cast<double>();
    DSOProjectionFactor::K=KG[0].cast<double>();
}

DSOCeresOpt::~DSOCeresOpt()
{

}

void DSOCeresOpt::optimization(std::vector<FrameHessian*> fhs)
{
    if(fhs.size() < 2) return;
#if USE_CERES
        ceres::Problem problem;
        ceres::LossFunction *loss_function;
        loss_function = new ceres::HuberLoss(1.0);
//        loss_function = new ceres::CauchyLoss(1.0);

        ceres::Solver::Options options;

        options.linear_solver_type = ceres::DENSE_SCHUR;
        //options.num_threads = 2;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 6;
        //options.use_explicit_schur_complement = true;
        options.minimizer_progress_to_stdout = true;
        //options.use_nonmonotonic_steps = true;
        TicToc t_solver;
        for(FrameHessian* fh : fhs)
        {
            Eigen::Vector3d P=fh->shell->camToWorld.translation();
            Eigen::Quaterniond Q=fh->shell->camToWorld.unit_quaternion();
            para_Pose[fh->idx][0] = P[0];
            para_Pose[fh->idx][1] = P[1];
            para_Pose[fh->idx][2] = P[2];
            para_Pose[fh->idx][3] = Q.x();
            para_Pose[fh->idx][4] = Q.y();
            para_Pose[fh->idx][5] = Q.z();
            para_Pose[fh->idx][6] = Q.w();
        }
        int co = 0;
        for(FrameHessian* fh : fhs)
            for(PointHessian* ph : fh->pointHessians)
                for(PointFrameResidual* r : ph->residuals)
                {
                    inv_d[co][0]=r->point->ceres_idepth[0];
                    if(r->host->idx!=r->target->idx){
                    DSOProjectionFactor *f = new DSOProjectionFactor(r->point,r->host,r->target);
                    problem.AddResidualBlock(f, loss_function, para_Pose[r->host->idx], para_Pose[r->target->idx], inv_d[co]);
                    co++;
                    }
                }
        co=0;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);std::cout<<"aa"<<std::endl;
         cout << summary.BriefReport() << endl;
//        ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
//        ROS_DEBUG("solver costs: %f", t_solver.toc());
#endif
}


DSOProjectionFactor::DSOProjectionFactor(  dso::PointHessian*  _pts_i,  dso::FrameHessian*  _i_cam, dso::FrameHessian*  _j_cam) :
    pts_i (_pts_i) ,i_cam(_i_cam), j_cam(_j_cam)
{


};

bool DSOProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double inv_dep_i = parameters[2][0];

    Eigen::Vector2d gradI;

    for(int i=0;i<patternNum;++i){
        Eigen::Vector3d new_pts_i = K.inverse()*Eigen::Vector3f(pts_i->u+patternP[i][0],pts_i->v+patternP[i][1],1).cast<double>();
        Eigen::Vector3d pts_camera_i = new_pts_i / inv_dep_i;
        Eigen::Vector3d pts_w = Qi * pts_camera_i + Pi;
        Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_w - Pj);

        double dep_j = pts_camera_j.z();
        Eigen::Vector3d unit_pts_camera_j = pts_camera_j / dep_j;
        Eigen::Vector3d new_pts_j(KG[0].cast<double>()*unit_pts_camera_j);

        Eigen::Vector3d hitColor = getInterpolatedElement33(j_cam->dI, new_pts_j(0), new_pts_j(1), wG[0]).cast<double>();
        if(i==4)gradI=hitColor.tail<2>();
        double w = sqrt((double)(setting_outlierTHSumComponent) / ((double)setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
        w = 0.5f*(w + (double)pts_i->weights[i]);
        residuals[0] += (w*(hitColor[0] - (pts_i->color)[i]));
    }

    Eigen::Vector3d new_pts_i=K.inverse()*Eigen::Vector3f(pts_i->u,pts_i->v,1).cast<double>();
    Eigen::Vector3d pts_camera_i = new_pts_i / inv_dep_i;
    Eigen::Vector3d pts_w = Qi * pts_camera_i + Pi;
    Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_w - Pj);
    double dep_j = pts_camera_j.z();
    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce23(2, 3);
        Eigen::Matrix<double, 1, 3> reduce(1, 3);

        reduce23 << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

        reduce = gradI.transpose() * sqrt_info * reduce23 * K;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() =  Rj.transpose();
            jaco_i.rightCols<3>() =  Rj.transpose() * Ri * -Utility::skewSymmetric(pts_camera_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() =   -Rj.transpose();
            jaco_j.rightCols<3>() =   Utility::skewSymmetric(pts_camera_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce *   Rj.transpose() * Ri * new_pts_i * -1.0 / (inv_dep_i * inv_dep_i);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

void DSOProjectionFactor::check(double **parameters)
{
    double *res = new double[15];
    double **jaco = new double *[4];
    jaco[0] = new double[2 * 7];
    jaco[1] = new double[2 * 7];
    jaco[2] = new double[2 * 7];
    jaco[3] = new double[2 * 1];
    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    //    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
    //              << std::endl;
    //    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
    //              << std::endl;
    //    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
    //              << std::endl;
    //    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
    //              << std::endl;
    //    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[3]) << std::endl
    //              << std::endl;

    //    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    //    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    //    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    //    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    //    double inv_dep_i = parameters[3][0];

    //    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    //    Eigen::Vector3d pts_w = Qi * pts_camera_i + Pi;
    //    Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_w - Pj);


    //    Eigen::Vector2d residual;
    //    double dep_j = pts_camera_j.z();
    //    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
    //    residual = sqrt_info * residual;

    //    puts("num");
    //    std::cout << residual.transpose() << std::endl;

    //    const double eps = 1e-6;
    //    Eigen::Matrix<double, 2, 19> num_jacobian;
    //    for (int k = 0; k < 19; k++)
    //    {
    //        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    //        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    //        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    //        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    //        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    //        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    //        double inv_dep_i = parameters[3][0];

    //        int a = k / 3, b = k % 3;
    //        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

    //        if (a == 0)
    //            Pi += delta;
    //        else if (a == 1)
    //            Qi = Qi * Utility::deltaQ(delta);
    //        else if (a == 2)
    //            Pj += delta;
    //        else if (a == 3)
    //            Qj = Qj * Utility::deltaQ(delta);
    //        else if (a == 4)
    //            tic += delta;
    //        else if (a == 5)
    //            qic = qic * Utility::deltaQ(delta);
    //        else if (a == 6)
    //            inv_dep_i += delta.x();

    //        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    //        Eigen::Vector3d pts_w = Qi * pts_camera_i + Pi;
    //        Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_w - Pj);

    //        Eigen::Vector2d tmp_residual;
    //        double dep_j = pts_camera_j.z();
    //        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
    //        tmp_residual = sqrt_info * tmp_residual;
    //        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    //    }
    //    std::cout << num_jacobian << std::endl;
}


// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
    template <typename T> bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

int example() {
//  google::InitGoogleLogging(argv[0]);
#if USE_CERES
  // The variable to solve for with its initial value.
  double initial_x = 5.0;
  double x = initial_x;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, NULL, &x);

  // Run the solver!
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
#endif
  return 0;
}
}

