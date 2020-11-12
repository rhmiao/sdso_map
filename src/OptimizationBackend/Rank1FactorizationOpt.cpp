#include "Rank1FactorizationOpt.h"
#include "Eigen/SVD"

namespace dso
{

//p3d is 2-d point expression in 3-d space with the last value equal to 1
Vec3 Rank1FactorizationOpt::getVijFromRT(dso::SE3 pose, Vec3 p3d)
{
    Mat33 Rji=pose.rotationMatrix();
    Vec3 Tji= pose.translation();
    Vec3 p3dj=Rji.transpose()*Tji;

    //solution of equation (4) of GSLAM
    Mat22 A;
    Vec2 f,x;
    A(0,0)=Tji.transpose()*Tji;
    A(0,1)=Tji.transpose()*p3dj;
    A(1,0)=p3dj.transpose()*Tji;
    A(1,1)=p3dj.transpose()*p3dj;

    f(0)=Tji.transpose()*p3d;
    f(1)=p3dj.transpose()*p3d;

    x=A.inverse()*f;

    //equation for (3)(5)(6)
    Vec3 cji_1=x(0)*Tji;
    Vec3 cji_2=p3d-x(1)*p3dj;
    Vec3 vij=0.5*(cji_1+cji_2);

    return vij;
}
void Rank1FactorizationOpt::addFrame(SE3 pose,std::vector<Vec2> Vp2d)
{
    std::vector<Vec3> Vv;
    for(int i=0;i<Vp2d.size();++i)
    {
        Vec2 p2d(Vp2d[i]);
        Vec3 p3d;
        p3d.head(2)=p2d;
        p3d(2)=1.0;
        Vec3 v=getVijFromRT(pose, p3d);
        Vv.push_back(v);
    }
    M.push_back(Vv);
}

void Rank1FactorizationOpt::addFrame(SE3 pose,std::vector<Vec3> Vp3d)
{
    std::vector<Vec3> Vv;
    for(int i=0;i<Vp3d.size();++i)
    {
        Vec3 v=getVijFromRT(pose, Vp3d[i]);
        Vv.push_back(v);
    }
    M.push_back(Vv);
}

void Rank1FactorizationOpt::optimize()
{
    LOG(INFO) << "[Rank1FactorizationOpt]:optimize:"<<3*M.size()<<" "<<M[0].size();
    const int rows=3*M.size();
    const int cols=M[0].size();
    Eigen::MatrixXd matrixM(rows,cols);
    for(int i=0;i<rows/3;++i)
        for(int j=0;j<cols;++j)
              matrixM.block<3,1>(3*i,j)=M[i][j];

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrixM, Eigen::ComputeThinU | Eigen::ComputeThinV );
    Eigen::MatrixXd pos=svd.matrixU().col(0);
    Eigen::MatrixXd invD=svd.matrixV().col(0);

    vPos.clear();
    vInvDepth.clear();

    Vec3 posTmp;
    for(int i=0;i<rows/3;++i){
        posTmp(0)=pos(3*i);
        posTmp(1)=pos(3*i+1);
        posTmp(2)=pos(3*i+2);
        vPos.push_back(posTmp);
    }

    for(int i=0;i<cols;++i)vInvDepth.push_back(1/invD(i));

    M.clear();
}
}
