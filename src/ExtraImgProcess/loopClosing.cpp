#include "loopClosing.h"
#include "util/FrameShell.h"
#include <opencv2/core/eigen.hpp>

bool LoopClosing::loopclose(dso::FrameHessian* curentFrameHessian,std::vector<dso::FrameHessian*> recordFrameHessians,int wl,int hl,int lvl) {
    bool loopdetected = false;
    dso::FrameHessian* lastKf = curentFrameHessian;
    cv::Mat checkDesc (lastKf->descriptors);
    DBoW3::QueryResults ret;
    db.query(checkDesc, ret, -1);
    dso::SE3 poseCurr = curentFrameHessian->shell->trackingRef->camToWorld * curentFrameHessian->shell->camToTrackingRef;
//    DBoW3::BowVector vkf1; voc.transform(checkDesc, vkf1);
//    DBoW3::BowVector vkf2; voc.transform((*(recordFrameHessians.end()-1))->descriptors, vkf2);
//    double r = voc.score(vkf1, vkf2);
//    for(int ii=0;ii<recordFrameHessians.size();++ii){
//        std::cout <<"pos:"<<ii<<" "<<recordFrameHessians[ii]->worldToCamMat.col(3)<<std::endl;
//    }

//    for(int ll=0;ll<ret.size();++ll)std::cout<<"pairs: "<<lastKf->recordID<<"    "<< ret[ll].Id<<" "<<ret[ll].Score<<std::endl;
    if(!ret.empty()){
        double noLoopScore = std::max(0.4-ret[0].Score,0.0);//we think score lager than 0.4 is 100% loop closing
        DBoW3::Result noLoopRt(noLoopID,noLoopScore);
        ret.push_back(noLoopRt);
    }
    double sumRet=0;
    for(DBoW3::Result _rt:ret){
        sumRet+=_rt.Score;
        LOG(INFO)<<"_rt.Score:"<<_rt.Id<<" "<<_rt.Score;
    }
    ret.scaleScores(1.0/sumRet);
    if(lastRet.size()==0){
        std::sort(ret.begin(), ret.end(), DBoW3::Result::ltId);
        lastRet=ret;
        lastTranslation = poseCurr.translation();
        return false;
    }

    DBoW3::QueryResults prob;
    dso::Vec3 dtranslation1 = poseCurr.translation()-lastTranslation;
    double normDt1=dtranslation1.squaredNorm();
    for(DBoW3::Result _rt:ret)
    {
        double pb=_rt.Score;
        if(_rt.Id==noLoopID)
        {
            DBoW3::Result _prob(_rt.Id,pb*(lastRet.back().Score));
            prob.push_back(_prob);
        }
        else
        {
            dso::vConnId ids = recordFrameHessians[_rt.Id]->connectID;
            double pb2=0;
            std::vector<double> scaleVec;
            for(dso::connId id:ids)
            {
                if(id.second<lastRet.size())
                {
                    dso::Vec3 dtranslation2 = recordFrameHessians[_rt.Id]->PRE_camToWorld.translation()-recordFrameHessians[id.second]->PRE_camToWorld.translation();
                    double normDt2=dtranslation2.squaredNorm(),maxNorm;
                    maxNorm = std::max(normDt1,normDt2);//maxNorm represent the similarity of distance
                    //similarity of direction and distance (dtranslation1.dot(dtranslation2)/(norm1*norm2)*(minNorm/maxNorm))
                    double crossCos = dtranslation1.dot(dtranslation2)/maxNorm;
                    double scale=0;
                    if(crossCos>0)scale = crossCos;
                    if(scale>0.866)scale=1;//cross angle less than 30 degree
                    scaleVec.push_back(scale);
                }
            }
            for(int i=0;i<ids.size();++i)
            {
                dso::connId id = ids[i];
                if(id.second<lastRet.size())pb2 += (scaleVec[i] * lastRet[id.second].Score);
            }
            DBoW3::Result _prob(_rt.Id,pb*pb2);
            prob.push_back(_prob);
        }

    }
    std::sort(prob.begin(), prob.end(), DBoW3::Result::ge);
    std::sort(ret.begin(), ret.end(), DBoW3::Result::ltId);
    for(DBoW3::Result _rt:prob)LOG(INFO)<<"prob.Score:"<<_rt.Id<<" "<<_rt.Score;
    lastRet = ret;
    lastTranslation = poseCurr.translation();
    unsigned int loopId = -1; double score = -1;
    if(!prob.empty()){
        loopId = prob[0].Id;
        score = prob[0].Score;
        std::cout <<" "<<lastKf->recordID << "\t" << loopId << "\t" << score << "\t"<<std::endl;
        std::cout <<std::endl;
        if(loopId != noLoopID && prob[0].Score/prob[1].Score >= 1){
            std::cout << ret[0] << std::endl;
            dso::FrameHessian * lkf1 = recordFrameHessians[loopId];
            dso::FrameHessian * lkf2 = lastKf;

            bool isMatcheSuccess = _imageMatch.matchForLoopclose(lkf1,lkf2, wl, hl,lvl,idepthPairVec);
            LOG(INFO)<<"isMatcheSuccess:"<<isMatcheSuccess<< "\t"<<std::endl;
            LOG(INFO) <<" "<<lastKf->frameID << "\t" << loopId << "\t" << score << "\t"<<std::endl;
            std::cout <<"isMatcheSuccess:"<<isMatcheSuccess<< "\t"<<std::endl;
           if(isMatcheSuccess) {
               cv::Mat r =_imageMatch.rvec ,t = _imageMatch.tvec;
               loopRelativePose(0)=t.at<double>(0,0);
               loopRelativePose(1)=t.at<double>(1,0);
               loopRelativePose(2)=t.at<double>(2,0);
               loopRelativePose(3)=r.at<double>(0,0);
               loopRelativePose(4)=r.at<double>(1,0);
               loopRelativePose(5)=r.at<double>(2,0);
                relocLoopID = loopId;
                loopdetected = true;
            } else{
                loopdetected = false;
            }
        } else{
            loopdetected = false;
        }
    } else{
        loopdetected = false;
    }

    if(loopdetected)loopDetectedType=0;
    else loopDetectedType=-1;
    assert(cv::countNonZero(checkDesc !=( lastKf->descriptors)) == 0);
    return loopdetected;
}
bool LoopClosing::loopclose(dso::FrameHessian* curentFrameHessian,std::vector<std::vector<dso::FrameHessian*>> recordFrameHessians,int wl,int hl,int lvl) {
    bool loopdetected = false;
    dso::FrameHessian* lastKf = curentFrameHessian;
    cv::Mat checkDesc (lastKf->descriptors);
    DBoW3::QueryResults ret;
    dbRecord.query(checkDesc, ret, -1);

    dso::SE3 poseCurr = curentFrameHessian->shell->trackingRef->camToWorld * curentFrameHessian->shell->camToTrackingRef;
    double sumRet=0;
    for(DBoW3::Result _rt:ret)sumRet+=_rt.Score;
    ret.scaleScores(1.0/sumRet);
    if(lastRetRecord.size()==0){
        std::sort(ret.begin(), ret.end(), DBoW3::Result::ltId);
        lastRetRecord=ret;
        lastTranslationRecord = poseCurr.translation();
        return false;
    }

    DBoW3::QueryResults prob;
    for(DBoW3::Result _rt:ret)
    {
        double pb=_rt.Score;
        int N = recordFrameHessians.size();
        int loopGroupID=0,loopSubID=_rt.Id;
        for(int i=0;i<N;++i)
        {
            if(loopSubID>=recordFrameHessians[i].size()) loopSubID=loopSubID-recordFrameHessians[i].size();
            else{
                loopGroupID=i;
                break;
            }
        }
        dso::vConnId ids = recordFrameHessians[loopGroupID][loopSubID]->connectID;
        double pb2=0;
        double scale = 1.0/ids.size();
        for(dso::connId id:ids)
        {
            int recordid=0;
            for(int i=0;i<id.first-1;++i)recordid+=recordFrameHessians[i].size();
            recordid+=id.second;
            pb2 += (scale * lastRetRecord[recordid].Score);
        }
        DBoW3::Result _prob(_rt.Id,pb*pb2);
        prob.push_back(_prob);
    }
    std::sort(prob.begin(), prob.end(), DBoW3::Result::ge);
    std::sort(ret.begin(), ret.end(), DBoW3::Result::ltId);
    lastRetRecord = ret;
    lastTranslationRecord = poseCurr.translation();

    int loopId = -1; double score = -1;
    if(!prob.empty()){
        loopId = prob[0].Id;
        score = prob[0].Score;
        int N = recordFrameHessians.size();
        int loopGroupID=0,loopSubID=loopId;
        for(int i=0;i<N;++i)
        {
            if(loopSubID>=recordFrameHessians[i].size()) loopSubID=loopSubID-recordFrameHessians[i].size();
            else{
                loopGroupID=i;
                break;
            }
        }
        std::cout <<" "<<lastKf->recordID << "\t" << loopId << "\t" << score << "\t"<<std::endl;
        std::cout <<std::endl;
        if(score >= 0){
            std::cout << ret[0] << std::endl;
            dso::FrameHessian * lkf1 = recordFrameHessians[loopGroupID][loopSubID];
            dso::FrameHessian * lkf2 = lastKf;

            bool isMatcheSuccess = _imageMatch.matchForLoopclose(lkf1,lkf2, wl, hl,lvl,idepthPairVec);
            LOG(INFO)<<"isMatcheSuccess:"<<isMatcheSuccess<< "\t"<<std::endl;
            LOG(INFO) <<" "<<lastKf->frameID << "\t" << loopId << "\t" << score << "\t"<<std::endl;
            std::cout <<"isMatcheSuccess:"<<isMatcheSuccess<< "\t"<<std::endl;
            if(isMatcheSuccess) {
                cv::Mat r =_imageMatch.rvec ,t = _imageMatch.tvec;
                loopRelativePose(0)=t.at<double>(0,0);
                loopRelativePose(1)=t.at<double>(1,0);
                loopRelativePose(2)=t.at<double>(2,0);
                loopRelativePose(3)=r.at<double>(0,0);
                loopRelativePose(4)=r.at<double>(1,0);
                loopRelativePose(5)=r.at<double>(2,0);
                relocLoopID = loopId;
                loopdetected = true;
            } else{
                loopdetected = false;
            }
        } else{
            loopdetected = false;
        }
    } else{
        loopdetected = false;
    }
    if(loopdetected){
        if(loopDetectedType==0)loopDetectedType=2;
        else loopDetectedType=1;
    }
    assert(cv::countNonZero(checkDesc !=( lastKf->descriptors)) == 0);
    return loopdetected;
}

void LoopClosing::addFrameInDBoWDb(dso::FrameHessian* curentFrameHessian,bool isRecord)
{
    if(isRecord) dbRecord.add(curentFrameHessian->descriptors);
    else db.add(curentFrameHessian->descriptors);
}

void LoopClosing::loopOptimize(dso::FrameHessian* newloopFrameHessian,dso::FrameHessian* loopFrame,std::vector<dso::FrameHessian*> recordFrameHessians,std::vector<dso::FrameHessian*> currentFrameHessians,Eigen::Matrix<double,6,1> _loopRelativePose)
{
    int loopID = relocLoopID;
    int currentFrameID = currentFrameHessians.back()->frameID;

    dso::SE3 poseLoop=loopFrame->PRE_camToWorld;
    dso::SE3 poseCurrent=newloopFrameHessian->shell->camToWorld;
//    dso::SE3 loopError = poseLoop*dso::SE3::exp(_loopRelativePose)*poseCurrent.inverse();
    dso::SE3 loopError = poseLoop*poseCurrent.inverse();
    LOG(INFO)<<"loopID:"<<" "<<loopID<<" "<<currentFrameID<<std::endl;
    LOG(INFO)<<"testopt:"<<" "<<poseLoop.matrix3x4()<<" "<<_loopRelativePose<<" "<<poseCurrent.matrix3x4()<<std::endl;

    dso::SE3::Tangent t=loopError.log();
    LOG(INFO)<<"loopError:"<<" "<<t<<std::endl;

    int M = currentFrameHessians.size();

    for(int i =0;i<M;++i)
    {
        dso::FrameHessian *Frame = currentFrameHessians[i];
        if(!Frame->isLoopOptimized){
            Frame->PRE_camToWorld = loopError*Frame->PRE_camToWorld;
            Frame->shell->camToWorld = Frame->PRE_camToWorld;
            Frame->PRE_worldToCam = Frame->PRE_camToWorld.inverse();
            Frame->worldToCam_evalPT = Frame->PRE_worldToCam;

            cv::eigen2cv(Frame->PRE_worldToCam.matrix3x4(),Frame->worldToCamMat);
        }
        Frame->isLoopOptimized = true;

    }

    int N = recordFrameHessians.size();

    for(int i=loopID;i<N;i++){

        dso::FrameHessian *Frame = recordFrameHessians[i];
        if(!Frame->isLoopOptimized){
            dso::SE3::Tangent dt(1.0*(i-loopID)/(N-loopID)*t);
            dso::SE3 dSE=dso::SE3::exp(dt);
            Frame->PRE_camToWorld = dSE*Frame->PRE_camToWorld;
            Frame->shell->camToWorld = Frame->PRE_camToWorld;
            Frame->PRE_worldToCam = Frame->PRE_camToWorld.inverse();
            Frame->worldToCam_evalPT = Frame->PRE_worldToCam;
            LOG(INFO)<<"loopError:"<<" "<<i<<" "<<dSE.matrix3x4()<<" "<<loopError.matrix3x4()<<std::endl;
            cv::eigen2cv(Frame->PRE_worldToCam.matrix3x4(),Frame->worldToCamMat);
            Frame->isLoopOptimized = true;
        }
    }



}

void LoopClosing::loopOptimizeWithLoadMap(dso::FrameHessian* newloopFrameHessian,dso::FrameHessian* loopFrame,std::vector<dso::FrameHessian*> recordFrameHessians,std::vector<dso::FrameHessian*> currentFrameHessians,Eigen::Matrix<double,6,1> _loopRelativePose)
{

    dso::SE3 poseLoop=loopFrame->PRE_camToWorld;
    dso::SE3 poseCurrent=newloopFrameHessian->shell->camToWorld;
    dso::SE3 loopError = poseLoop*poseCurrent.inverse();

//    std::cout<<"loop_with_load_map:"<<" "<<poseLoop.log()<<std::endl<<_loopRelativePose<<std::endl<<poseCurrent.log()<<std::endl;

    dso::SE3::Tangent t=loopError.log();
//    std::cout<<"loopError:"<<" "<<t<<std::endl;

    int M = currentFrameHessians.size();

    for(int i =0;i<M;++i)
    {
        dso::FrameHessian *Frame = currentFrameHessians[i];
        Frame->PRE_camToWorld = loopError*Frame->PRE_camToWorld;
        Frame->shell->camToWorld = Frame->PRE_camToWorld;
        Frame->PRE_worldToCam = Frame->PRE_camToWorld.inverse();
        Frame->worldToCam_evalPT = Frame->PRE_worldToCam;

        cv::eigen2cv(Frame->PRE_worldToCam.matrix3x4(),Frame->worldToCamMat);

    }

    int N = recordFrameHessians.size();

    for(int i=0;i<N;i++){
        dso::FrameHessian *Frame = recordFrameHessians[i];
        int flag =0 ;
        for(int j =0;j<M;++j)
        {
            if(Frame == currentFrameHessians[j]){
                flag=1;
                break;
            }
        }
        if(flag==1)continue;
        Frame->PRE_camToWorld = loopError*Frame->PRE_camToWorld;
        Frame->shell->camToWorld = Frame->PRE_camToWorld;
        Frame->PRE_worldToCam = Frame->PRE_camToWorld.inverse();
        Frame->worldToCam_evalPT = Frame->PRE_worldToCam;
        cv::eigen2cv(Frame->PRE_worldToCam.matrix3x4(),Frame->worldToCamMat);
    }


}

