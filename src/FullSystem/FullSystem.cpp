/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include "ExtraImgProcess/drawMatches.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;



FullSystem::FullSystem()
{

    int retstat =0;
    if(setting_logStuff)
    {

        retstat += system("rm -rf logs");
        retstat += system("mkdir logs");

        retstat += system("rm -rf mats");
        retstat += system("mkdir mats");

        calibLog = new std::ofstream();
        calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
        calibLog->precision(12);

        numsLog = new std::ofstream();
        numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
        numsLog->precision(10);

        coarseTrackingLog = new std::ofstream();
        coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
        coarseTrackingLog->precision(10);

        eigenAllLog = new std::ofstream();
        eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
        eigenAllLog->precision(10);

        eigenPLog = new std::ofstream();
        eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
        eigenPLog->precision(10);

        eigenALog = new std::ofstream();
        eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
        eigenALog->precision(10);

        DiagonalLog = new std::ofstream();
        DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
        DiagonalLog->precision(10);

        variancesLog = new std::ofstream();
        variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
        variancesLog->precision(10);


        nullspacesLog = new std::ofstream();
        nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
        nullspacesLog->precision(10);
    }
    else
    {
        nullspacesLog=0;
        variancesLog=0;
        DiagonalLog=0;
        eigenALog=0;
        eigenPLog=0;
        eigenAllLog=0;
        numsLog=0;
        calibLog=0;
    }

    dsoTrackingLog= new std::ofstream();
    dsoTrackingLog->open("logs/dsoTracking.txt", std::ios::trunc | std::ios::out);
    dsoTrackingLog->precision(10);

    gtTrackingLog= new std::ofstream();
    gtTrackingLog->open("logs/gtTracking.txt", std::ios::trunc | std::ios::out);
    gtTrackingLog->precision(10);

    dsotrackTimeAndPtsNumLog= new std::ofstream();
    dsotrackTimeAndPtsNumLog->open("logs/trackTimeAndPtsNum.txt", std::ios::trunc | std::ios::out);
    dsotrackTimeAndPtsNumLog->precision(10);

    dsotrackDposAndAvgidpthLog= new std::ofstream();
    dsotrackDposAndAvgidpthLog->open("logs/trackDeltaPosAndAvgidepth.txt", std::ios::trunc | std::ios::out);
    dsotrackDposAndAvgidpthLog->precision(10);


    assert(retstat!=293847);

    timeLast=0;

    selectionMap = new float[wG[0]*hG[0]];

    coarseDistanceMap = new CoarseDistanceMapWideAngle(wG[0], hG[0]);
    coarseTracker = new CoarseTrackerWideAngle(wG[0], hG[0]);
    coarseTracker_forNewKF = new CoarseTrackerWideAngle(wG[0], hG[0]);

//        coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
//        coarseTracker = new CoarseTracker(wG[0], hG[0]);
//        coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);

    coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
    pixelSelector = new PixelSelector(wG[0], hG[0]);

    statistics_lastNumOptIts=0;
    statistics_numDroppedPoints=0;
    statistics_numActivatedPoints=0;
    statistics_numCreatedPoints=0;
    statistics_numForceDroppedResBwd = 0;
    statistics_numForceDroppedResFwd = 0;
    statistics_numMargResFwd = 0;
    statistics_numMargResBwd = 0;

    lastCoarseRMSE.setConstant(100);

    currentMinActDist=2;
    initialized=false;


    ef = new EnergyFunctional();
    ef->red = &this->treadReduce;

    isLost=false;
    initFailed=false;
    relocMode=0;
    isProcessingKF =false;
    needNewIMUPreInte =false;
    needToRecordKF = false;
    Eigen::Vector3d G_in_cam(G[0],-G[2],G[1]);
    pre_integration = new IntegrationBase{Vec3(0,0,0), Vec3(0,0,0),G_in_cam,Vec3(0,0,0), Vec3(0,0,0)};

    needNewKFAfter = -1;

    linearizeOperation=true;
    runMapping=true;
    mappingThread = boost::thread(&FullSystem::mappingLoop, this);
    lastRefStopID=0;

    ceresOpt = new DSOCeresOpt();

    minIdJetVisDebug = -1;
    maxIdJetVisDebug = -1;
    minIdJetVisTracker = -1;
    maxIdJetVisTracker = -1;

    frameCount=0;
    reLocalizationMode=false;
    isLoadMap=false;
    loop_delay = 0;
    needToRecordKFCount=0;

    setting_maxShiftWeightT= 30.0f / (sqrtf(double(wG[0]*wG[0]+hG[0]*hG[0])));
    setting_maxShiftWeightR= 0.0f / (sqrtf(double(wG[0]*wG[0]+hG[0]*hG[0])));
    setting_maxShiftWeightRT= 10.0f / (sqrtf(double(wG[0]*wG[0]+hG[0]*hG[0])));
}

FullSystem::~FullSystem()
{
    blockUntilMappingIsFinished();

    if(setting_logStuff)
    {
        calibLog->close(); delete calibLog;
        numsLog->close(); delete numsLog;
        coarseTrackingLog->close(); delete coarseTrackingLog;
        //errorsLog->close(); delete errorsLog;
        eigenAllLog->close(); delete eigenAllLog;
        eigenPLog->close(); delete eigenPLog;
        eigenALog->close(); delete eigenALog;
        DiagonalLog->close(); delete DiagonalLog;
        variancesLog->close(); delete variancesLog;
        nullspacesLog->close(); delete nullspacesLog;
    }

    dsoTrackingLog->close();delete dsoTrackingLog;
    gtTrackingLog->close();delete gtTrackingLog;
    dsotrackTimeAndPtsNumLog->close();delete dsotrackTimeAndPtsNumLog;
    delete[] selectionMap;
    for(FrameShell* s : allFrameHistory)
        delete s;

    for(FrameHessian* fh:(recordMap.myMap.currentRecKeyFrameVec))
        delete fh;
    for(std::vector<FrameHessian*> fhVec:(recordMap.myMap.loadRecKeyFrameVVec))
        for(FrameHessian* fh:fhVec)
            if(std::find(recordMap.myMap.currentRecKeyFrameVec.begin(),recordMap.myMap.currentRecKeyFrameVec.end(),fh)==recordMap.myMap.currentRecKeyFrameVec.end())
                delete fh;
    for(FrameHessian* fh : unmappedTrackedFrames)
        delete fh;
    if (pre_integration != nullptr)
        delete pre_integration;

    delete coarseDistanceMap;
    delete coarseTracker;
    delete coarseTracker_forNewKF;
    delete coarseInitializer;
    delete pixelSelector;
    //    delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
    if(BInv==0) return;

    // copy BInv.
    memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


    // invert.
    for(int i=1;i<255;i++)
    {
        // find val, such that Binv[val] = i.
        // I dont care about speed for this, so do it the stupid way.

        for(int s=1;s<255;s++)
        {
            if(BInv[s] <= i && BInv[s+1] >= i)
            {
                Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
                break;
            }
        }
    }
    Hcalib.B[0] = 0;
    Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
    boost::unique_lock<boost::mutex> lock(trackMutex);
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    std::ofstream myfile;
    myfile.open (file.c_str());
    myfile << std::setprecision(15);

    for(FrameShell* s : allFrameHistory)
    {
        if(!s->poseValid) continue;

        if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

        myfile << s->timestamp <<
                  " " << s->camToWorld.translation().transpose()<<
                  " " << s->camToWorld.so3().unit_quaternion().x()<<
                  " " << s->camToWorld.so3().unit_quaternion().y()<<
                  " " << s->camToWorld.so3().unit_quaternion().z()<<
                  " " << s->camToWorld.so3().unit_quaternion().w() << "\n";
    }
    myfile.close();
}

void FullSystem::writePoseGt(FrameHessian* fh)
{

    fh->shell->viconPose.normalize();
    (*gtTrackingLog) << std::setprecision(16)
                         << fh->ab_exposure << " "
                         << fh->shell->viconPose.translation().transpose()<< " "
                         <<fh->shell->viconPose.log().transpose().tail<3>()<< "\n";
}

void FullSystem::writePoseDSO(FrameHessian* fh)
{
    //write to log file
    fh->PRE_camToWorld.normalize();
    (*dsoTrackingLog) << std::setprecision(16)
                         << fh->ab_exposure << " "
                         << fh->PRE_camToWorld.translation().transpose()<< " "
                         << fh->PRE_camToWorld.log().transpose().tail<3>() << "\n";
}

void FullSystem::writeTrackTimeAndPtsNum(double dt,int numPts,SE3 dpos)
{
    //write to log file
    dpos.normalize();
    (*dsotrackTimeAndPtsNumLog) << std::setprecision(10)
                         << dt<< " "
                         << numPts<< " "
                         << dpos.translation().transpose()<< " "
                         << dpos.log().transpose().tail<3>()<< "\n";

}

void FullSystem::writeDeltaPosAndAvgidpeth(Vec3 dpos,Vec3 drot,double avgidepth)
{
    //write to log file
    (*dsotrackDposAndAvgidpthLog) << std::setprecision(10)
                         << dpos[0]<< " "
                         << dpos[1]<< " "
                         << dpos[2]<< " "
                         << drot.norm() <<" "
                         << avgidepth<< "\n";
}

void FullSystem::stereoMatch( ImageAndExposure<float>* imagel,ImageAndExposure<float>* imager, int id,cv::Mat &idepthMap,std::vector<cv::Point3f> &KeyPoints3d)
{
    // =========================== add into allFrameHistory =========================
    FrameHessian* fh = new FrameHessian();
    FrameShell* shell = new FrameShell();
    shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = imagel->timestamp;
    shell->incoming_id = id; // id passed into DSO
    fh->shell = shell;

    // =========================== make Images / derivatives etc. =========================
    fh->ab_exposure = imagel->exposure_time;
    fh->makeImages(imagel->image, &Hcalib);
    fh->setRightImg(imager->image);

    FrameHessian* fhr = new FrameHessian();
    fhr->makeImages(imager->image,&Hcalib);
    int counter = 0,allcount=0;

    makeNewTracesForTestStereo(fh, 0);
    makeNewTracesForTestStereo(fhr,0);

    unsigned  char * idepthMapPtr = idepthMap.data;

    std::vector<cv::KeyPoint> KeyPointsRef;
    cv::Mat depth;
    float stereoscale;
    int lvl = 0;

    _imageMatch.calcFeature(fh,wG[lvl],hG[lvl],lvl);
    if(fh->dIr[0]!=0)_imageMatch.trackingWithOFStereo(fh,wG[lvl],hG[lvl],lvl,KeyPointsRef, KeyPoints3d,stereoscale,depth,0);
    _imageMatch.depth.copyTo(idepthMap);
    boost::posix_time::ptime lastTime = boost::posix_time::microsec_clock::local_time();
    for(ImmaturePoint* ph : fh->immaturePoints)
    {

        ImmaturePointStatus phTraceRightStatus = ph->traceStereo(fh->dIr, 0);

        float depth = 1.0f/ph->idepth_stereo;

        if(phTraceRightStatus == ImmaturePointStatus::IPS_GOOD  && depth > 0 && depth < 70)    //original u_stereo_delta 1 depth < 70
        {
            ph->idepth_min = ph->idepth_min_stereo;
            ph->idepth_max = ph->idepth_max_stereo;

            //            *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3) = ph->idepth_stereo;
            //            *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3 + 1) = ph->idepth_min;
            //            *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3 + 2) = ph->idepth_max;

            counter++;
        }
        allcount++;
    }
    boost::posix_time::time_duration elapsed = (boost::posix_time::microsec_clock::local_time() - lastTime);
    std::cout << "Time for traceStereo matching: " << elapsed.total_microseconds()/1.0e3 << "ms" << std::endl;
    //    std::sort(error.begin(), error.end());
    //    std::cout << 0.25 <<" "<<error[error.size()*0.25].first<<" "<<
    //              0.5 <<" "<<error[error.size()*0.5].first<<" "<<
    //              0.75 <<" "<<error[error.size()*0.75].first<<" "<<
    //              0.1 <<" "<<error.back().first << std::endl;

    //    for(int i = 0; i < error.size(); i++)
    //        std::cout << error[i].first << " " << error[i].second.first << " " << error[i].second.second << std::endl;

    std::cout<< " got good matches: " << counter <<" "<<allcount<< std::endl;

    delete fh;
    delete fhr;

    return;
}

Vec5 FullSystem::trackNewCoarse(FrameHessian* fh,bool useInitialPose)
{

    assert(allFrameHistory.size() > 0);
    // set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);

    FrameHessian* lastF = coarseTracker->lastRef;

    AffLight aff_last_2_l = AffLight(0,0);

    int64 start = cv::getTickCount();
    std::vector<cv::Point3f> KeyPoints3d;
    std::vector<cv::KeyPoint> KeyPointsRef;
    int trackingFlag = 0;
    for(cv::KeyPoint kpt:lastF->keyPoints)
    {
        cv::Point3f pt3;
        if(kpt.response>0)
        {
            KeyPointsRef.push_back(kpt);
            pt3.z=1/kpt.response;
            pt3.x=(kpt.pt.x-dso::cxG[featuresLvl])*dso::fxiG[featuresLvl]*pt3.z;
            pt3.y=(kpt.pt.y-dso::cyG[featuresLvl])*dso::fyiG[featuresLvl]*pt3.z;
            KeyPoints3d.push_back(pt3);
//            std::cout<<"kk depth:"<<pt3.z<<std::endl;
        }
    }

    std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
    FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
    FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
    SE3 slast_2_sprelast;
    SE3 lastF_2_slast;
    SE3 lastF_2_fh_features;
    if(allFrameHistory.size() == 2)
        lastF_2_fh_tries.push_back(SE3());
    else
    {
        {	// lock on global pose consistency!
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
            lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
            aff_last_2_l = slast->aff_g2l;
        }
        SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.
        Eigen::Vector2f matchRelAff;
        lastF_2_fh_features=_imageMatch.trackingWithOF(lastF,fh,wG[featuresLvl],hG[featuresLvl],featuresLvl,KeyPointsRef,KeyPoints3d,matchRelAff, fh_2_slast.inverse()*lastF_2_slast,trackingFlag);
        int64 end = cv::getTickCount();
        std::cout << "The differences of lastF_2_fh_features: " << 1000.0*(end - start)/cv::getTickFrequency()<<" ms"<< std::endl;
        std::cout<<"lastF_2_fh_features:"<<lastF_2_fh_features.translation().transpose()<<std::endl
                <<(fh_2_slast.inverse() * lastF_2_slast).translation().transpose()<<std::endl
                <<(fh->shell->viconPose.inverse()*lastF->shell->viconPose).translation().transpose()<<std::endl;
        // get last delta-movement.
        if(useInitialPose)lastF_2_fh_tries.push_back(fh->PRE_camToWorld.inverse()*lastF->shell->camToWorld);
        if(trackingFlag){
//            lastF_2_fh_tries.push_back(lastF_2_fh_features);
//            aff_last_2_l=dso::AffLight(lastF->aff_g2l().a-log(matchRelAff[0]),matchRelAff[0]*lastF->aff_g2l().b-matchRelAff[1]);
        }
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
        lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
        lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
        lastF_2_fh_tries.push_back(rotationNow.inverse()*rotationLast);
        lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

        // just try a TON of different initializations (all rotations). In the end,
        // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
        // also, if tracking rails here we loose, so we really, really want to avoid that.
                for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
                {
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
                }

        if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
        {
            lastF_2_fh_tries.clear();
            lastF_2_fh_tries.push_back(SE3());
        }
    }

    Vec6 flowVecs = Vec6::Constant(100);
    SE3 lastF_2_fh = SE3();
    AffLight aff_g2l = AffLight(0,0);


    // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    // I'll keep track of the so-far best achieved residual for each level in achievedRes.
    // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations=0;
    for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
    {
        AffLight aff_g2l_this = aff_last_2_l;
        SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
        bool trackingIsGood = coarseTracker->trackNewestCoarse(
                    fh, lastF_2_fh_this, aff_g2l_this,
                    pyrLevelsUsed-1,
                    achievedRes);	// in each level has to be at least as good as the last try.
        tryIterations++;

        if(i != 0)
        {
            printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
                   i,
                   i, pyrLevelsUsed-1,
                   aff_g2l_this.a,aff_g2l_this.b,
                   achievedRes[0],
                    achievedRes[1],
                    achievedRes[2],
                    achievedRes[3],
                    achievedRes[4],
                    coarseTracker->lastResiduals[0],
                    coarseTracker->lastResiduals[1],
                    coarseTracker->lastResiduals[2],
                    coarseTracker->lastResiduals[3],
                    coarseTracker->lastResiduals[4]);
        }


        // do we have a new winner?
        if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
        {
            //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
            flowVecs = coarseTracker->lastFlowIndicators;
            aff_g2l = aff_g2l_this;
            lastF_2_fh = lastF_2_fh_this;
            haveOneGood = true;
            std::cout<<"lastF_2_fh_features:"<<lastF_2_fh_this.translation().transpose()<<std::endl;

        }

        // take over achieved res (always).
        if(haveOneGood)
        {
            for(int i=0;i<5;i++)
            {
                if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
                    achievedRes[i] = coarseTracker->lastResiduals[i];
            }
        }


        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

    }
    fh->avgIdepth=flowVecs[3];

    if(!haveOneGood)
    {
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
        flowVecs = Vec6::Constant(0);
        aff_g2l = aff_last_2_l;
        lastF_2_fh = lastF_2_fh_tries[0];
    }

    lastCoarseRMSE = achievedRes;
    {
        boost::unique_lock<boost::mutex> imulock(imuIntegrationMutex);
        SE3 IMUPreInte(pre_integration->delta_q,pre_integration->delta_p);
        //    LOG(INFO)<<"pre_integration:"<<std::endl<<IMUPreInte.rotationMatrix()<<std::endl<<"pre_int_trans:"<<std::endl<<IMUPreInte.translation()<<std::endl;
        //    LOG(INFO)<<"dso:"<<std::endl<<lastF_2_fh.rotationMatrix()<<std::endl<<"dso_trans:"<<std::endl<<lastF_2_fh.translation()<<std::endl;
    }


    SE3 fh_2_slast=lastF_2_fh.inverse()*lastF_2_slast;
    double velVision = fh_2_slast.translation().norm()/(fh->shell->timestamp-slast->timestamp);
//    std::cout<<"velVision:"<<velVision<<" "<<fh->shell->timestamp-slast->timestamp<<std::endl;
    //veloctiy limit
    if(velVision>max_speed)lastF_2_fh= fh_2_slast.inverse() * lastF_2_slast;

    fh->shell->camToTrackingRef = lastF_2_fh.inverse();
    fh->shell->trackingRef = lastF->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

    if(coarseTracker->firstCoarseRMSE < 0)
        coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

    if(setting_logStuff)
    {
        (*coarseTrackingLog) << std::setprecision(16)
                             << fh->shell->id << " "
                             << fh->shell->timestamp << " "
                             << fh->ab_exposure << " "
                             << fh->shell->camToWorld.log().transpose() << " "
                             << aff_g2l.a << " "
                             << aff_g2l.b << " "
                             << achievedRes[0] << " "
                             << tryIterations << "\n";
    }


    Vec5 res;
    res<<achievedRes[0], flowVecs[2], flowVecs[5], flowVecs[4],flowVecs[3];

    return res;
}

Vec5 FullSystem::featureTrackNewCoarse(FrameHessian* fh,bool useInitialPose)
{

    assert(allFrameHistory.size() > 0);
    // set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);

    FrameHessian* lastF = coarseTracker->lastRef;

    AffLight aff_last_2_l = AffLight(0,0);

    int64 start = cv::getTickCount();
    std::vector<cv::Point3f> KeyPoints3d;
    std::vector<cv::KeyPoint> KeyPointsRef;
    int trackingFlag = 0;
    for(cv::KeyPoint kpt:lastF->keyPoints)
    {
        cv::Point3f pt3;
        if(kpt.response>0)
        {
            KeyPointsRef.push_back(kpt);
            pt3.z=1/kpt.response;
            pt3.x=(kpt.pt.x-dso::cxG[featuresLvl])*dso::fxiG[featuresLvl]*pt3.z;
            pt3.y=(kpt.pt.y-dso::cyG[featuresLvl])*dso::fyiG[featuresLvl]*pt3.z;
            KeyPoints3d.push_back(pt3);
//            std::cout<<"kk depth:"<<pt3.z<<std::endl;
        }
    }

    FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
    FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
    SE3 slast_2_sprelast;
    SE3 lastF_2_slast;
    SE3 lastF_2_fh_features;
    if(allFrameHistory.size() == 2){
        lastF_2_fh_features=SE3();
        fh->shell->camToTrackingRef = lastF_2_fh_features.inverse();
        fh->shell->trackingRef = lastF->shell;
        fh->shell->aff_g2l = aff_last_2_l;
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

        Vec5 res;
        res<<0, 0,0,0,0;
        return res;
    }
    else
    {
        {	// lock on global pose consistency!
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
            lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
            aff_last_2_l = slast->aff_g2l;
        }
        SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.
        Eigen::Vector2f matchRelAff;
        lastF_2_fh_features=_imageMatch.trackingWithOF(lastF,fh,wG[featuresLvl],hG[featuresLvl],featuresLvl,KeyPointsRef,KeyPoints3d,
                                                                matchRelAff, fh_2_slast.inverse()*lastF_2_slast,trackingFlag);
        int64 end = cv::getTickCount();
        std::cout << "The differences of lastF_2_fh_features: " << 1000.0*(end - start)/cv::getTickFrequency()<<" ms"<< std::endl;
        std::cout<<"lastF_2_fh_features:"<<lastF_2_fh_features.translation().transpose()<<std::endl
                <<(fh_2_slast.inverse() * lastF_2_slast).translation().transpose()<<std::endl
                <<(fh->shell->viconPose.inverse()*lastF->shell->viconPose).translation().transpose()<<std::endl;

    }

    Vec6 flowVecs = Vec6::Constant(100);
    SE3 lastF_2_fh = lastF_2_fh_features;
    AffLight aff_g2l = aff_last_2_l;

    coarseTracker->newFrame = fh;
    flowVecs = coarseTracker->calcRes(0, lastF_2_fh, aff_g2l, setting_coarseCutoffTH);
    float achievedRes = sqrtf((float)(flowVecs[0] / flowVecs[1]));
    fh->avgIdepth=flowVecs[3];

    SE3 fh_2_slast=lastF_2_fh.inverse()*lastF_2_slast;
    double velVision = fh_2_slast.translation().norm()/(fh->shell->timestamp-slast->timestamp);
//    std::cout<<"velVision:"<<velVision<<" "<<fh->shell->timestamp-slast->timestamp<<std::endl;
    //veloctiy limit
    if(velVision>max_speed)lastF_2_fh= fh_2_slast.inverse() * lastF_2_slast;

    fh->shell->camToTrackingRef = lastF_2_fh.inverse();
    fh->shell->trackingRef = lastF->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

    if(coarseTracker->firstCoarseRMSE < 0)
        coarseTracker->firstCoarseRMSE = achievedRes;

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes);

    Vec5 res;
    res<<achievedRes, flowVecs[2], flowVecs[5], flowVecs[4],flowVecs[3];

    return res;
}

void FullSystem::traceNewCoarse(FrameHessian* fh)
{
    boost::unique_lock<boost::mutex> lock(mapMutex);

    int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

    Mat33f K = Mat33f::Identity();
    K(0,0) = Hcalib.fxl();
    K(1,1) = Hcalib.fyl();
    K(0,2) = Hcalib.cxl();
    K(1,2) = Hcalib.cyl();

    for(FrameHessian* host : frameHessians)		// go through all active frames
    {
        SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
        Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
        Vec3f Kt = K * hostToNew.translation().cast<float>();

        Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();
        //        SE3::Tangent tr=hostToNew.log();
        //        float rotationIndex=tr.tail<3>().norm()/tr.head<3>().norm();
        for(ImmaturePoint* ph : host->immaturePoints)
        {
            //if(rotationIndex>10)ph->traceOnRotation(fh, KRKi, Kt, aff, &Hcalib, false );
            ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );
            if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
            if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
            if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
            if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
            if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
            if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
            trace_total++;
        }
    }
    //	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
    //			trace_total,
    //			trace_good, 100*trace_good/(float)trace_total,
    //			trace_skip, 100*trace_skip/(float)trace_total,
    //			trace_badcondition, 100*trace_badcondition/(float)trace_total,
    //			trace_oob, 100*trace_oob/(float)trace_total,
    //			trace_out, 100*trace_out/(float)trace_total,
    //			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}




void FullSystem::activatePointsMT_Reductor(
        std::vector<PointHessian*>* optimized,
        std::vector<ImmaturePoint*>* toOptimize,
        int min, int max, Vec10* stats, int tid)
{
    ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
    for(int k=min;k<max;k++)
    {
        (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
    }
    delete[] tr;
}



void FullSystem::activatePointsMT()
{

    if(ef->nPoints < setting_desiredPointDensity*0.66)
        currentMinActDist -= 0.8;
    if(ef->nPoints < setting_desiredPointDensity*0.8)
        currentMinActDist -= 0.5;
    else if(ef->nPoints < setting_desiredPointDensity*0.9)
        currentMinActDist -= 0.2;
    else if(ef->nPoints < setting_desiredPointDensity)
        currentMinActDist -= 0.1;

    if(ef->nPoints > setting_desiredPointDensity*1.5)
        currentMinActDist += 0.8;
    if(ef->nPoints > setting_desiredPointDensity*1.3)
        currentMinActDist += 0.5;
    if(ef->nPoints > setting_desiredPointDensity*1.15)
        currentMinActDist += 0.2;
    if(ef->nPoints > setting_desiredPointDensity)
        currentMinActDist += 0.1;

    if(currentMinActDist < 0) currentMinActDist = 0;
    if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
               currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



    FrameHessian* newestHs = frameHessians.back();

    // make dist map.
    coarseDistanceMap->makeK(&Hcalib);
    coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

    //coarseTracker->debugPlotDistMap("distMap");

    std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);


    for(FrameHessian* host : frameHessians)		// go through all active frames
    {
        if(host == newestHs) continue;

        SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
        Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * KiG[0]);
        Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


        for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
        {
            ImmaturePoint* ph = host->immaturePoints[i];
            ph->idxInImmaturePoints = i;

            // delete points that have never been traced successfully, or that are outlier on the last trace.
            if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
            {
                //				immature_invalid_deleted++;
                // remove point.
                delete ph;
                host->immaturePoints[i]=0;
                continue;
            }

            // can activate only if this is true.
            bool canActivate = (ph->lastTraceStatus == IPS_GOOD
                                || ph->lastTraceStatus == IPS_SKIPPED
                                || ph->lastTraceStatus == IPS_BADCONDITION
                                || ph->lastTraceStatus == IPS_OOB
                                )
                    && ph->lastTracePixelInterval < 8
                    && ph->quality > setting_minTraceQuality
                    && ph->idepth_max>0&&ph->idepth_min > 0;


            // if I cannot activate the point, skip it. Maybe also delete it.
            if(!canActivate)
            {
                // if point will be out afterwards, delete it instead.
                if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
                {
                    //					immature_notReady_deleted++;
                    delete ph;
                    host->immaturePoints[i]=0;
                }
                //				immature_notReady_skipped++;
                continue;
            }


            // see if we need to activate point due to distance map.
            Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
//            int u = ptp[0] / ptp[2] + 0.5f;
//            int v = ptp[1] / ptp[2] + 0.5f;
            int u = ptp[0] / ptp[2] + coarseDistanceMap->waddleft/2;
            int v = ptp[1] / ptp[2] + coarseDistanceMap->haddup/2;

            if((u > 0 && v > 0 && u < coarseDistanceMap->w[1] && v < coarseDistanceMap->h[1]))
            {

                float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+(coarseDistanceMap->w[1])*v] + (ptp[0]-floorf((float)(ptp[0])));

                if(dist>=currentMinActDist* ph->my_type)
                {
                    coarseDistanceMap->addIntoDistFinal(u,v);
                    toOptimize.push_back(ph);
                }
            }
            else
            {
                delete ph;
                host->immaturePoints[i]=0;
            }
        }
    }

    //	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
    //			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

    std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

    if(multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

    else
        activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


    for(unsigned k=0;k<toOptimize.size();k++)
    {
        PointHessian* newpoint = optimized[k];
        ImmaturePoint* ph = toOptimize[k];

        if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
        {
            newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
            newpoint->host->pointHessians.push_back(newpoint);
            ef->insertPoint(newpoint);
            for(PointFrameResidual* r : newpoint->residuals)
                ef->insertResidual(r);
            assert(newpoint->efPoint != 0);
            delete ph;
        }
        else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
        {
            delete ph;
            ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
        }
        else
        {
            assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
        }
    }


    for(FrameHessian* host : frameHessians)
    {
        for(int i=0;i<(int)host->immaturePoints.size();i++)
        {
            if(host->immaturePoints[i]==0)
            {
                host->immaturePoints[i] = host->immaturePoints.back();
                host->immaturePoints.pop_back();
                i--;
            }
        }
    }


}






void FullSystem::activatePointsOldFirst()
{
    assert(false);
}

void FullSystem::flagPointsForRemoval()
{
    assert(EFIndicesValid);

    std::vector<FrameHessian*> fhsToKeepPoints;
    std::vector<FrameHessian*> fhsToMargPoints;

    //if(setting_margPointVisWindow>0)
    {
        for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
            if(!frameHessians[i]->flaggedForMarginalization||frameHessians[i]->isRecord) fhsToKeepPoints.push_back(frameHessians[i]);

        for(int i=0; i< (int)frameHessians.size();i++)
            if(frameHessians[i]->flaggedForMarginalization && !(frameHessians[i]->isRecord)) fhsToMargPoints.push_back(frameHessians[i]);
    }


    //ef->setAdjointsF();
    //ef->setDeltaF(&Hcalib);
    int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

    for(FrameHessian* host : frameHessians)		// go through all active frames
    {
        for(unsigned int i=0;i<host->pointHessians.size();i++)
        {
            PointHessian* ph = host->pointHessians[i];
            if(ph==0) continue;
            if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
            {
                host->pointHessiansOut.push_back(ph);
                ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                host->pointHessians[i]=0;
                flag_nores++;
            }
            else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || (host->flaggedForMarginalization&&!(host->isRecord)))
//                else if( (host->flaggedForMarginalization&&!(host->isRecord)))
            {
                flag_oob++;
                if(ph->isInlierNew())
                {
                    flag_in++;
                    int ngoodRes=0;
                    for(PointFrameResidual* r : ph->residuals)
                    {
                        r->resetOOB();
                        r->linearize(&Hcalib);
                        r->efResidual->isLinearized = false;
                        r->applyRes(true);
                        if(r->efResidual->isActive())
                        {
                            r->efResidual->fixLinearizationF(ef);
                            ngoodRes++;
                        }
                    }
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
                    {
                        flag_inin++;
                        ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
                        host->pointHessiansMarginalized.push_back(ph);
                    }
                    else
                    {
                        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                        host->pointHessiansOut.push_back(ph);
                    }


                }
                else
                {
                    host->pointHessiansOut.push_back(ph);
                    ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

                    //printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
                }
                host->pointHessians[i]=0;
            }
        }


        for(int i=0;i<(int)host->pointHessians.size();i++)
        {
            if(host->pointHessians[i]==0)
            {
                host->pointHessians[i] = host->pointHessians.back();
                host->pointHessians.pop_back();
                i--;
            }
        }
    }

}

void FullSystem::restart(FrameHessian* fh,FrameHessian* loopFh)
{
    coarseInitializer->setFirst(&Hcalib, fh);



//    for(int i=0;i<coarseInitializer->KeyPointsRef.size();i++)
//    {
//        cv::KeyPoint kpt=coarseInitializer->KeyPointsRef[i];
//        ImmaturePoint* pt = new ImmaturePoint(kpt.pt.x+0.5f,kpt.pt.y+0.5f,fh,point->my_type, &Hcalib);

//        pt->idepth_max=pt->idepth_min=1;
//        idepthStereo=point->iR*rescaleFactor;
//    if(!std::isfinite(pt->energyTH))
//    {
//        delete pt;
//        continue;

//    }

//    PointHessian* ph = new PointHessian(pt, &Hcalib);
//    delete pt;
//    if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

//    ph->setIdepthScaled(idepthStereo);
//    ph->setIdepthZero(idepthStereo);
//    ph->hasDepthPrior=true;
//    ph->setPointStatus(PointHessian::ACTIVE);

//    firstFrame->pointHessians.push_back(ph);
//    ef->insertPoint(ph);
//    }

}

void FullSystem::addActiveFrame(ImageAndExposure<float>* imagel,ImageAndExposure<float>* imager, int id)
{
    //    if(isLost) return;
    boost::unique_lock<boost::mutex> lock(trackMutex);

    // =========================== add into allFrameHistory =========================
    FrameHessian* fh = new FrameHessian();
    FrameShell* shell = new FrameShell();
    shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
    shell->viconPose =viconNow;
    shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = imagel->timestamp;
    shell->incoming_id = id;
    fh->shell = shell;
    allFrameHistory.push_back(shell);


    // =========================== make Images / derivatives etc. =========================
    fh->ab_exposure = imagel->exposure_time;
    fh->minvall=imagel->minval;
    fh->maxvall=imagel->maxval;
    fh->minvalr=imager->minval;
    fh->maxvalr=imager->maxval;
    fh->makeImages(imagel->image, &Hcalib);
    if(imager!=0)fh->setRightImg(imager->image);
    //generate descriptos
    int64 start=0,end=0;
    start = cv::getTickCount();
//    _imageMatch.calcFeature(fh,wG[featuresLvl],hG[featuresLvl],featuresLvl);
    end = cv::getTickCount();
    LOG(INFO) << "The differences of calcfeature: " << 1000.0*(end - start)/cv::getTickFrequency()<<" ms"<< std::endl;
    int initialTrackFlag=0;
    if(!initialized)
    {
        // use initializer!
        if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
        {
            rotationFirst = rotationNow;
            viconFirst = viconNow;
            fh->shell->viconPose =SE3();
            coarseInitializer->setFirst(&Hcalib, fh);
        }
        else if((initialTrackFlag=coarseInitializer->trackFrame(fh, outputWrapper))==1)	// if SNAPPED
        {

            fh->shell->viconPose =viconFirst.inverse()*viconNow;
            initializeFromInitializer(fh,SE3());
            camNow=fh->PRE_camToWorld;
            writePoseGt(fh);
            lock.unlock();
            deliverTrackedFrame(fh, true);
            writePoseDSO(fh);
        }
        else
        {
            if(initialTrackFlag==-1)coarseInitializer->frameID=-1;
            // if still initializing
            fh->shell->poseValid = false;
            delete fh;
        }
        return;
    }
    else if(relocMode)
    {
        std::cout<<"relocMode:"<<relocMode<<std::endl;
        // use initializer!
        if(relocMode==1)	// first frame set. fh is kept by coarseInitializer.
        {
            coarseInitializer->setFirst(&Hcalib, fh);
            if((coarseInitializer->frameID) == 0)relocMode=2;
        }
        else if(relocMode==2)
        {
            if((initialTrackFlag=coarseInitializer->trackFrame(fh, outputWrapper))==1)
            {
                fh->shell->viconPose =viconNow;
                initializeFromInitializer(fh,coarseInitializer->firstFrame->shell->camToWorld);
                camNow=fh->PRE_camToWorld;
                writePoseGt(fh);
                lock.unlock();
                deliverTrackedFrame(fh, true);
                writePoseDSO(fh);
                relocMode=0;
            }
            else
            {
                if(initialTrackFlag==-1)relocMode=1;
                // if still initializing
                fh->shell->poseValid = false;
                delete fh;
            }
        }

        return;
    }
    else	// do front-end operation.
    {
        // =========================== SWAP tracking reference?. =========================
        if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
        {
            boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
            CoarseTrackerWideAngle* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
        }

        fh->rotationIMU=rotationNow;
        cv::eigen2cv(fh->rotationIMU.matrix3x4(),fh->rotationIMUMat);

        start = cv::getTickCount();
//        Vec5 tres = featureTrackNewCoarse(fh,false);
        Vec5 tres = trackNewCoarse(fh,false);
        end = cv::getTickCount();
        LOG(INFO) << "The differences of track: " << 1000.0*(end - start)/cv::getTickFrequency()<<" ms"<< std::endl;
        std::cout<<"avg idepth:"<<tres[4]<<std::endl;
        if(allFrameHistory.size()>2)
        {
            Vec3 dpos=(allFrameHistory[allFrameHistory.size()-2]->camToWorld.inverse()*fh->shell->camToWorld).translation();
            Vec3 drot=(allFrameHistory[allFrameHistory.size()-2]->camToWorld.inverse()*fh->shell->camToWorld).log().tail<3>();
            SE3 trackpos=fh->shell->camToTrackingRef;
            SE3 trackposGT=coarseTracker->lastRef->shell->viconPose.inverse()*fh->shell->viconPose;
            writeTrackTimeAndPtsNum(1000.0*(end - start)/cv::getTickFrequency(),-1,trackpos.inverse()*trackposGT);
            writeDeltaPosAndAvgidpeth(dpos,drot,tres[4]);
        }
        if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3])||(tres[4]>0&&tres[4]>4))
        {

            std::cout << "image match!!!"<< std::endl;
            std::vector<cv::KeyPoint> KeyPointsRef;
            std::vector<cv::Point3f> KeyPoints3d;
            int trackingFlag = 0;
            Eigen::Vector2f matchRelAff;
            float stereoScale=0;
            cv::Mat idepthMat=cv::Mat::zeros(hG[0],wG[0],CV_32F);

            FrameHessian* loopFh=coarseTracker->lastRef;
            dso::SE3 lastF_2_fh;
            int loop_size=frameHessians.size();

            int scale = (1<<featuresLvl);
            float wshift=coarseTracker_forNewKF->waddleft/scale;
            float hshift=coarseTracker_forNewKF->haddup/scale;
            for(int i=0;i<wG[0];++i)
                for(int j=0;j<hG[0];++j)
                    idepthMat.at<float>(j,i)=coarseTracker_forNewKF->idepth[0][int(i+wshift)+int(j+hshift)*coarseTracker_forNewKF->w[0]];
            dso::SE3 init_pose;
            for(int i=-1;i<loop_size;i++){
                if(i==-1){
                    loopFh=coarseTracker->lastRef;
                    if(_imageMatch.trackingWithOFStereo(fh,wG[featuresLvl],hG[featuresLvl],featuresLvl,KeyPointsRef, KeyPoints3d,stereoScale)){
                         _imageMatch.trackingWithOF(fh,loopFh,wG[featuresLvl],hG[featuresLvl],featuresLvl,KeyPointsRef,KeyPoints3d,matchRelAff,init_pose,trackingFlag);
                    }
                }
                else{
                    loopFh=frameHessians[i];
                    if(_imageMatch.trackingWithOFStereo(fh,wG[featuresLvl],hG[featuresLvl],featuresLvl,KeyPointsRef, KeyPoints3d,stereoScale))
                        _imageMatch.trackingWithOF(fh,loopFh,wG[featuresLvl],hG[featuresLvl],featuresLvl,KeyPointsRef,KeyPoints3d,matchRelAff,init_pose,trackingFlag);
                }

                if(trackingFlag!=0){
                    if(_imageMatch.trackingWithOFStereo(fh,wG[0],hG[0],0,KeyPointsRef, KeyPoints3d,stereoScale,idepthMat,1))
                        lastF_2_fh=_imageMatch.trackingWithOF(fh,loopFh,wG[0],hG[0],0,KeyPointsRef,KeyPoints3d,matchRelAff,init_pose,trackingFlag);
                    if(trackingFlag!=0)break;
                }
            }
            if(trackingFlag!=0){
                fh->shell->camToTrackingRef = lastF_2_fh;
                fh->shell->trackingRef = loopFh->shell;
                fh->shell->aff_g2l = loopFh->shell->aff_g2l;
                fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

                if(tres[4]>2)
                {
                    for(int i=0;i<(int)frameHessians.size();i++)
                    {
                        FrameHessian* fhOld = frameHessians[i];
                        fhOld->flaggedForMarginalization = true;
                    }
                    // =========================== Marginalize Frames =========================
                    for(unsigned int i=0;i<frameHessians.size();i++)
                        if(frameHessians[i]->flaggedForMarginalization)
                        {marginalizeFrame(frameHessians[i]); i--;}
                }

                coarseInitializer->setFirst(&Hcalib, fh);
                if((coarseInitializer->frameID) == 0)relocMode=2;
                else  relocMode=1;
//                updateScales(stereoScale);
//                std::cout<<"KeyPointsRef size:"<<KeyPointsRef.size()<<std::endl;
//                std::cout<<"trackingFlag:"<<lastF_2_fh.inverse().translation()<<std::endl;
//                std::cout<<"trackingFlag_gt:"<<(loopFh->shell->viconPose.inverse()*fh->shell->viconPose).translation()<<std::endl;
//                fh->shell->camToTrackingRef = lastF_2_fh;
//                fh->shell->trackingRef = loopFh->shell;
//                fh->shell->aff_g2l = loopFh->shell->aff_g2l;
//                fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
//                fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
                for(IOWrap::Output3DWrapper* ow : outputWrapper)
                    ow->publishCamPose(fh->shell, &Hcalib);
                recordWrapper->publishCamPose(fh->shell, &Hcalib);
//                deliverTrackedFrame(fh, true);
                camNow=fh->PRE_camToWorld;
                return;
            }
            else
            {
                printf("Initial Tracking failed: LOST!\n");
                allFrameHistory.pop_back();
                isLost=true;
                delete fh;
                return;
            }
        }
        else{
            isLost=false;
        }

        bool needToMakeKF = false;
        if(setting_keyframesPerSecond > 0)
        {
            needToMakeKF = allFrameHistory.size()== 1 ||
                    (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;

            needToRecordKF=allFrameHistory.size()== 1 ||
                    (fh->shell->timestamp - (recordMap.myMap.currentRecKeyFrameVec.back())-> shell->timestamp) > 5.0f/setting_keyframesPerSecond;
        }
        else
        {
            Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                     coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

            needToMakeKF = (allFrameHistory.size()== 1 ||
                    (setting_kfGlobalWeight*sqrtf((double)tres[1])*setting_maxShiftWeightT    +
//                    setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
                    setting_kfGlobalWeight*sqrtf((double)tres[3])*setting_maxShiftWeightRT  +
                    setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 0.5 ||
                    2*coarseTracker->firstCoarseRMSE < tres[0]));

            if(needToMakeKF)needToRecordKFCount++;
            if(needToRecordKFCount>=20&&tres[2]<0.2){
                needToRecordKF = true;
                needToRecordKFCount=0;
            }
            else needToRecordKF = false;

//            SE3 fh2lastRecordKeyFm= (recordMap.myMap.newComeInFrame->shell->camToWorld).inverse()*(fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef );
//            needToRecordKF = allFrameHistory.size()== 1 ||
//                    setting_kfGlobalWeight*
//                    (setting_maxShiftWeightT *  ((KG[0].cast<double>()*fh2lastRecordKeyFm.translation()).norm()) / (wG[0]+hG[0]) +
//                    setting_maxShiftWeightR *  0/ (wG[0]+hG[0]) +
//                    setting_maxShiftWeightRT *((KG[0].cast<double>()*fh2lastRecordKeyFm.rotationMatrix()*fh2lastRecordKeyFm.translation()).norm())/ (wG[0]+hG[0]) +
//                    setting_maxAffineWeight * fabs(logf((float)refToFh[0]))) > 10 ;


        }

        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);
        recordWrapper->publishCamPose(fh->shell, &Hcalib);

        writePoseGt(fh);

        lock.unlock();
        if(needToMakeKF&&needToRecordKF){
            //                _loopClosing.loopclose(fh,recordMap.myMap.currentRecKeyFrameVec,wG[pyrLevelsUsed-1],hG[pyrLevelsUsed-1],pyrLevelsUsed-1);
            //                _loopClosing.loopclose(fh,recordMap.myMap.loadRecKeyFrameVVec,wG[pyrLevelsUsed-1],hG[pyrLevelsUsed-1],pyrLevelsUsed-1);
        }
        deliverTrackedFrame(fh, needToMakeKF);
//        if(needToMakeKF&&needToRecordKF)checkLoopClose(fh);
        camNow=fh->PRE_camToWorld;
        writePoseDSO(fh);
        return;

    }
}

void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{


    if(linearizeOperation)
    {
        if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
        {
            while(true)
            {
                char k=IOWrap::waitKey(0);
                if(k==' ') break;
                handleKey( k );
            }
            lastRefStopID = coarseTracker->refFrameID;
        }
        else handleKey( IOWrap::waitKey(1) );

        if(needKF) makeKeyFrame(fh);
        else makeNonKeyFrame(fh);
    }
    else
    {
        boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
        unmappedTrackedFrames.push_back(fh);
        if(needKF) {
            fh->isKeyframe =true;
            needNewKFAfter=fh->shell->trackingRef->id;
        }
        trackedFrameSignal.notify_all();

        while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
        {
            mappedFrameSignal.wait(lock);
        }

        lock.unlock();
    }
}

void FullSystem::mappingLoop()
{
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

    while(runMapping)
    {
        while(unmappedTrackedFrames.size()==0)
        {
            trackedFrameSignal.wait(lock);
            if(!runMapping) return;
        }

        FrameHessian* fh = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();


        if(fh->isKeyframe)
        {
            lock.unlock();
            isProcessingKF=true;
            makeKeyFrame(fh);
            isProcessingKF=false;
            lock.lock();
        }
        else
        {
            lock.unlock();
            makeNonKeyFrame(fh);
            lock.lock();
        }
        //        // guaranteed to make a KF for the very first two tracked frames.
        //        if(allKeyFramesHistory.size() <= 2)
        //        {
        //            lock.unlock();
        //            makeKeyFrame(fh);
        //            lock.lock();
        //            mappedFrameSignal.notify_all();
        //            continue;
        //        }

        //        if(unmappedTrackedFrames.size() > 3)
        //            needToKetchupMapping=true;


        //        if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
        //        {
        //            lock.unlock();
        //            makeNonKeyFrame(fh);
        //            lock.lock();

        //            if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
        //            {
        //                FrameHessian* fh = unmappedTrackedFrames.front();
        //                unmappedTrackedFrames.pop_front();
        //                {
        //                    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        //                    assert(fh->shell->trackingRef != 0);
        //                    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
        //                    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
        //                }
        //                delete fh;
        //            }

        //        }
        //        else
        //        {
        //            if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
        //            {
        //                lock.unlock();
        //                makeKeyFrame(fh);
        //                needToKetchupMapping=false;
        //                lock.lock();
        //            }
        //            else
        //            {
        //                lock.unlock();
        //                makeNonKeyFrame(fh);
        //                lock.lock();
        //            }
        //        }
        mappedFrameSignal.notify_all();
    }
    printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    runMapping = false;
    trackedFrameSignal.notify_all();
    lock.unlock();

    mappingThread.join();

}

void FullSystem::closeLogFile()
{
    dsoTrackingLog->close() ;
    gtTrackingLog->close() ;
    dsotrackTimeAndPtsNumLog->close();
}

void FullSystem::localOPtimizationBetweenKeyFrames(int localWindowSize)
{
    int _size=0;
    if(localWindowSize!=0){
        _size=localWindowSize-1;
        if(frameHessians.size()<2)return;
        else if(localWindowSize>frameHessians.size())_size=frameHessians.size()-1;
    }
    else{
        if(frameHessians.size()<2)return;
        _size=frameHessians.size()-1;
    }
    FrameHessian *last=frameHessians[frameHessians.size()-_size-1];//because the last frame's idepth is equal to 0, we use the first frame in the window
    SE3 poseLast= last->shell->camToWorld;
    SE3 poseLastInv= poseLast.inverse();
    std::vector<Vec3> Vp3d;
    //add points
    //Generata cv mat
    Eigen::Vector3f* colori = last->dIp[0];
    cv::Mat imgi=cv::Mat(wG[0],hG[0],CV_8U);
    for(int i=0;i<hG[0];++i)
        for(int j=0;j<wG[0];++j){
            imgi.at<uchar>(i,j)=(uchar)(colori[i*wG[0]+j][0]);
        }

    for(PointHessian* point:(last->pointHessiansMarginalized)){
        if(point->idepth!=0){
            for(int i=0;i<_size;++i){
                FrameHessian *frame=frameHessians[frameHessians.size()-i-1];
                //Generata cv mat
                Eigen::Vector3f* colorj = frame->dIp[0];
                cv::Mat imgj=cv::Mat(wG[0],hG[0],CV_8U);
                for(int i=0;i<hG[0];++i)
                    for(int j=0;j<wG[0];++j){
                        imgj.at<uchar>(i,j)=(uchar)(colorj[i*wG[0]+j][0]);
                    }
            }
            Vp3d.push_back(Vec3(point->u,point->v,1/point->idepth));
        }
    }
    if(Vp3d.size()==0){
        printf("the GSLAM optimize point's size is 0!");
        return;
    }
    //add frame to produce M matrix
    for(int i=0;i<_size;++i){
        FrameHessian *frame=frameHessians[frameHessians.size()-i-1];
        SE3 pose= (frame->shell->camToWorld)*poseLastInv;
        localOPtiMethod.addFrame(pose,Vp3d);
    }
    //optimize
    localOPtiMethod.optimize();
    //fix the position and inverse depth
    int pointCount=0;
    for(PointHessian* point:(last->pointHessiansMarginalized)){
        if(point->idepth!=0){
            point->idepth_scaled = point->idepth=localOPtiMethod.vInvDepth[pointCount];
            pointCount++;
        }
    }
    for(int i=0;i<_size;++i){
        FrameHessian *frame=frameHessians[frameHessians.size()-i-2];
        Vec3 pos=poseLast.rotationMatrix()*(localOPtiMethod.vPos[i])+poseLast.translation();
        SE3 newPose(frame->shell->camToWorld.rotationMatrix(),pos);
        //std::cout<<"newPose:"<<newPose.translation()<<std::endl<<"old pose:"<<frame->shell->camToWorld.translation()<<std::endl;
    }

}

bool FullSystem::isFrameViewCrossed(FrameHessian* fh1,FrameHessian* fh2)
{
    float maxDepthInv = dso::ibfG[0];
    float minDepthInv = dso::bfG[0]/3;
    SE3 fhToNew = fh1->PRE_worldToCam * (fh2->PRE_camToWorld);
    Mat33f PRE_RTll_0 = (fhToNew.rotationMatrix()).cast<float>();
    Vec3f PRE_tTll_0 = (fhToNew.translation()).cast<float>();
    float drescale, u, v, new_idepth1,new_idepth2;
    float Ku1, Kv1,Ku2, Kv2;
    Vec3f KliP;
    projectPoint(cxG[0], cyG[0], maxDepthInv, 0, 0,&Hcalib,
            PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku1, Kv1, KliP, new_idepth1);
    projectPoint(cxG[0], cyG[0], minDepthInv, 0, 0,&Hcalib,
            PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku2, Kv2, KliP, new_idepth2);

    bool isinView=false;
    if(new_idepth1>0&&new_idepth2>0){
        if((Ku1>1.1f&&Ku2>1.1f)&&(Ku1<wM3G&&Ku2<wM3G)&&(Kv1>1.1f&&Kv2>1.1f)&&(Kv1<wM3G&&Kv2<wM3G))
            isinView= true;
    }
    std::cout<<"cross:"<<isinView<<" "<<fh1->recordID<<" "<<fh2->recordID<<std::endl;
    std::cout<<"crossdata:"<<Ku1<<" "<<Ku2<<" "<<Kv1<<" "<<Kv2<<" "<<new_idepth1<<" "<<new_idepth2<<std::endl;
    return isinView;
}

FrameHessian* FullSystem::getLocalFrame(int idx)
{
    return frameHessians[idx];
}

void FullSystem::addIMU(const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0,const Eigen::Quaterniond &q_0,double dt)
{
    boost::unique_lock<boost::mutex> imulock(imuIntegrationMutex);
    if(needNewIMUPreInte)
    {
        Eigen::Vector3d Ba(0,0,0);
        Eigen::Vector3d Bg(0,0,0);
        pre_integration->reset(acc_0, gyr_0, q_0,Ba, Bg);
        needNewIMUPreInte = false;
    }
    else
    {
        if(pre_integration!=nullptr)pre_integration->push_back(dt,acc_0,gyr_0);
        //        LOG(INFO)<<"dt:"<<dt<<"acc_0:"<<std::endl<<acc_0<<"gyr_0:"<<std::endl<<gyr_0;
    }
}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
    // needs to be set by mapping thread. no lock required since we are in mapping thread.
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        assert(fh->shell->trackingRef != 0);
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
        fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
    }
    traceNewCoarse(fh);
    delete fh;
}

void FullSystem::makeKeyFrame( FrameHessian* fh)
{
    rotationLast = rotationNow;
    // needs to be set by mapping thread
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        assert(fh->shell->trackingRef != 0);
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
        fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
    }
    int64 start1=0,end1=0;
    start1 = cv::getTickCount();


    traceNewCoarse(fh);
    end1 = cv::getTickCount();
    std::cout << "The differences of traceNewCoarse: " << 1000.0*(end1 - start1)/cv::getTickFrequency()<<" ms"<< std::endl;
    LOG(INFO) << "The differences of traceNewCoarse: " << 1000.0*(end1 - start1)/cv::getTickFrequency()<<" ms"<< std::endl;
    boost::unique_lock<boost::mutex> lock(mapMutex);

    // =========================== Flag Frames to be Marginalized. =========================
    flagFramesForMarginalization(fh);

    // =========================== add New Frame to Hessian Struct. =========================
    fh->idx = frameHessians.size();
    frameHessians.push_back(fh);
    fh->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(fh->shell);
    ef->insertFrame(fh, &Hcalib);

    setPrecalcValues();

    // =========================== add new residuals for old points =========================
    int numFwdResAdde=0;
    for(FrameHessian* fh1 : frameHessians)		// go through all active frames
    {
        if(fh1 == fh) continue;
        for(PointHessian* ph : fh1->pointHessians)
        {
            if(!(ph->set_stereo_res)){
                PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh1);
                r->setState(ResState::IN);
                ph->residuals.push_back(r);
                ef->insertResidual(r);
                ph->lastResiduals[1] = ph->lastResiduals[0];
                ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
                numFwdResAdde+=1;
                ph->set_stereo_res=true;
            }

            PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
            r->setState(ResState::IN);
            ph->residuals.push_back(r);
            ef->insertResidual(r);
            ph->lastResiduals[1] = ph->lastResiduals[0];
            ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
            numFwdResAdde+=1;
        }
    }
    writeTrackTimeAndPtsNum(-1,numFwdResAdde,SE3());
    // =========================== Activate Points (& flag for marginalization). =========================
    activatePointsMT();
    ef->makeIDX();

    start1 = cv::getTickCount();
    // =========================== OPTIMIZE ALL =========================

    //           frameCount++;
    //        if(frameCount==5){
    //            localOPtimizationBetweenKeyFrames(frameCount);
    //            frameCount=0;
    //        }
    //        localOPtimizationBetweenKeyFrames(0);
    fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;

    float rmse;
#if USE_CERES
    rmse=1;
    std::cout<<"frameHessians:"<<frameHessians.size()<<std::endl;
    ceresOpt->optimization(frameHessians);

#else
//    if(allKeyFramesHistory.size() <= 4)rmse = optimize(setting_maxOptIterations);
//    else rmse = OneStepOptimize(0);
        rmse=optimize(setting_maxOptIterations);
    printf("rmse:%f!\n",rmse);
#endif
    end1 = cv::getTickCount();
    std::cout << "The differences of optimize: " << 1000.0*(end1 - start1)/cv::getTickFrequency()<<" ms"<< std::endl;
    LOG(INFO) << "The differences of optimize: " << 1000.0*(end1 - start1)/cv::getTickFrequency()<<" ms"<< std::endl<<std::endl;

    start1 = cv::getTickCount();


    // =========================== Figure Out if INITIALIZATION FAILED =========================
    if(allKeyFramesHistory.size() <= 4)
    {
        if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
        {
            printf("INITIALIZATINO rmse:2;%f and %f! Resetting.\n",rmse,20*benchmark_initializerSlackFactor);
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed=true;
        }
        if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
        {
            printf("INITIALIZATINO rmse:3;%f and %f! Resetting.\n",rmse,13*benchmark_initializerSlackFactor);
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed=true;
        }
        if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
        {
            printf("INITIALIZATINO rmse:4;%f and %f! Resetting.\n",rmse,9*benchmark_initializerSlackFactor);
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed=true;
        }
    }


    if(initFailed) return;
    if(isLost) return;

    // =========================== REMOVE OUTLIER =========================
    removeOutliers();



    {
        boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        coarseTracker_forNewKF->makeK(&Hcalib);
        coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians,recordMap.myMap.currentRecKeyFrameVec,&Hcalib);



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
    }


    debugPlot("post Optimize");

    end1 = cv::getTickCount();
    std::cout << "The differences of removeOutliers: " << 1000.0*(end1 - start1)/cv::getTickFrequency()<<" ms"<< std::endl;
    LOG(INFO) << "The differences of removeOutliers: " << 1000.0*(end1 - start1)/cv::getTickFrequency()<<" ms"<< std::endl;



    // =========================== (Activate-)Marginalize Points =========================
    flagPointsForRemoval();
    ef->dropPointsF();
    getNullspaces(
                ef->lastNullspaces_pose,
                ef->lastNullspaces_scale,
                ef->lastNullspaces_affA,
                ef->lastNullspaces_affB);
    ef->marginalizePointsF();


    // =========================== add new Immature points & new residuals =========================
    start1 = cv::getTickCount();
    makeNewTraces(fh, 0);
    end1 = cv::getTickCount();
    LOG(INFO) << "The differences of getdepthFromStereo: " << 1000.0*(end1 - start1)/cv::getTickFrequency()<<" ms"<< std::endl;
    int scale = (1<<featuresLvl);
    float wshift=coarseTracker_forNewKF->waddleft/scale;
    float hshift=coarseTracker_forNewKF->haddup/scale;
    for(int i=0;i<wG[featuresLvl];++i)
        for(int j=0;j<hG[featuresLvl];++j)
            fh->depthMatrix.at<float>(j,i)=coarseTracker_forNewKF->idepth[featuresLvl][int(i+wshift)+int(j+hshift)*coarseTracker_forNewKF->w[featuresLvl]];
    for(int i=0;i<fh->keyPoints.size();++i){
        cv::KeyPoint kp= fh->keyPoints[i];
        float idpt=coarseTracker_forNewKF->idepth[featuresLvl][int(kp.pt.x+wshift)+(int)(kp.pt.y+hshift)*coarseTracker_forNewKF->w[featuresLvl]];
        if(idpt!=0){
            fh->keyPoints[i].response=idpt;
        }
        else{
            fh->keyPoints[i].response=-1;
        }

    }
    //    std::cout<<"count:"<< frameHessians[frameSize]->keyPoints.size()<<" "<<cc<<" "<<std::endl;
    //    std::cout<<"*******************************************************"<<std::endl;



    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }
    recordWrapper->publishRecordKeyframes(recordMap.myMap.loadRecKeyFrameVVec,fh, false, &Hcalib);

    // =========================== Marginalize Frames =========================
    for(unsigned int i=0;i<frameHessians.size();i++)
        if(frameHessians[i]->flaggedForMarginalization)
        {marginalizeFrame(frameHessians[i]); i--;}
    //        if(idpt!=0){

    printLogLine();
    if(needToRecordKF&&_loopClosing.loopDetectedType==-1)
    {
        recordMap.addKeyFrame(fh);
        LOG(INFO)<<"new key frame:"<<recordMap.myMap.currentRecKeyFrameVec.size()<<std::endl;
        int recordmapSize = recordMap.myMap.currentRecKeyFrameVec.size();
        if( recordmapSize>= loop_delay+1){
            int nKf = recordmapSize - 1 - loop_delay;
            dso::FrameHessian * addKf = recordMap.myMap.currentRecKeyFrameVec[nKf];
            _loopClosing.addFrameInDBoWDb(addKf,false);
        }
    }

    needNewIMUPreInte = true;
    LOG(INFO)<<"needNewIMUPreInte:"<<needNewIMUPreInte;

}

void  FullSystem::checkLoopClose(FrameHessian* fh)
{
    FrameHessian* loopFh = new FrameHessian();
    FrameShell* shell = new FrameShell();
    shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = fh->shell->timestamp;
    shell->incoming_id = fh->shell->id;
    loopFh->shell = shell;

//    std::cout<<"_loopClosing.loopDetectedType:"<<_loopClosing.loopDetectedType<<std::endl;
    //loopclose detect twice
    if(_loopClosing.loopDetectedType == 2){

    }
    else if(_loopClosing.loopDetectedType == 0)
    {
        LOG(INFO)<<"Loop Detected! Type is "<<_loopClosing.loopDetectedType;
        int LoopID=_loopClosing.relocLoopID;
        FrameHessian* loopFhTmp = recordMap.myMap.currentRecKeyFrameVec[LoopID];
        // needs to be set by mapping thread. no lock required since we are in mapping thread.
        loopFh->copyFrame(loopFhTmp);
        Vec5 tres = trackNewCoarse(loopFh,true);
        if((tres[1]==0&&tres[2]==0&&tres[3]==0)||!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            _loopClosing.loopDetectedType = -1;
        }
        else
        {
            _loopClosing.loopOptimize(loopFh,loopFhTmp,recordMap.myMap.currentRecKeyFrameVec,frameHessians,_loopClosing.loopRelativePose);

            recordMap.mergeKeyFrameLoopInCurRecFrames(LoopID);
            //need to optimize scale here
            updateScales(frameHessians,recordMap.myMap.currentRecKeyFrameVec[LoopID],_loopClosing.idepthPairVec);
        }

    }
    else if(_loopClosing.loopDetectedType == 1)
    {
        LOG(INFO)<<"Loop Detected! Type is "<<_loopClosing.loopDetectedType;
        int N = recordMap.myMap.loadRecKeyFrameVVec.size();
        int loopGroupID=0,loopSubID=_loopClosing.relocLoopID;
        for(int i=0;i<N;++i)
        {
            if(loopSubID>=recordMap.myMap.loadRecKeyFrameVVec[i].size()) loopSubID=loopSubID-recordMap.myMap.loadRecKeyFrameVVec[i].size();
            else{
                loopGroupID=i;
                break;
            }
        }
        dso::FrameHessian *loopFhTmp =recordMap.myMap.loadRecKeyFrameVVec[loopGroupID][loopSubID];
        loopFh->copyFrame(loopFhTmp);
        Vec5 tres = trackNewCoarse(loopFh,true);
        if((tres[1]==0&&tres[2]==0&&tres[3]==0)||!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            _loopClosing.loopDetectedType = -1;

        }
        else{
            _loopClosing.loopOptimizeWithLoadMap(loopFh,loopFhTmp,recordMap.myMap.currentRecKeyFrameVec,frameHessians,_loopClosing.loopRelativePose);
            //need to optimize scale here
            updateScales(frameHessians,loopFhTmp,_loopClosing.idepthPairVec);
            recordMap.mergeKeyFrameLoopInLoadRecFrames(loopGroupID,loopSubID);
        }
    }
    delete loopFh;
}

void FullSystem::updateScales(std::vector<FrameHessian*> currentFhs,FrameHessian* loopFh,std::vector<std::vector<float>> idepthPairVec)
{
    float scale = 1.0;
    std::vector<float> scaleVec;
    FrameHessian* fhCur = currentFhs.back();
    SE3 fhToNew = loopFh->PRE_worldToCam * (fhCur->PRE_camToWorld);
    Mat33f PRE_RTll_0 = (fhToNew.rotationMatrix()).cast<float>();
    Vec3f PRE_tTll_0 = (fhToNew.translation()).cast<float>();

    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3f KliP;

    for(int i=0;i<idepthPairVec.size();++i)
    {
        u =idepthPairVec[i][0];
        v =idepthPairVec[i][1];
        if(projectPoint(u, v, idepthPairVec[i][3], 0, 0,&Hcalib,PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
            scaleVec.push_back(new_idepth/idepthPairVec[i][2]);
        else scaleVec.push_back(idepthPairVec[i][3]/idepthPairVec[i][2]);
    }

    std::random_device rd;
    int sizeScales = scaleVec.size();
    int iter=sizeScales/3;
    float sigma=0.2,best=0;
    for (int i=0;i<iter;++i){
        int idx1=rd()%sizeScales;
        int idx2=rd()%sizeScales;
        int idx3=rd()%sizeScales;
        int idx4=rd()%sizeScales;
        //        std::cout<<"random:"<<idx1<<" "<<idx2<<std::endl;
        int count=0,bestCount=0;
        float avr = (scaleVec[idx1]+scaleVec[idx2]+scaleVec[idx3]+scaleVec[idx4])/2;
        float sum=0,limit=avr*sigma;
        for(int j=0;j<sizeScales;++j){
            if(fabs(scaleVec[j]-avr)<limit){
                sum+=scaleVec[j];
                count++;
            }
        }
        if(bestCount<count){
            bestCount=count;
            best=sum/count;
            if(bestCount*1.0/sizeScales>0.8)
                break;
        }
    }
    scale=best;
    std::cout<<"scale:"<<scale<<std::endl;


}

void FullSystem::updateScales(float scale)
{
    if(std::abs(scale)>0.1f)
    {
        for(FrameHessian* fh:frameHessians)
        {
            for(PointHessian* pt:fh->pointHessians)pt->setIdepth(pt->idepth*scale);
            for(PointHessian* pt:fh->pointHessiansMarginalized)pt->setIdepth(pt->idepth*scale);
            for(ImmaturePoint* impt:fh->immaturePoints)
            {
                impt->idepth_min=impt->idepth_min*scale;
                impt->idepth_max=impt->idepth_max*scale;
            }

        }

    }
}

void FullSystem::initializeFromInitializer(FrameHessian* newFrame,SE3 initialPose)
{
    boost::unique_lock<boost::mutex> lock(mapMutex);

    // add firstframe.
    FrameHessian* firstFrame = coarseInitializer->firstFrame;
    firstFrame->idx = frameHessians.size();
    frameHessians.push_back(firstFrame);
    recordMap.addKeyFrame(firstFrame);
    firstFrame->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(firstFrame->shell);
    ef->insertFrame(firstFrame, &Hcalib);
    setPrecalcValues();

    //int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    //int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

    firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);


    //    float sumID=1e-5, numID=1e-5;

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    float rescaleFactor = 1;

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
               (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

    float idepthStereo;
    for(int i=0;i<coarseInitializer->numPoints[0];i++)
    {
        if(rand()/(float)RAND_MAX > keepPercentage) continue;

        Pnt* point = coarseInitializer->points[0]+i;
        ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

        pt->idepth_max=pt->idepth_min=1;
        idepthStereo=point->iR*rescaleFactor;


        if(!std::isfinite(pt->energyTH))
        {
            delete pt;
            continue;

        }

        PointHessian* ph = new PointHessian(pt, &Hcalib);
        delete pt;
        if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

        ph->setIdepthScaled(idepthStereo);
        ph->setIdepthZero(idepthStereo);
        ph->hasDepthPrior=true;
        ph->setPointStatus(PointHessian::ACTIVE);

        firstFrame->pointHessians.push_back(ph);
        ef->insertPoint(ph);
    }
//    _imageMatch.calcCornersInSelectPts(firstFrame,firstFrame->pointHessians,0);

    SE3 firstToNew = coarseInitializer->thisToNext;
    firstToNew.translation() /= rescaleFactor;


    // really no lock required, as we are initializing.
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        firstFrame->shell->camToWorld = initialPose;
        firstFrame->shell->aff_g2l = AffLight(0,0);
        firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
        firstFrame->shell->trackingRef=0;
        firstFrame->shell->camToTrackingRef = SE3();

        newFrame->shell->trackingRef = firstFrame->shell;
        newFrame->shell->camToTrackingRef = firstToNew.inverse();
        newFrame->shell->camToWorld = newFrame->shell->trackingRef->camToWorld*newFrame->shell->camToTrackingRef;
        newFrame->shell->aff_g2l = AffLight(0,0);
        newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);

    }

    initialized=true;
    printf("INITIALIZE FROM INITIALIZER (%d pts),rescaleFactor(%f)!\n", (int)firstFrame->pointHessians.size(),rescaleFactor);
}

void FullSystem::makeNewTracesForTestStereo(FrameHessian* newFrame, float* gtDepth)
{
    pixelSelector->allowFast = true;
    //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

    newFrame->pointHessians.reserve(numPointsTotal*1.2f);
    //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

    for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
        for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
        {
            int i = x+y*wG[0];
            if(selectionMap[i]==0) continue;

            ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
            if(!std::isfinite(impt->energyTH)) delete impt;
            else newFrame->immaturePoints.push_back(impt);

        }

    //printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
    pixelSelector->allowFast = true;
    //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

    newFrame->pointHessians.reserve(numPointsTotal*1.2f);
    //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

    int lvl = pyrLevelsUsed-1;
    int scale = (1<<lvl);
    cv::Mat idepthMat(hG[lvl],wG[lvl],CV_32FC1);
    float wshift=coarseTracker_forNewKF->waddleft/scale;
    float hshift=coarseTracker_forNewKF->haddup/scale;
    for(int y=0;y<hG[lvl];y++)
        for(int x=0;x<wG[lvl];x++)
        {
            idepthMat.at<float>(y,x)=coarseTracker_forNewKF->idepth[lvl][int(x+wshift)+int(y+hshift)*wG[lvl]];
        }
    cv::imshow("depthMat",idepthMat);
    cv::waitKey(1);
    std::vector<cv::KeyPoint> KeyPointsRef;
    std::vector<cv::Point3f> KeyPoints3d;
    float idepthScale=0;
    if(newFrame->dIr[0]!=0)_imageMatch.trackingWithOFStereo(newFrame,wG[lvl],hG[lvl],lvl,KeyPointsRef, KeyPoints3d,idepthScale,idepthMat,1);

    int c1=0,c2=0;
    std::cout<<"idepthScale:"<<" "<<idepthScale<<std::endl;
    std::cout<<"newFrame affine:"<<" "<<newFrame->leftToright_affine[0]<<" "<<newFrame->leftToright_affine[1]<<std::endl;
    if(idepthScale>0&&fabs(1-idepthScale)>0.1){
//        scale_XI_stereo= 1/idepthScale;
//        scale_idepth_stereo= idepthScale;
    }
    for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
        for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
        {
            int i = x+y*wG[0];
            if(selectionMap[i]==0) continue;

            ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
//            float idpethPre = coarseTracker_forNewKF->idepth[lvl][int(x/scale+wshift)+int(y/scale+hshift)*wG[lvl]];
//            if(idpethPre > dso::ibfG[0]*2 && idpethPre<5){
//                impt->idepth_min = SCALE_IDEPTH*idpethPre*0.75f;
//                impt->idepth_max = SCALE_IDEPTH*idpethPre*1.25f;c1++;
//            }
//            else
//                if(newFrame->dIr[0]!=0){
//                impt->traceOnRight(newFrame, &Hcalib, false );c2++;
//            }
            if(!std::isfinite(impt->energyTH)) delete impt;
            else newFrame->immaturePoints.push_back(impt);

        }
//    _imageMatch.calcCornersInSelectPts(newFrame,newFrame->immaturePoints,0);
    std::cout<<"makeNewTraces:"<<c1<<" "<<c2<<std::endl;

    //printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}

void FullSystem::setPrecalcValues()
{
    for(FrameHessian* fh : frameHessians)
    {
        fh->targetPrecalc.resize(frameHessians.size());
        for(unsigned int i=0;i<frameHessians.size();i++)
            fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }

    ef->setDeltaF(&Hcalib);
}

void FullSystem::printLogLine()
{
    if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
               allKeyFramesHistory.back()->id,
               statistics_lastFineTrackRMSE,
               ef->resInA,
               ef->resInL,
               ef->resInM,
               (int)statistics_numForceDroppedResFwd,
               (int)statistics_numForceDroppedResBwd,
               allKeyFramesHistory.back()->aff_g2l.a,
               allKeyFramesHistory.back()->aff_g2l.b,
               frameHessians.back()->shell->id - frameHessians.front()->shell->id,
               (int)frameHessians.size());


    if(!setting_logStuff) return;

    if(numsLog != 0)
    {
        (*numsLog) << allKeyFramesHistory.back()->id << " "  <<
                      statistics_lastFineTrackRMSE << " "  <<
                      (int)statistics_numCreatedPoints << " "  <<
                      (int)statistics_numActivatedPoints << " "  <<
                      (int)statistics_numDroppedPoints << " "  <<
                      (int)statistics_lastNumOptIts << " "  <<
                      ef->resInA << " "  <<
                      ef->resInL << " "  <<
                      ef->resInM << " "  <<
                      statistics_numMargResFwd << " "  <<
                      statistics_numMargResBwd << " "  <<
                      statistics_numForceDroppedResFwd << " "  <<
                      statistics_numForceDroppedResBwd << " "  <<
                      frameHessians.back()->aff_g2l().a << " "  <<
                      frameHessians.back()->aff_g2l().b << " "  <<
                      frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
                      (int)frameHessians.size() << " "  << "\n";
        numsLog->flush();
    }


}

void FullSystem::printEigenValLine()
{
    if(!setting_logStuff) return;
    if(ef->lastHS.rows() < 12) return;


    MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
    MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
    int n = Hp.cols()/8;
    assert(Hp.cols()%8==0);

    // sub-select
    for(int i=0;i<n;i++)
    {
        MatXX tmp6 = Hp.block(i*8,0,6,n*8);
        Hp.block(i*6,0,6,n*8) = tmp6;

        MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
        Ha.block(i*2,0,2,n*8) = tmp2;
    }
    for(int i=0;i<n;i++)
    {
        MatXX tmp6 = Hp.block(0,i*8,n*8,6);
        Hp.block(0,i*6,n*8,6) = tmp6;

        MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
        Ha.block(0,i*2,n*8,2) = tmp2;
    }

    VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
    VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
    VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
    VecX diagonal = ef->lastHS.diagonal();

    std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
    std::sort(eigenP.data(), eigenP.data()+eigenP.size());
    std::sort(eigenA.data(), eigenA.data()+eigenA.size());

    int nz = std::max(100,setting_maxFrames*10);

    if(eigenAllLog != 0)
    {
        VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
        (*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
        eigenAllLog->flush();
    }
    if(eigenALog != 0)
    {
        VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
        (*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
        eigenALog->flush();
    }
    if(eigenPLog != 0)
    {
        VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
        (*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
        eigenPLog->flush();
    }

    if(DiagonalLog != 0)
    {
        VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
        (*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
        DiagonalLog->flush();
    }

    if(variancesLog != 0)
    {
        VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
        (*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
        variancesLog->flush();
    }

    std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
    (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
    for(unsigned int i=0;i<nsp.size();i++)
        (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
    (*nullspacesLog) << "\n";
    nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
    if(!setting_logStuff) return;


    boost::unique_lock<boost::mutex> lock(trackMutex);

    std::ofstream* lg = new std::ofstream();
    lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
    lg->precision(15);

    for(FrameShell* s : allFrameHistory)
    {
        (*lg) << s->id
              << " " << s->marginalizedAt
              << " " << s->statistics_goodResOnThis
              << " " << s->statistics_outlierResOnThis
              << " " << s->movedByOpt;



        (*lg) << "\n";
    }





    lg->close();
    delete lg;

}

void FullSystem::saveMap()
{
    recordMap.SaveMap();
}

void FullSystem::loadMap()
{
    recordMap.setDir(loadMapPath);
    recordMap.LoadMap();
    recordMap.setDir(saveMapPath);
    preProcessAfterLoad();
}

void FullSystem::preProcessAfterLoad()
{
    int N=recordMap.myMap.loadRecKeyFrameVVec.size();
    LOG(INFO)<<"frame vec size:"<<N;
    std::cout<<"frame vec size"<<N<<std::endl;

    for(int i=0;i<N;++i){
        for(FrameHessian* frame:recordMap.myMap.loadRecKeyFrameVVec[i])
        {
            frame->frameID = frame->recordID;

            frame->makeImages(frame->imgdata,&Hcalib);
            Eigen::Matrix3d rotation_matrix;
            Eigen::Vector3d translation;

            for(int i=0;i<3;++i)
            {
                translation(i) = frame->rotationIMUMat.at<double>(i,3);
                for(int j=0;j<3;++j)
                    rotation_matrix(i,j)=frame->rotationIMUMat.at<double>(i,j);
            }
            dso::SE3 _se3imu( rotation_matrix,translation);

            frame->rotationIMU = _se3imu;
            for(int i=0;i<3;++i)
            {
                translation(i) = frame->worldToCamMat.at<double>(i,3);
                for(int j=0;j<3;++j)
                    rotation_matrix(i,j)=frame->worldToCamMat.at<double>(i,j);
            }
            dso::SE3 _se3cam( rotation_matrix,translation);

            frame->PRE_worldToCam = _se3cam;
            frame->PRE_camToWorld=frame->PRE_worldToCam.inverse();
            frame->shell = new FrameShell();
            frame->shell->aff_g2l = AffLight(0,0);
            frame->shell->marginalizedAt = frame->shell->id = frame->recordID;
            frame->shell->camToWorld = frame->PRE_camToWorld;

            _loopClosing.addFrameInDBoWDb(frame,true);
        }
    }
    recordWrapper->publishRecordKeyframes(recordMap.myMap.loadRecKeyFrameVVec,recordMap.myMap.loadRecKeyFrameVVec[N-1].back(), false, &Hcalib);

    reLocalizationMode=true;
}

void FullSystem::reLocalization(FrameHessian *fh, FrameShell* shell)
{

}

void FullSystem::printEvalLine()
{
    return;
}





}
