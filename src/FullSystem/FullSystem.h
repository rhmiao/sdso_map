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


#pragma once
#define MAX_ACTIVE_FRAMES 100

#include <deque>
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "vector"
 
#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/CeresOpt.h"
#include "FullSystem/PixelSelector2.h"
#include "ExtraImgProcess/imageMatch.h"
#include "OptimizationBackend/Rank1FactorizationOpt.h"
#include "ExtraImgProcess/loopClosing.h"
#include "ExtraImgProcess/map.h"
#include "util/integration_base.h"

#include <math.h>

namespace dso
{
namespace IOWrap
{
class Output3DWrapper;
}

class PixelSelector;
class PCSyntheticPoint;
class CoarseTracker;
struct FrameHessian;
struct PointHessian;
class CoarseInitializer;
struct ImmaturePointTemporaryResidual;
//class ImageAndExposure<Vec3f>;
//class ImageAndExposure<float>;
class CoarseDistanceMap;
class CoarseDistanceMapWideAngle;
class CoarseTrackerWideAngle;

class EnergyFunctional;

template<typename T> inline void deleteOut(std::vector<T*> &v, const int i)
{
	delete v[i];
	v[i] = v.back();
	v.pop_back();
}
template<typename T> inline void deleteOutPt(std::vector<T*> &v, const T* i)
{
	delete i;

	for(unsigned int k=0;k<v.size();k++)
		if(v[k] == i)
		{
			v[k] = v.back();
			v.pop_back();
		}
}
template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const int i)
{
	delete v[i];
	for(unsigned int k=i+1; k<v.size();k++)
		v[k-1] = v[k];
	v.pop_back();
}
template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const T* element,bool isRecord=false)
{
	int i=-1;
	for(unsigned int k=0; k<v.size();k++)
	{
		if(v[k] == element)
		{
			i=k;
			break;
		}
	}
	assert(i!=-1);

	for(unsigned int k=i+1; k<v.size();k++)
		v[k-1] = v[k];
	v.pop_back();

   if(!isRecord) delete element;
}


inline bool eigenTestNan(const MatXX &m, std::string msg)
{
	bool foundNan = false;
	for(int y=0;y<m.rows();y++)
		for(int x=0;x<m.cols();x++)
		{
			if(!std::isfinite((double)m(y,x))) foundNan = true;
		}

	if(foundNan)
	{
		printf("NAN in %s:\n",msg.c_str());
		std::cout << m << "\n\n";
	}


	return foundNan;
}





class FullSystem {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	FullSystem();
	virtual ~FullSystem();

     void stereoMatch( ImageAndExposure<float>* imagel,ImageAndExposure<float>* imager, int id,cv::Mat &idepthMap,std::vector<cv::Point3f> &KeyPoints3d);
     void restart(FrameHessian* fh,FrameHessian* loopFh);
     // adds a new frame, and creates point & residual structs.
     void addActiveFrame(ImageAndExposure<float>* imagel,ImageAndExposure<float>* imager, int id);


	// marginalizes a frame. drops / marginalizes points & residuals.
	void marginalizeFrame(FrameHessian* frame);

	void blockUntilMappingIsFinished();

    void closeLogFile();

	float optimize(int mnumOptIts);

    float OneStepOptimize(int mnumOptIts);

	void printResult(std::string file);

    void writePoseGt(FrameHessian* fh);

    void writePoseDSO(FrameHessian* fh);

    void writeTrackTimeAndPtsNum(double dt,int numPts,SE3 dpos);

    void writeDeltaPosAndAvgidpeth(Vec3 dpos,Vec3 drot,double avgidepth);

	void debugPlot(std::string name);

	void printFrameLifetimes();
	// contains pointers to active frames
    void saveMap();

    void loadMap();

    void preProcessAfterLoad();

    void reLocalization(FrameHessian* fh,FrameShell* shell);

    bool isFrameViewCrossed(FrameHessian* fh1,FrameHessian* fh2);

    FrameHessian* getLocalFrame(int idx);

    void addIMU(const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0,const Eigen::Quaterniond &q_0,double dt);

    std::vector<IOWrap::Output3DWrapper*> outputWrapper;

    IOWrap::Output3DWrapper* recordWrapper;

	bool isLost;
	bool initFailed;
	bool initialized;
    int relocMode;
	bool linearizeOperation;
    bool isLoadMap;
    int loop_delay;
    float viewAngle;
    bool isProcessingKF;
    dso::SE3 rotationNow,rotationFirst;
    dso::SE3 viconNow,viconFirst;
    dso::SE3 rotationLast;

    DSOCeresOpt *ceresOpt;

    bool needNewIMUPreInte;
    IntegrationBase *pre_integration;

    dso::SE3 camNow;
    double timeNow;
    double timeLast;


	void setGammaFunction(float* BInv);
	void setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH);

    CalibHessian Hcalib;



private:
   //save map
    bool reLocalizationMode;

	// opt single point
	int optimizePoint(PointHessian* point, int minObs, bool flagOOB);
	PointHessian* optimizeImmaturePoint(ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);

	double linAllPointSinle(PointHessian* point, float outlierTHSlack, bool plot);

	// mainPipelineFunctions
    Vec5 trackNewCoarse(FrameHessian* fh,bool useInitialPose);
    Vec5 featureTrackNewCoarse(FrameHessian* fh,bool useInitialPose);
	void traceNewCoarse(FrameHessian* fh);
	void activatePoints();
	void activatePointsMT();
	void activatePointsOldFirst();
	void flagPointsForRemoval();
    void makeNewTracesForTestStereo(FrameHessian* newFrame, float* gtDepth);
	void makeNewTraces(FrameHessian* newFrame, float* gtDepth);
    void initializeFromInitializer(FrameHessian* newFrame,SE3 initialPose);
	void flagFramesForMarginalization(FrameHessian* newFH);


	void removeOutliers();


	// set precalc values.
	void setPrecalcValues();


	// solce. eventually migrate to ef.
    void oneStepSolveSystem();
	void solveSystem(int iteration, double lambda);
	Vec3 linearizeAll(bool fixLinearization);
	bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
	void backupState(bool backupLastStep);
	void loadSateBackup();
	double calcLEnergy();
	double calcMEnergy();
	void linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
	void activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,std::vector<ImmaturePoint*>* toOptimize,int min, int max, Vec10* stats, int tid);
	void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);

	void printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);

	void debugPlotTracking();

	std::vector<VecX> getNullspaces(
			std::vector<VecX> &nullspaces_pose,
			std::vector<VecX> &nullspaces_scale,
			std::vector<VecX> &nullspaces_affA,
			std::vector<VecX> &nullspaces_affB);

	void setNewFrameEnergyTH();


	void printLogLine();
	void printEvalLine();
	void printEigenValLine();
	std::ofstream* calibLog;
	std::ofstream* numsLog;
	std::ofstream* errorsLog;
	std::ofstream* eigenAllLog;
	std::ofstream* eigenPLog;
	std::ofstream* eigenALog;
	std::ofstream* DiagonalLog;
	std::ofstream* variancesLog;
	std::ofstream* nullspacesLog;

	std::ofstream* coarseTrackingLog;

    std::ofstream* dsoTrackingLog;
    std::ofstream* gtTrackingLog;
    std::ofstream* dsotrackTimeAndPtsNumLog;
    std::ofstream* dsotrackDposAndAvgidpthLog;

	// statistics
	long int statistics_lastNumOptIts;
	long int statistics_numDroppedPoints;
	long int statistics_numActivatedPoints;
	long int statistics_numCreatedPoints;
	long int statistics_numForceDroppedResBwd;
	long int statistics_numForceDroppedResFwd;
	long int statistics_numMargResFwd;
	long int statistics_numMargResBwd;
	float statistics_lastFineTrackRMSE;

    std::vector<FrameHessian*> frameHessians;	// ONLY changed in marginalizeFrame and addFrame.

	// =================== changed by tracker-thread. protected by trackMutex ============
	boost::mutex trackMutex;
    boost::mutex imuIntegrationMutex;
	std::vector<FrameShell*> allFrameHistory;
	CoarseInitializer* coarseInitializer;
	Vec5 lastCoarseRMSE;


	// ================== changed by mapper-thread. protected by mapMutex ===============
	boost::mutex mapMutex;
	std::vector<FrameShell*> allKeyFramesHistory;

	EnergyFunctional* ef;
	IndexThreadReduce<Vec10> treadReduce;

	float* selectionMap;
	PixelSelector* pixelSelector;
    CoarseDistanceMapWideAngle* coarseDistanceMap;

    dso::Map recordMap;//for recording and loopclosing
	std::vector<PointFrameResidual*> activeResiduals;
	float currentMinActDist;

    //for local optimization
    Rank1FactorizationOpt localOPtiMethod;
    std::vector<FrameHessian*> localOptiFrames;
    int frameCount;
    void localOPtimizationBetweenKeyFrames(int localWindowSize=0);
    //loopclosing
    LoopClosing _loopClosing;
    bool needToRecordKF;
    int needToRecordKFCount;


	std::vector<float> allResVec;



	// mutex etc. for tracker exchange.
	boost::mutex coarseTrackerSwapMutex;			// if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
    CoarseTrackerWideAngle* coarseTracker_forNewKF;			// set as as reference. protected by [coarseTrackerSwapMutex].
    CoarseTrackerWideAngle* coarseTracker;					// always used to track new frames. protected by [trackMutex].

    ImageMatch _imageMatch;
	float minIdJetVisTracker, maxIdJetVisTracker;
	float minIdJetVisDebug, maxIdJetVisDebug;





	// mutex for camToWorl's in shells (these are always in a good configuration).
	boost::mutex shellPoseMutex;



/*
 * tracking always uses the newest KF as reference.
 *
 */

	void makeKeyFrame( FrameHessian* fh);
	void makeNonKeyFrame( FrameHessian* fh);
    void updateScales(std::vector<FrameHessian*> currentFhs,FrameHessian* loopFh,std::vector<std::vector<float>> idepthPairVec);
    void updateScales(float scale);
	void deliverTrackedFrame(FrameHessian* fh, bool needKF);
	void mappingLoop();
    void checkLoopClose(FrameHessian* fh);


	// tracking / mapping synchronization. All protected by [trackMapSyncMutex].
	boost::mutex trackMapSyncMutex;
	boost::condition_variable trackedFrameSignal;
	boost::condition_variable mappedFrameSignal;
	std::deque<FrameHessian*> unmappedTrackedFrames;
	int needNewKFAfter;	// Otherwise, a new KF is *needed that has ID bigger than [needNewKFAfter]*.
	boost::thread mappingThread;
	bool runMapping;
	bool needToKetchupMapping;

	int lastRefStopID;
};
}

