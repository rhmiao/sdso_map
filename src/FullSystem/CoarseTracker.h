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

 
#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"




namespace dso
{
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class CoarseTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CoarseTracker() {}
	CoarseTracker(int w, int h);
    virtual  ~CoarseTracker();

    virtual bool trackNewestCoarse(
			FrameHessian* newFrameHessian,
			SE3 &lastToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort,
			IOWrap::Output3DWrapper* wrap=0);

    void setCoarseTrackingRef(
            std::vector<FrameHessian*> frameHessians,std::vector<FrameHessian*> recordFrameHessians,CalibHessian* HCalib);

    virtual void makeK(
			CalibHessian* HCalib);

	bool debugPrint, debugPlot;

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

    virtual void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    virtual void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

	FrameHessian* lastRef;
	AffLight lastRef_aff_g2l;
	FrameHessian* newFrame;
	int refFrameID;

	// act as pure ouptut
	Vec5 lastResiduals;
    Vec6 lastFlowIndicators;
	double firstCoarseRMSE;
    float* idepth[PYR_LEVELS];

    virtual void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians1,std::vector<FrameHessian*> frameHessians2,CalibHessian* HCalib);
	float* weightSums[PYR_LEVELS];
	float* weightSums_bak[PYR_LEVELS];


	Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
    virtual Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
	void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

	// pc buffers
	float* pc_u[PYR_LEVELS];
	float* pc_v[PYR_LEVELS];
	float* pc_idepth[PYR_LEVELS];
	float* pc_color[PYR_LEVELS];
	int pc_n[PYR_LEVELS];

	// warped buffers
	float* buf_warped_idepth;
	float* buf_warped_u;
	float* buf_warped_v;
	float* buf_warped_dx;
	float* buf_warped_dy;
	float* buf_warped_residual;
	float* buf_warped_weight;
	float* buf_warped_refColor;
	int buf_warped_n;


    std::vector<float*> ptrToDelete;


	Accumulator9 acc;
};


class CoarseDistanceMap {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseDistanceMap() {}
	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

    virtual void makeDistanceMap(
			std::vector<FrameHessian*> frameHessians,
			FrameHessian* frame);

    virtual void makeInlierVotes(
			std::vector<FrameHessian*> frameHessians);

    virtual void makeK( CalibHessian* HCalib);


	float* fwdWarpedIDDistFinal;

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

    virtual void addIntoDistFinal(int u, int v);


	PointFrameResidual** coarseProjectionGrid;
	int* coarseProjectionGridNum;
	Eigen::Vector2i* bfsList1;
	Eigen::Vector2i* bfsList2;

	void growDistBFS(int bfsNum);
};

class CoarseTrackerWideAngle:public CoarseTracker
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseTrackerWideAngle(int w, int h);
    virtual ~CoarseTrackerWideAngle();

    virtual bool trackNewestCoarse(
            FrameHessian* newFrameHessian,
            SE3 &lastToNew_out, AffLight &aff_g2l_out,
            int coarsestLvl, Vec5 minResForAbort,
            IOWrap::Output3DWrapper* wrap=0);

    virtual void makeK(
            CalibHessian* HCalib);

    float waddright;
    float haddbottom;
    float waddleft;
    float haddup;
    float projectMaxX;
    float projectMinX;
    float projectMaxY;
    float projectMinY;

    virtual void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    virtual void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

    // act as pure ouptut
    float* color[PYR_LEVELS];

    virtual void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians1,std::vector<FrameHessian*> frameHessians2,CalibHessian* HCalib);

    virtual Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
    float calcShift(float diff,float minDiff,float maxDiff);
};


class CoarseDistanceMapWideAngle:public CoarseDistanceMap
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseDistanceMapWideAngle(int w, int h);
    virtual ~CoarseDistanceMapWideAngle();

    virtual void makeDistanceMap(
            std::vector<FrameHessian*> frameHessians,
            FrameHessian* frame);

    virtual void makeInlierVotes(
            std::vector<FrameHessian*> frameHessians);

    virtual void makeK( CalibHessian* HCalib);

    float waddright;
    float haddbottom;
    float waddleft;
    float haddup;

    virtual void addIntoDistFinal(int u, int v);

};
/*
class CoarseTrackerPanorama:public CoarseTracker
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseTrackerPanorama(int panoramaF);
    virtual ~CoarseTrackerPanorama();

    virtual bool trackNewestCoarse(
            FrameHessian* newFrameHessian,
            SE3 &lastToNew_out, AffLight &aff_g2l_out,
            int coarsestLvl, Vec5 minResForAbort,
            IOWrap::Output3DWrapper* wrap=0);

    virtual void makeK(
            CalibHessian* HCalib);

    virtual void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    virtual void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

    // act as pure ouptut
    float* color[PYR_LEVELS];
    float panoramaFocal[PYR_LEVELS];

    virtual void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians1,std::vector<FrameHessian*> frameHessians2,CalibHessian* HCalib);

    virtual Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
    void pinholeToPanorama(int pinhole_u,int pinhole_v,float depth,int &pano_u, int &pano_v,int level);
    void panoramaToPinhole(int pano_u, int pano_v,int &pinhole_u,int &pinhole_v,int level);
};


class CoarseDistanceMapPanorama:public CoarseDistanceMap
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseDistanceMapPanorama(int w, int h);
    virtual ~CoarseDistanceMapPanorama();

    virtual void makeDistanceMap(
            std::vector<FrameHessian*> frameHessians,
            FrameHessian* frame);

    virtual void makeInlierVotes(
            std::vector<FrameHessian*> frameHessians);

    virtual void makeK( CalibHessian* HCalib);

    virtual void addIntoDistFinal(int u, int v);

};
*/


}



