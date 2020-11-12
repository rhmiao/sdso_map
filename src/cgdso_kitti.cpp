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


#include "ros/ros.h"
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdexcept>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"
#include "stdio.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
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
#include "OptimizationBackend/CeresOpt.h"


#include "util/ImageAndExposure.h"
#include "boost/endian/conversion.hpp"


#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Imu.h"
#include "geometry_msgs/TransformStamped.h"
#include "geometry_msgs/PointStamped.h"
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace dso;

#define DEQUE_SIZE 5
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicyCam;

std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;
bool viconData= false;

int mode=0;

bool firstRosSpin=false;

FullSystem* fullSystem = 0;

IOWrap::PangolinDSOViewer* viewer = 0;
IOWrap::PangolinDSOViewer* recordViewer = 0;
ImageFolderReader* readerl = 0;
ImageFolderReader* readerr = 0;
clock_t started = clock();
cv::Mat cam_P;
cv::Mat image_size;
double imu_time_last = 0;
std::deque<std::pair<ImageAndExposure<float>*,double>> leftImgQ,rightImgQ;
std::deque<dso::SE3*> viconQ;
dso::SE3 viconPose;
boost::mutex imageMutex;
boost::mutex imuMutex;
std::ofstream* kittiPoseLog=new std::ofstream();
std::ofstream* runtimeLog=new std::ofstream();

double sInitializerOffset=0;
bool threadQuit=false;
int frameCount=0;
std::vector<double> timestampVec;
int needLoadMap = 0;
int useStereo =0;
int testStereoMatch =0;
float baseline =0.1f;
bool stopped=false;
bool cameraInfoUpdated=true;
int lostCount = 0;

ros::Publisher cloud_pub;
ros::Publisher pose_pub;
ros::Publisher trajectory_pub;
nav_msgs::Path dso_path;

void my_exit_handler(int s)
{
    printf("Caught signal %d\n",s);
    exit(1);
}

void exitThread()
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    firstRosSpin=true;
    while(true) pause();
}

void readParam()
{
    cv::FileStorage fs(settingFile, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Could not open the setting file: \"" << settingFile << "\"" << std::endl;
        exit(-1);
    }
    fs["loadMap"] >> needLoadMap;
    fs["testStereoMatch"] >>testStereoMatch;
    fs["loadMapPath"] >> loadMapPath;
    fs["saveMapPath"] >> saveMapPath;
    fs["vocPath"] >> vocPath;

    cv::FileStorage platform_fs(platformFile, cv::FileStorage::READ);
    if (!platform_fs.isOpened())
    {
        std::cout << "Could not open the setting file: \"" << platformFile << "\"" << std::endl;
        exit(-1);
    }
    platform_fs["useStereo"] >> useStereo;
    platform_fs["baseline"] >> baseline;
    platform_fs["max_speed"]>>max_speed;
    platform_fs["panoramaTracker"]>>setting_panoramaTracker;
    ACC_N = platform_fs["acc_n"];
    ACC_W = platform_fs["acc_w"];
    GYR_N = platform_fs["gyr_n"];
    GYR_W = platform_fs["gyr_w"];
    G.z() = platform_fs["g_norm"];
    platform_fs["cameraConfigPath"]>>stereo_param_file;
    std::cout<<"cameraConfigPath:"<<stereo_param_file<<std::endl;


    cv::FileStorage cam_fs(stereo_param_file, cv::FileStorage::READ);
    if (!cam_fs.isOpened())
    {
        std::cout << "Could not open the setting file: \"" << stereo_param_file << "\"" << std::endl;
        exit(-1);
    }
    cam_fs["P1"] >> cam_P;
    cam_fs["size"] >> image_size;
    int w = image_size.at<int>(0);
    int h = image_size.at<int>(1);
    Eigen::Matrix3f ROS_K;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            ROS_K(i,j) = cam_P.at<double>(i,j);
    readerl->setGlobalCalibration(w,h,ROS_K,baseline);
    printf("New Camera Matrix:\n");
    std::cout << ROS_K << "\n\n";

    fs.release();
    platform_fs.release();
    cam_fs.release();


}

void settingsDefault(int preset)
{
    printf("\n=============== PRESET Settings: ===============\n");
    if(preset == 0 || preset == 1)
    {
        printf("DEFAULT settings:\n"
                "- %s real-time enforcing\n"
                "- 2000 active points\n"
                "- 5-7 active frames\n"
                "- 1-6 LM iteration each KF\n"
                "- original image resolution\n", preset==0 ? "no " : "1x");

        playbackSpeed = (preset==0 ? 0 : 1);
        preload = preset==1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations=6;
        setting_minOptIterations=1;

        setting_logStuff = false;
    }

    if(preset == 2 || preset == 3)
    {
        printf("FAST settings:\n"
                "- %s real-time enforcing\n"
                "- 800 active points\n"
                "- 4-6 active frames\n"
                "- 1-4 LM iteration each KF\n"
                "- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

        playbackSpeed = (preset==2 ? 0 : 5);
        preload = preset==3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations=4;
        setting_minOptIterations=1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}

void parseArgument(char* arg)
{
    int option;
    float foption;
    char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }


    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if(1==sscanf(arg,"preset=%d",&option))
    {
        settingsDefault(option);
        return;
    }


    if(1==sscanf(arg,"rec=%d",&option))
    {
        if(option==0)
        {
            disableReconfigure = true;
            printf("DISABLE RECONFIGURE!\n");
        }
        return;
    }



    if(1==sscanf(arg,"noros=%d",&option))
    {
        if(option==1)
        {
            disableROS = true;
            disableReconfigure = true;
            printf("DISABLE ROS (AND RECONFIGURE)!\n");
        }
        return;
    }

    if(1==sscanf(arg,"nolog=%d",&option))
    {
        if(option==1)
        {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }
    if(1==sscanf(arg,"reverse=%d",&option))
    {
        if(option==1)
        {
            reverse = true;
            printf("REVERSE!\n");
        }
        return;
    }
    if(1==sscanf(arg,"nogui=%d",&option))
    {
        if(option==1)
        {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if(1==sscanf(arg,"nomt=%d",&option))
    {
        if(option==1)
        {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }
    if(1==sscanf(arg,"prefetch=%d",&option))
    {
        if(option==1)
        {
            prefetch = true;
            printf("PREFETCH!\n");
        }
        return;
    }
    if(1==sscanf(arg,"start=%d",&option))
    {
        start = option;
        printf("START AT %d!\n",start);
        return;
    }
    if(1==sscanf(arg,"end=%d",&option))
    {
        end = option;
        printf("END AT %d!\n",start);
        return;
    }

    if(1==sscanf(arg,"platformSettingPath=%s",buf))
    {
        platformFile=buf;
        printf("platform is %s!\n", platformFile.c_str());
        return;
    }

    if(1==sscanf(arg,"files=%s",buf))
    {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
        return;
    }

    if(1==sscanf(arg,"calib=%s",buf))
    {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }

    if(1==sscanf(arg,"vignette=%s",buf))
    {
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }

    if(1==sscanf(arg,"gamma=%s",buf))
    {
        gammaCalib = buf;
        printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
        return;
    }

    if(1==sscanf(arg,"rescale=%f",&foption))
    {
        rescale = foption;
        printf("RESCALE %f!\n", rescale);
        return;
    }

    if(1==sscanf(arg,"speed=%f",&foption))
    {
        playbackSpeed = foption;
        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
        return;
    }

    if(1==sscanf(arg,"save=%d",&option))
    {
        if(option==1)
        {
            debugSaveImages = true;
            if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            printf("SAVE IMAGES!\n");
        }
        return;
    }

    if(1==sscanf(arg,"mode=%d",&option))
    {

        mode = option;
        if(option==0)
        {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if(option==1)
        {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        }
        if(option==2)
        {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
        }
        return;
    }
    printf("could not parse argument \"%s\"!!!!\n", arg);
}

void fullSystemReset()
{
    printf("RESETTING!\n");

    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
    IOWrap::Output3DWrapper* recordWrap=fullSystem->recordWrapper;
    delete fullSystem;

    for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

    recordWrap->reset();

    boost::unique_lock<boost::mutex> imulock(imuMutex);
    fullSystem = new FullSystem();
    imulock.unlock();
    fullSystem->setGammaFunction(readerl->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed==0);

    if(!disableAllDisplay)
    {
        fullSystem->outputWrapper.push_back(viewer);
        fullSystem->recordWrapper = recordViewer;
    }

    fullSystem->isLoadMap=needLoadMap;
    if(fullSystem->isLoadMap)fullSystem->loadMap();


    setting_fullResetRequested=false;
}

void writeKittiPose(dso::SE3 pose,int i)
{
    pose.normalize();
    (*kittiPoseLog) << std::setprecision(7)
                     << pose.matrix3x4()(0,0)<<" "<< pose.matrix3x4()(0,1)<<" "<< pose.matrix3x4()(0,2)<<" "<< pose.matrix3x4()(0,3)<<" "
                     << pose.matrix3x4()(1,0)<<" "<< pose.matrix3x4()(1,1)<<" "<< pose.matrix3x4()(1,2)<<" "<< pose.matrix3x4()(1,3)<<" "
                     << pose.matrix3x4()(2,0)<<" "<< pose.matrix3x4()(2,1)<<" "<< pose.matrix3x4()(2,2)<<" "<< pose.matrix3x4()(2,3)<<"\n";
}

void writeRunTime(double t)
{
    (*runtimeLog) << std::setprecision(7)
                     << t <<"\n";
}

void dsoLoop()
{
    if(fullSystem->initFailed || setting_fullResetRequested)fullSystemReset();
    if(fullSystem->isLost)
    {
        printf("LOST!!\n");
        lostCount += 1;
        if(lostCount>=5){
            fullSystemReset();
            lostCount = 0;
        }
        //        threadQuit = true;
    }
    else{
        lostCount = 0;
    }
}


cv::Ptr<cv::TonemapDrago> tonemap = cv::createTonemapDrago(1.5f,0.5F);

void viconCb(const geometry_msgs::TransformStampedPtr& msg)
{

    double w,x,y,z;
    w=msg->transform.rotation.w;
    x=msg->transform.rotation.x;
    y=msg->transform.rotation.y;
    z=msg->transform.rotation.z;
    double tx,ty,tz;
    tx= msg->transform.translation.x;
    ty= msg->transform.translation.y;
    tz= msg->transform.translation.z;
    Sophus::Quaterniond q1(w,x,y,z);
    Sophus::Quaterniond q2(0.5,-0.5,0.5,-0.5);
    Sophus::Quaterniond q3=q2.inverse()*q1*q2;
    viconPose = dso::SE3(q3,dso::Vec3(-ty,-tz,tx));
    viconData=true;

}

int main( int argc, char** argv )
{
//    cv::Mat a=cv::imread("/home/mrh/Record/cgdso_test/vignette.png");
//    cv::Mat b;
//    cv::resize(a,b,cv::Size(384,512));
//    cv::imwrite("/home/mrh/Record/cgdso_test/vignette_devon.png",b);

    ros::init(argc, argv,"cgdso");
    ros::NodeHandle nh;

    google::InitGoogleLogging((const char *)argv[0]);

    google::SetLogDestination(google::GLOG_INFO, "/home/mrh/Record/cgdso_test/logs/cgdso_Info");

    kittiPoseLog->open("logs/kittiPose.txt", std::ios::trunc | std::ios::out);
    runtimeLog->open("logs/runTimes.txt", std::ios::trunc | std::ios::out);

    LOG(INFO) << "[main]This is the 1st log info!";

    for(int i=1; i<argc;i++)
        parseArgument(argv[i]);

    readerl = new ImageFolderReader(source+"/image_0",calib, gammaCalib, vignette);
    readerr = new ImageFolderReader(source+"/image_1",calib, gammaCalib, vignette);
//    readerl = new ImageFolderReader(source+"/left",calib, gammaCalib, vignette);
//    readerr = new ImageFolderReader(source+"/right",calib, gammaCalib, vignette);

    readParam();

    cloud_pub= nh.advertise<sensor_msgs::PointCloud>("/cgdso/cloud", 50);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/cgdso/pose", 1);
    trajectory_pub = nh.advertise<nav_msgs::Path>("/cgdso/trajectory", 1);
    ros::Rate r(100.0);

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);

//    example();


    if(setting_photometricCalibration > 0 && readerl->getPhotometricGamma() == 0)
    {
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }

    int lstart=start;
    int lend = end;
    int linc = 1;
    if(reverse)
    {
        printf("REVERSE!!!!");
        lstart=end-1;
        if(lstart >= readerl->getNumImages())
            lstart = readerl->getNumImages()-1;
        lend = start;
        linc = -1;
    }

    fullSystem = new FullSystem();
    fullSystem->setGammaFunction(readerl->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed==0);


    if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0],"Main", false);
        fullSystem->outputWrapper.push_back(viewer);

        recordViewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0],"record",true);
        fullSystem->recordWrapper = recordViewer;
    }

    fullSystem->isLoadMap=needLoadMap;
    if(fullSystem->isLoadMap)fullSystem->loadMap();



    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    std::thread rosthread([&]() {
        while (ros::ok()&&!threadQuit){
            ros::spinOnce();
            r.sleep();
        }

    });

    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for(int i=lstart;i>= 0 && i< readerl->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = readerl->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = readerl->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }


        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;

        bool stopped=false;
        for(int ii=50;ii<(int)idsToPlay.size(); ii++)
        {
            if(!stopped){
                if(!fullSystem->initialized)	// if not initialized: reset start time.
                {
                    gettimeofday(&tv_start, NULL);
                    started = clock();
                    sInitializerOffset = timesToPlayAt[ii];
                }

                int i = idsToPlay[ii];


                MinimalImageB* imgMBl,*imgMBr;

                imgMBl = readerl->getImageRaw(i);
                imgMBr = readerr->getImageRaw(i);

                ImageAndExposure<float>* imgEl=new ImageAndExposure<float>(imgMBl->w, imgMBl->h,readerl->getTimestamp(i));
                ImageAndExposure<float>* imgEr=new ImageAndExposure<float>(imgMBr->w, imgMBr->h,readerl->getTimestamp(i));
                for(int i=0;i<(imgMBl->w)*(imgMBl->h);++i){
                    imgEl->image[i]=1.0*(imgMBl->data[i]);
                    imgEr->image[i]=1.0*(imgMBr->data[i]);
                }
                delete imgMBr;
                delete imgMBl;

                std::vector<double> kitti_gt=readerl->getGroundtruth(i);
                Eigen::Matrix<double,4,4> se3m;
                if(kitti_gt.size()>0){
                    se3m << kitti_gt[0], kitti_gt[1], kitti_gt[2],kitti_gt[3],
                            kitti_gt[4], kitti_gt[5], kitti_gt[6],kitti_gt[7],
                            kitti_gt[8], kitti_gt[9], kitti_gt[10],kitti_gt[11],
                            0,0,0,1;
                }
                dso::SE3 viconPose(se3m);
//                viconPose = ;dso::SE3(q3,dso::Vec3(-ty,-tz,tx));


                bool skipFrame=false;
                if(playbackSpeed!=0)
                {
                    struct timeval tv_now; gettimeofday(&tv_now, NULL);
                    double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

                    if(sSinceStart < timesToPlayAt[ii])
                        usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
                    else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
                    {
                        printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                        skipFrame=true;
                    }
                }



                if(!skipFrame)
                {
                    if(testStereoMatch){
                        cv::Mat depthMat(imgEl->h, imgEl->w, CV_32FC3, cv::Scalar(0,0,0));
                        std::vector<cv::Point3f> KeyPoints3d;
                        KeyPoints3d.clear();
                        fullSystem->stereoMatch(imgEl,imgEr,i,depthMat,KeyPoints3d);
                        sensor_msgs::PointCloud cloud;
                        cloud.header.stamp = ros::Time::now();
                        cloud.header.frame_id = "dso_frame";
                        //we'll also add an intensity channel to the cloud
                        cloud.channels.resize(1);
                        cloud.channels[0].name = "rgb";
                        if(KeyPoints3d.size()==0){
                            std::vector<cv::Point3f> pts;
                            for(int i=0;i<depthMat.cols;++i)
                                for(int j=0;j<depthMat.rows;++j){
                                    if(depthMat.at<cv::Vec3f>(j,i)[0]!=0){
                                        cv::Point3f pt;
                                        pt.x=i;
                                        pt.y=j;
                                        pt.z=depthMat.at<cv::Vec3f>(j,i)[0];
                                        pts.push_back(pt);

                                    }
                                }
                            cloud.points.resize(pts.size());
                            cloud.channels[0].values.resize(pts.size());
                            for(int i=0;i<pts.size();i++){
                                cloud.points[i].x=pts[i].z;
                                cloud.points[i].y=-(pts[i].x-cxG[0])*fxiG[0]*pts[i].z;
                                cloud.points[i].z = -(pts[i].y-cyG[0])*fyiG[0]*pts[i].z;
                                cloud.channels[0].values[i] = 255;
                            }
                        }
                        else
                        {
                            cloud.points.resize(KeyPoints3d.size());
                            cloud.channels[0].values.resize(KeyPoints3d.size());
                            for(int i=0;i<KeyPoints3d.size();i++){
                                cloud.points[i].x=KeyPoints3d[i].z;
                                cloud.points[i].y=KeyPoints3d[i].x;
                                cloud.points[i].z = KeyPoints3d[i].y;
                                cloud.channels[0].values[i] = 255;
                            }
                        }
                        cloud_pub.publish(cloud);
                    }
                    else
                    {
                        if(fullSystem->initialized)fullSystem->viconNow=fullSystem->viconFirst.inverse()*viconPose;
                        else fullSystem->viconNow=viconPose;
                        int64 start=0,end=0;
                        start = cv::getTickCount();
                        fullSystem->addActiveFrame(imgEl,imgEr, i);
                        end = cv::getTickCount();
                        writeRunTime(1000.0*(end - start)/cv::getTickFrequency());
                        writeKittiPose(fullSystem->camNow,i);
                    }
                }

//                MinimalImageF Mimg(wG[0], hG[0], imgEl->image);
//                IOWrap::displayImage("frameToTrack", &Mimg);


                delete imgEl;
                if(imgEr!=0)delete imgEr;

                frameCount++;
                geometry_msgs::PoseStamped poseNow;
                poseNow.header.stamp = ros::Time::now();
                poseNow.header.frame_id = "dso_frame";
                Eigen::Quaterniond qC(fullSystem->camNow.rotationMatrix());
                Sophus::Quaterniond qWC(0.5,0.5,-0.5,0.5);
                Sophus::Quaterniond qW=qWC.inverse()*qC*qWC;
                poseNow.pose.orientation.w = qW.w();
                poseNow.pose.orientation.x = qW.x();
                poseNow.pose.orientation.y = qW.y();
                poseNow.pose.orientation.z = qW.z();
                poseNow.pose.position.x = fullSystem->camNow.translation()(0);
                poseNow.pose.position.y = fullSystem->camNow.translation()(2);
                poseNow.pose.position.z = -fullSystem->camNow.translation()(1);
                pose_pub.publish(poseNow);

                dso_path.header.frame_id = "dso_frame";
                dso_path.poses.push_back(poseNow);
                trajectory_pub.publish(dso_path);

                dsoLoop();
            }
            bool quit=false;
            int ch=IOWrap::waitKey(1);
            switch (ch)
            {
            case 3:
                printf("key:ctrl+c(quit thread function)\n\r");
                quit=true;
                break;
            case 'q':
                printf("key:q(quit thread function)\n\r");
                quit=true;
                break;
            case 's':
                fullSystem->saveMap();
                printf("key:s(save map!)\n\r");
                break;
            case ' ':
                stopped=!stopped;
                if(stopped)printf("key:space(stop dso!)\n\r");
                else printf("key:space(start dso!)\n\r");
                break;
            default:
                break;
            }
            if(quit){
                viewer->close();
                recordViewer->close();
                threadQuit = true;
            }
            if(threadQuit)break;
        }

        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        fullSystem->blockUntilMappingIsFinished();
        kittiPoseLog->close();
        runtimeLog->close();
        fullSystem->closeLogFile();
        threadQuit=true;
        //save map
        fullSystem->saveMap();

        fullSystem->printResult("result.txt");

        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog.flush();
            tmlog.close();
        }

    });


    if(viewer != 0)
        viewer->run();

    runthread.join();
    rosthread.join();


    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }


    fullSystem->recordWrapper->join();
    delete fullSystem->recordWrapper;


    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    printf("DELETE READER!\n");
    delete readerl;
    delete readerr;

    printf("EXIT NOW!\n");
    ros::shutdown();

    google::ShutdownGoogleLogging();
    return 0;

}
