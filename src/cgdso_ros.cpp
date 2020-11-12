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
ImageFolderReader* reader = 0;
clock_t started = clock();
cv::Mat cam_P;
cv::Mat image_size;
double imu_time_last = 0;
std::deque<std::pair<ImageAndExposure<float>*,double>> leftImgQ,rightImgQ;
std::deque<dso::SE3*> viconQ;
dso::SE3 viconPose;
boost::mutex imageMutex;
boost::mutex imuMutex;

// Compression formats
enum compressionFormat
{
  UNDEFINED = -1, INV_DEPTH
};

// Compression configuration
struct ConfigHeader
{
  // compression format
  compressionFormat format;
  // quantization parameters (used in depth image compression)
  float depthParam[2];
};


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

Eigen::Matrix3d v2cm;
dso::SE3 vicon2cam;

Eigen::Matrix3d lc2cm;
dso::SE3 leica2cm;

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
    platform_fs["panoramaTracker"]>>setting_panoramaTracker;
    platform_fs["baseline"] >> baseline;
    platform_fs["max_speed"]>>max_speed;
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
    reader->setGlobalCalibration(w,h,ROS_K,baseline);
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
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = true;//(playbackSpeed==0);

    if(!disableAllDisplay)
    {
        fullSystem->outputWrapper.push_back(viewer);
        fullSystem->recordWrapper = recordViewer;
    }

    fullSystem->isLoadMap=needLoadMap;
    if(fullSystem->isLoadMap)fullSystem->loadMap();


    setting_fullResetRequested=false;
}

void dsoLoop()
{
    if(fullSystem->initFailed || setting_fullResetRequested)fullSystemReset();
    if(fullSystem->isLost)
    {
        printf("LOST!!\n");
        lostCount += 1;
        if(lostCount>=10){
            fullSystemReset();
            lostCount = 0;
        }
        //        threadQuit = true;
    }
    else{
        lostCount = 0;
    }
}

void depthCompressedCb(const sensor_msgs::CompressedImagePtr& msg)
{
      cv::Mat image;
      // Decode message data
      if (msg->data.size() > sizeof(ConfigHeader))
      {

        // Read compression type from stream
        ConfigHeader compressionConfig;
        memcpy(&compressionConfig, &msg->data[0], sizeof(compressionConfig));

        // Get compressed image data
        const std::vector<uint8_t> imageData(msg->data.begin() + sizeof(compressionConfig), msg->data.end());

        // Depth map decoding
        float depthQuantA, depthQuantB;

        // Read quantization parameters
        depthQuantA = compressionConfig.depthParam[0];
        depthQuantB = compressionConfig.depthParam[1];

        cv::Mat decompressed;
        // Decode image data
        decompressed = cv::imdecode(imageData, cv::IMREAD_UNCHANGED);

        size_t rows = decompressed.rows;
        size_t cols = decompressed.cols;

        if ((rows > 0) && (cols > 0))
        {
          image = cv::Mat(rows, cols, CV_32FC1);

          // Depth conversion
          cv::MatIterator_<float> itDepthImg = image.begin<float>(),
                              itDepthImg_end = image.end<float>();
          cv::MatConstIterator_<unsigned short> itInvDepthImg = decompressed.begin<unsigned short>(),
                                            itInvDepthImg_end = decompressed.end<unsigned short>();

          for (; (itDepthImg != itDepthImg_end) && (itInvDepthImg != itInvDepthImg_end); ++itDepthImg, ++itInvDepthImg)
          {
            // check for NaN & max depth
            if (*itInvDepthImg)
            {
              *itDepthImg = depthQuantA / ((float)*itInvDepthImg - depthQuantB);
            }
            else
            {
              *itDepthImg = std::numeric_limits<float>::quiet_NaN();
            }
          }

        }

        cv::imshow("ss",image);
      }
}

void imgCompressedCb(const sensor_msgs::CompressedImagePtr& msg){
    cv::Mat imgl = cv::imdecode(cv::Mat(msg->data),1),imgMat;

    cv::Size size=cv::Size(wG[0],hG[0]);
    cv::resize(imgl,imgMat,size,CV_INTER_LINEAR);
    cv::cvtColor(imgMat,imgMat,CV_BGR2GRAY);
    double timestamp = msg->header.stamp.sec + msg->header.stamp.nsec*1e-9;
    timestampVec.push_back(timestamp);

    cv::imshow("frameToTrack",imgMat);
    cv::waitKey(1);
    if(!fullSystem->initialized)	// if not initialized: reset start time.
    {
        started = clock();
    }

    ImageAndExposure<float>* img=new ImageAndExposure<float>(imgMat.cols, imgMat.rows, timestamp);
    ImageAndExposure<float>* imgr=0;
    for(int i=0;i<imgMat.cols*imgMat.rows;++i)
        img->image[i]=1.0*imgMat.data[i];


    fullSystem->addActiveFrame(img,imgr, frameCount);
    frameCount++;

    delete img;
    delete imgr;

    if(fullSystem->initFailed || setting_fullResetRequested)
    {
        printf("RESETTING!\n");

        std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
        IOWrap::Output3DWrapper* recordWrap=fullSystem->recordWrapper;
        delete fullSystem;

        for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
        recordWrap->reset();

        fullSystem = new FullSystem();
        fullSystem->setGammaFunction(reader->getPhotometricGamma());
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
    if(fullSystem->isLost)
    {
        printf("LOST!!\n");
//        threadQuit = true;
    }

}

int getCvType(const std::string& encoding)
{
  // Check for the most common encodings first
  if (encoding == sensor_msgs::image_encodings::BGR8)   return CV_8UC3;
  if (encoding == sensor_msgs::image_encodings::MONO8)  return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::RGB8)   return CV_8UC3;
  if (encoding == sensor_msgs::image_encodings::MONO16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BGR16)  return CV_16UC3;
  if (encoding == sensor_msgs::image_encodings::RGB16)  return CV_16UC3;
  if (encoding == sensor_msgs::image_encodings::BGRA8)  return CV_8UC4;
  if (encoding == sensor_msgs::image_encodings::RGBA8)  return CV_8UC4;
  if (encoding == sensor_msgs::image_encodings::BGRA16) return CV_16UC4;
  if (encoding == sensor_msgs::image_encodings::RGBA16) return CV_16UC4;

  // For bayer, return one-channel
  if (encoding == sensor_msgs::image_encodings::BAYER_RGGB8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_BGGR8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GBRG8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GRBG8) return CV_8UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_RGGB16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_BGGR16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GBRG16) return CV_16UC1;
  if (encoding == sensor_msgs::image_encodings::BAYER_GRBG16) return CV_16UC1;

  // Miscellaneous
  if (encoding == sensor_msgs::image_encodings::YUV422) return CV_8UC2;
}

cv::Mat matFromImage(const sensor_msgs::Image& source)
{
  int source_type = getCvType(source.encoding);
  int byte_depth = sensor_msgs::image_encodings::bitDepth(source.encoding) / 8;
  int num_channels = sensor_msgs::image_encodings::numChannels(source.encoding);

  if (source.step < source.width * byte_depth * num_channels)
  {
    std::stringstream ss;
    ss << "Image is wrongly formed: step < width * byte_depth * num_channels  or  " << source.step << " != " <<
        source.width << " * " << byte_depth << " * " << num_channels;
    throw ss.str();
  }

  if (source.height * source.step != source.data.size())
  {
    std::stringstream ss;
    ss << "Image is wrongly formed: height * step != size  or  " << source.height << " * " <<
              source.step << " != " << source.data.size();
    throw ss.str();
  }

  // If the endianness is the same as locally, share the data
  cv::Mat mat(source.height, source.width, source_type, const_cast<uchar*>(&source.data[0]), source.step);
  if ((boost::endian::order::native == boost::endian::order::big && source.is_bigendian) ||
      (boost::endian::order::native == boost::endian::order::little && !source.is_bigendian) ||
      byte_depth == 1)
    return mat;

  // Otherwise, reinterpret the data as bytes and switch the channels accordingly
  mat = cv::Mat(source.height, source.width, CV_MAKETYPE(CV_8U, num_channels*byte_depth),
                const_cast<uchar*>(&source.data[0]), source.step);
  cv::Mat mat_swap(source.height, source.width, mat.type());

  std::vector<int> fromTo;
  fromTo.reserve(num_channels*byte_depth);
  for(int i = 0; i < num_channels; ++i)
    for(int j = 0; j < byte_depth; ++j)
    {
      fromTo.push_back(byte_depth*i + j);
      fromTo.push_back(byte_depth*i + byte_depth - 1 - j);
    }
  cv::mixChannels(std::vector<cv::Mat>(1, mat), std::vector<cv::Mat>(1, mat_swap), fromTo);

  // Interpret mat_swap back as the proper type
  mat_swap.reshape(num_channels);

  return mat_swap;
}

void imgCb(const sensor_msgs::ImagePtr& msg){
    cv::Mat imgl=matFromImage(*msg),imgMat;
    cv::Size size=cv::Size(wG[0],hG[0]);
    cv::resize(imgl,imgMat,size,CV_INTER_LINEAR);
    cv::cvtColor(imgMat,imgMat,CV_BGR2GRAY);
    double timestamp = msg->header.stamp.sec + msg->header.stamp.nsec*1e-9;
    timestampVec.push_back(timestamp);

    cv::imshow("frameToTrack",imgMat);
    cv::waitKey(1);
    if(!fullSystem->initialized)	// if not initialized: reset start time.
    {
        started = clock();
    }

    ImageAndExposure<float>* imgEl=new ImageAndExposure<float>(imgMat.cols, imgMat.rows, timestamp);
    for(int i=0;i<imgMat.cols*imgMat.rows;++i)
        imgEl->image[i]=1.0*imgMat.data[i];
    ImageAndExposure<float>* imgEr = 0;
    double timeNow = ros::Time::now().toSec();
    {
        boost::unique_lock<boost::mutex> imagelock(imageMutex);
        leftImgQ.push_front(std::make_pair(imgEl,timeNow));
        rightImgQ.push_front(std::make_pair(imgEr,timeNow));
        if(leftImgQ.size()>=DEQUE_SIZE)
        {
            ImageAndExposure<float>* imgTmp=leftImgQ.back().first;
            leftImgQ.pop_back();
            delete imgTmp;

            imgTmp = rightImgQ.back().first;
            rightImgQ.pop_back();
            if(imgTmp!=0)delete imgTmp;
        }
    }

}
cv::Ptr<cv::TonemapDrago> tonemap = cv::createTonemapDrago(1.5f,0.5F);
void stereoCallback(const sensor_msgs::ImageConstPtr& imagel,const sensor_msgs::ImageConstPtr& imager)
{
    cv::Mat imgl=matFromImage(*imagel),imgr=matFromImage(*imager),imgMatl,imgMatr;
    cv::Size size=cv::Size(wG[0],hG[0]);
    int64 start=0;
    double minvall,maxvall,minvalr,maxvalr;
    start = cv::getTickCount();
//    cv::resize(imgl,imgMatl,size,CV_INTER_LINEAR);
//    cv::resize(imgr,imgMatr,size,CV_INTER_LINEAR);
//    if(imgMatl.channels()==1)cv::cvtColor(imgMatl,imgMatl,CV_GRAY2BGR);
//    if(imgMatr.channels()==1)cv::cvtColor(imgMatr,imgMatr,CV_GRAY2BGR);

//    cv::Mat imgMatlF,imgMatrF,imgMatlTone,imgMatrTone;
//    imgMatl.convertTo(imgMatlTone, CV_32FC3);
//    imgMatr.convertTo(imgMatrTone, CV_32FC3);

//    //! [Tonemap HDR image]
//    tonemap->process(imgMatlTone, imgMatlF);
//    tonemap->process(imgMatrTone, imgMatrF);
//    cv::normalize(imgMatlF, imgMatlF, 0, 255, CV_MINMAX);
//    cv::normalize(imgMatrF, imgMatrF, 0, 255, CV_MINMAX);

//    if(imgMatlF.channels()>1)cv::cvtColor(imgMatlF,imgMatlF,CV_BGR2GRAY);
//    if(imgMatrF.channels()>1)cv::cvtColor(imgMatrF,imgMatrF,CV_BGR2GRAY);


    if(imgl.channels()>1)cv::cvtColor(imgl,imgl,CV_BGR2GRAY);
    cv::resize(imgl,imgMatl,size,CV_INTER_LINEAR);
    if(imgr.channels()>1)cv::cvtColor(imgr,imgr,CV_BGR2GRAY);
    cv::resize(imgr,imgMatr,size,CV_INTER_LINEAR);

    cv::Mat imgMatlF,imgMatrF;

    imgMatl.convertTo(imgMatlF, CV_32F,1,1);
    imgMatr.convertTo(imgMatrF, CV_32F,1,1);

    cv::minMaxLoc(imgMatl,&minvall,&maxvall);
    cv::minMaxLoc(imgMatr,&minvalr,&maxvalr);

    cv::log(imgMatlF,imgMatlF);
    cv::log(imgMatrF,imgMatrF);

    //归一化到0~255
    cv::normalize(imgMatlF, imgMatlF, 255, 0, CV_MINMAX);
    cv::normalize(imgMatrF, imgMatrF, 255, 0, CV_MINMAX);

//    std::cout << "The differences of TONEMAP: " << 1000.0*(cv::getTickCount() - start)/cv::getTickFrequency()<<" ms"<< std::endl;

//    cv::GaussianBlur( imgMatr, imgMatr, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
//    cv::Sobel(imgMatr,grad_x,CV_16S,1,0);
//    cv::convertScaleAbs(grad_x, abs_grad_x);
//    cv::Sobel(imgMatr, grad_y, CV_16S, 0, 1);
//    cv::convertScaleAbs(grad_y, abs_grad_y);
//    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgMatr);
//    cv::medianBlur(imgMatr,imgMatr,3);
//    cv::addWeighted(imgMatr, 1, abs_grad, 0.5, 0, imgMatr);

    double timestamp = imagel->header.stamp.sec + imagel->header.stamp.nsec*1e-9;
    timestampVec.push_back(timestamp);

//    cv::Mat imgMatStereo;
//    cv::hconcat(imgMatl, imgMatr, imgMatStereo);
//    cv::imshow("stereo",imgMatStereo);
//    cv::waitKey(1);
    if(!fullSystem->initialized)	// if not initialized: reset start time.
    {
        started = clock();
    }

    ImageAndExposure<float>* imgEl=new ImageAndExposure<float>(imgMatl.cols, imgMatl.rows, timestamp);
    ImageAndExposure<float>* imgEr=new ImageAndExposure<float>(imgMatr.cols, imgMatr.rows, timestamp);
    for(int i=0;i<imgMatl.cols*imgMatl.rows;++i){
        imgEl->image[i]=1.0*(imgMatl.data[i]);
        imgEr->image[i]=1.0*(imgMatr.data[i]);
    }
//    for(int i=0;i<imgMatl.rows;++i)
//        for(int j=0;j<imgMatl.cols;++j)
//        {
//            imgEl->image[i*imgMatl.cols+j]=imgMatlF.at<float>(i,j);
//            imgEr->image[i*imgMatl.cols+j]=imgMatrF.at<float>(i,j);
//        }
    imgEl->minval=minvall;
    imgEl->maxval=maxvall;
    imgEr->minval=minvalr;
    imgEr->minval=maxvalr;
    if(testStereoMatch){
        cv::Mat depthMat(imgEl->h, imgEl->w, CV_32FC3, cv::Scalar(0,0,0));
        std::vector<cv::Point3f> KeyPoints3d;
        KeyPoints3d.clear();
        fullSystem->stereoMatch(imgEl,imgEr,frameCount,depthMat,KeyPoints3d);
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
        delete imgEl;
        delete imgEr;
    }
    else
    {
        double timeNow = ros::Time::now().toSec();

        boost::unique_lock<boost::mutex> imagelock(imageMutex);
        leftImgQ.push_back(std::make_pair(imgEl,timeNow));
        rightImgQ.push_back(std::make_pair(imgEr,timeNow));
        dso::SE3 *v_pose=new dso::SE3(viconPose.so3(),viconPose.translation());
        viconQ.push_back(v_pose);
//        if(leftImgQ.size()>=DEQUE_SIZE)
//        {
//            ImageAndExposure<float>* imgTmp=leftImgQ.front().first;
//            leftImgQ.pop_front();
//            delete imgTmp;

//            imgTmp = rightImgQ.front().first;
//            rightImgQ.pop_front();
//            if(imgTmp!=0)delete imgTmp;
//        }

    }

}

void imuCb(const sensor_msgs::ImuPtr& msg)
{ 
    boost::unique_lock<boost::mutex> imulock(imuMutex);
    double w,x,y,z;
    w=msg->orientation.w;
    x=msg->orientation.x;
    y=msg->orientation.y;
    z=msg->orientation.z;
    Sophus::Quaterniond q1(w,x,y,z);
    Sophus::Quaterniond q2(0.5,-0.5,0.5,-0.5);
    Sophus::Quaterniond q3=q2.inverse()*q1*q2;
    if(fullSystem->initialized){
        dso::SE3 rotation(q3,dso::Vec3(0,0,0));
        fullSystem->rotationNow=fullSystem->rotationFirst.inverse()*rotation;
    }
    else{
        dso::SE3 rotation(q3,dso::Vec3(0,0,0));
        fullSystem->rotationNow=rotation;
    }
    Eigen::Vector3d acc_0(msg->linear_acceleration.x,-msg->linear_acceleration.z,msg->linear_acceleration.y);
    Eigen::Vector3d gyr_0(msg->angular_velocity.x,-msg->angular_velocity.z,msg->angular_velocity.y);
    double dt = msg->header.stamp.toSec() - imu_time_last;
    imu_time_last = msg->header.stamp.toSec();
    fullSystem->addIMU(acc_0,gyr_0,q3,dt);

//    std::cout<<"fullSystem->rotationNow"<<msg->header.stamp.now()<<fullSystem->rotationNow.rotationMatrix()<<std::endl;
}

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
//    Sophus::Quaterniond q2(0.5,-0.5,0.5,-0.5);
//    Sophus::Quaterniond q3=q2.inverse()*q1*q2;
//    viconPose = vicon2cam*dso::SE3(q3,dso::Vec3(-ty,-tz,tx));
    viconPose = dso::SE3(q1,dso::Vec3(tx,ty,tz))*vicon2cam.inverse();
    viconData=true;

}

void leicaCb(const geometry_msgs::PointStampedPtr& msg)
{
    double w,x,y,z;
    w=1;
    x=0;
    y=0;
    z=0;
    double tx,ty,tz;
    tx= msg->point.x;
    ty= msg->point.y;
    tz= msg->point.z;
    Sophus::Quaterniond q1(w,x,y,z);
    if(!viconData)viconPose = dso::SE3(q1,dso::Vec3(-tx,tz,-ty));
}

void camInfoCb(const sensor_msgs::CameraInfoPtr& msg)
{
    if(cameraInfoUpdated){

    }
}

int main( int argc, char** argv )
{
    ros::init(argc, argv,"cgdso");
    ros::NodeHandle nh;

    google::InitGoogleLogging((const char *)argv[0]);

    google::SetLogDestination(google::GLOG_INFO, "/home/mrh/Record/cgdso_test/logs/cgdso_Info");

    LOG(INFO) << "[main]This is the 1st log info!";

    for(int i=1; i<argc;i++)
        parseArgument(argv[i]);

    reader = new ImageFolderReader(source,calib, gammaCalib, vignette);

    readParam();

    v2cm << -0.040040339936491,-0.999133451137274,0.011533485353292,
                -0.333115040827263,0.002459366958011,-0.942886993857460,
                0.942032684147960,-0.041593839527698, -0.332924225490473;

    vicon2cam.setRotationMatrix(v2cm);

    lc2cm <<    -0.985573665752501,   0.037657385552090,   0.165004456300703,
                0.024519742981225,  -0.932877440129823 ,  0.359358405822668,
               -0.167461432905708,  -0.358220048265695 , -0.918496089029057;

    leica2cm.setRotationMatrix(lc2cm);
    ros::Subscriber leftvidsub;
    ros::Subscriber vidsub;
    message_filters::Subscriber<sensor_msgs::Image> *imagel_sub;
    message_filters::Subscriber<sensor_msgs::Image> *imager_sub;
    message_filters::Synchronizer<syncPolicyCam> *sync;
    if(useStereo == 1) {
        imagel_sub=new message_filters::Subscriber<sensor_msgs::Image>(nh, "/camera/left/image_raw",1);
        imager_sub=new message_filters::Subscriber<sensor_msgs::Image>(nh,  "/camera/right/image_raw", 1);
        sync=new message_filters::Synchronizer<syncPolicyCam>(syncPolicyCam(2), *imagel_sub, *imager_sub);
        sync->registerCallback(boost::bind(&stereoCallback, _1, _2));
    }
    else
    {
        leftvidsub = nh.subscribe("/camera/left/image_raw", 1,imgCb);
        //vidsub = nh.subscribe("/zed/left/image_rect_color/compressed", 1,imgCompressedCb);
    }
//    vidsub = nh.subscribe("/sensors/stereo_cam/depth/depth_registered/compressedDepth", 1,depthCompressedCb);
    ros::Subscriber camInfosub = nh.subscribe("/camera/left/camera_info", 1,camInfoCb);
    ros::Subscriber imusub = nh.subscribe("/imu",1,imuCb);
    ros::Subscriber viconsub = nh.subscribe("/vicon/firefly_sbx/firefly_sbx",1,viconCb);
    ros::Subscriber leica = nh.subscribe("/leica/position",1,leicaCb);
    cloud_pub= nh.advertise<sensor_msgs::PointCloud>("/cgdso/cloud", 50);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/cgdso/pose", 1);
    trajectory_pub = nh.advertise<nav_msgs::Path>("/cgdso/trajectory", 1);
    ros::Rate r(100.0);

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);

//    example();


    if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
    {
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }

    fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = true;//(playbackSpeed==0);


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
        double timeLast=0;
        while (ros::ok()&&!threadQuit){

            if(leftImgQ.size()>0&&rightImgQ.size()>0)
            {
                ImageAndExposure<float>* imgEl;
                ImageAndExposure<float>* imgEr;
                dso::SE3 *v_pose;
                double timeImg;
                {
                    boost::unique_lock<boost::mutex> imagelock(imageMutex);
                    imgEl = leftImgQ.front().first;
                    imgEr = rightImgQ.front().first;
                    timeImg = leftImgQ.front().second;
                    leftImgQ.pop_front();
                    rightImgQ.pop_front();
                    if(viconQ.size()>0){
                        v_pose = viconQ.front();
                        if(fullSystem->initialized)fullSystem->viconNow=fullSystem->viconFirst.inverse()*(*v_pose);
                        else fullSystem->viconNow=(*v_pose);
                        viconQ.pop_front();
                        delete v_pose;
                    }
                }
                if(timeImg>timeLast)
                {
                    timeLast = timeImg;
                    fullSystem->addActiveFrame(imgEl,imgEr, frameCount);
                }

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
        }

        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        fullSystem->blockUntilMappingIsFinished();
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

    rosthread.join();
    runthread.join();

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
    delete reader;

    printf("EXIT NOW!\n");
    ros::shutdown();

    google::ShutdownGoogleLogging();
    return 0;

}
