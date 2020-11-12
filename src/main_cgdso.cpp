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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

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


#include "util/ImageAndExposure.h"



#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


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
bool needLoadMap = false;
float baseline = 0.12f;
int mode=0;

bool firstRosSpin=false;
FullSystem* fullSystem = 0;

using namespace dso;
//#define TTY_PATH            "/dev/tty"
//#define STTY_US             "stty raw -echo -F "
//#define STTY_DEF            "stty -raw echo -F "
//static int get_char()
//{
//    fd_set rfds;
//    struct timeval tv;
//    int ch = 0;

//    FD_ZERO(&rfds);
//    FD_SET(0, &rfds);
//    tv.tv_sec = 0;
//    tv.tv_usec = 10; //设置等待超时时间

//    //检测键盘是否有输入
//    if (select(1, &rfds, NULL, NULL, &tv) > 0)
//    {
//        ch = getchar();
//    }

//    return ch;
//}

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



int main( int argc, char** argv )
{
    //setlocale(LC_ALL, "");
    for(int i=1; i<argc;i++)
        parseArgument(argv[i]);

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);


    ImageFolderReader* reader = new ImageFolderReader(source,calib, gammaCalib, vignette);
    reader->setGlobalCalibration(baseline);



    if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
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
        if(lstart >= reader->getNumImages())
            lstart = reader->getNumImages()-1;
        lend = start;
        linc = -1;
    }

    fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed==0);



    IOWrap::PangolinDSOViewer* viewer = 0;
    IOWrap::PangolinDSOViewer* recordViewer = 0;
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

#define TEST_DSO 1
#if TEST_DSO==1
    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }


        std::vector<ImageAndExposure<float>*> preloadedImages;
        if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;

        bool stopped=false;
        for(int ii=0;ii<(int)idsToPlay.size(); ii++)
        {
            if(!stopped){
                if(!fullSystem->initialized)	// if not initialized: reset start time.
                {
                    gettimeofday(&tv_start, NULL);
                    started = clock();
                    sInitializerOffset = timesToPlayAt[ii];
                }

                int i = idsToPlay[ii];


                ImageAndExposure<float>* img;
                ImageAndExposure<float>* imgr=0;
                if(preload)
                    img = preloadedImages[ii];
                else
                    img = reader->getImage(i);



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
                    fullSystem->addActiveFrame(img,imgr, i);
                }

                MinimalImageF Mimg(wG[0], hG[0], img->image);
                IOWrap::displayImage("frameToTrack", &Mimg);


                delete img;
                delete imgr;

                if(fullSystem->initFailed || setting_fullResetRequested)
                {
                    if(ii < 250 || setting_fullResetRequested)
                    {
                        printf("RESETTING!\n");

                        std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                        delete fullSystem;

                        for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                        fullSystem = new FullSystem();
                        fullSystem->setGammaFunction(reader->getPhotometricGamma());
                        fullSystem->linearizeOperation = (playbackSpeed==0);
                        fullSystem->isLoadMap = needLoadMap;
                        if(fullSystem->isLoadMap)fullSystem->loadMap();

                        fullSystem->outputWrapper = wraps;

                        setting_fullResetRequested=false;
                    }
                }
                if(fullSystem->isLost)
                {
                    printf("LOST!!\n");
                    break;
                }
            }
            else{
                --ii;
            }
            bool quit=false;
            int ch=IOWrap::waitKey(1);
            if (ch)
            {
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
            }
            if(quit){
                viewer->close();
                recordViewer->close();
                break;
            }

        }
        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        //save map
        fullSystem->saveMap();

        fullSystem->printResult("result.txt");


        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });
#else
    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }

   typedef dso::Vec3f imageType ;
  #define COLORIMG 1
        std::vector<ImageAndExposure<imageType>*> preloadedImages;
        if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
#if COLORIMG==1
                preloadedImages.push_back(reader->getColorImage(i));
#else
                preloadedImages.push_back(reader->getImage(i));
#endif
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;


        for(int ii=0;ii<(int)idsToPlay.size();ii++)
        {
            if(!fullSystem->initialized)	// if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];



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
            if(fullSystem->initFailed || setting_fullResetRequested)
            {
                if(ii < 250 || setting_fullResetRequested)
                {
                    printf("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                    fullSystem = new FullSystem();
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed==0);


                    fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested=false;
                }
            }

            if(fullSystem->isLost)
            {
                    printf("LOST!!\n");
                    break;
            }

        }

        ImageAndExposure<imageType>* img_first;
        if(preload)
            img_first = preloadedImages[5];
        else{
#if (COLORIMG==1)
            img_first = reader->getColorImage(idsToPlay[5]);
#else
        img_first = reader->getImage(idsToPlay[i]);
#endif
        }
        std::vector<FrameShell*> allFrameHistory;
         CoarseInitializer* coarseInitializer;
         coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
         // =========================== add into allFrameHistory =========================
         FrameHessian* fh_first = new FrameHessian();
         FrameShell* shell_first = new FrameShell();
         shell_first->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
         shell_first->aff_g2l = AffLight(0,0);
         shell_first->marginalizedAt = shell_first->id = allFrameHistory.size();
         shell_first->timestamp = img_first->timestamp;
         shell_first->incoming_id = 0;
         fh_first->shell = shell_first;
         allFrameHistory.push_back(shell_first);

         // =========================== make Images / derivatives etc. =========================
         fh_first->ab_exposure = img_first->exposure_time;
         fh_first->makeImages(img_first->image, &(fullSystem->Hcalib));
         coarseInitializer->setFirst(&(fullSystem->Hcalib), fh_first);

         int tmpSeq[7]={1,2,16,18,21,24,25};
        for(int i=0;i<7;++i)
        {
            ImageAndExposure<imageType>* img;
            if(preload)
                img = preloadedImages[tmpSeq[i]];
            else{
    #if (COLORIMG==1)
                img = reader->getColorImage(idsToPlay[tmpSeq[i]]);
    #else
            img = reader->getImage(idsToPlay[i]);
    #endif
            }
            // =========================== add into allFrameHistory =========================
            FrameHessian* fh = new FrameHessian();
            FrameShell* shell = new FrameShell();
            shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
            shell->aff_g2l = AffLight(0,0);
            shell->marginalizedAt = shell->id = allFrameHistory.size();
            shell->timestamp = img->timestamp;
            shell->incoming_id = 0;
            fh->shell = shell;
            allFrameHistory.push_back(shell);

            // =========================== make Images / derivatives etc. =========================
            fh->ab_exposure = img->exposure_time;
            fh->makeImages(img->image, &(fullSystem->Hcalib));
            coarseInitializer->trackFrame(fh, fullSystem->outputWrapper);
            coarseInitializer->setFirst(&(fullSystem->Hcalib), fh);
            std::vector<Pnt> Vpnt;
            Vpnt=coarseInitializer->InvDepthVec[i];

            delete img;
        }

        delete img_first;
        delete coarseInitializer;
        allFrameHistory.clear();

        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        fullSystem->printResult("result.txt");


        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }
        while(1){
            cv::waitKey(33);
        }

    });
#endif


    if(viewer != 0)
        viewer->run();



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


    return 0;
}
