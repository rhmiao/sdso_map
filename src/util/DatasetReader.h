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
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"

#include "IOWrapper/ImageDisplay.h"

#include <opencv2/highgui/highgui.hpp>

#include <string>
#if HAS_ZIPLIB
	#include "zip.h"
#endif

#include <boost/thread.hpp>

using namespace dso;



inline int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != "..")
    		files.push_back(name);
    }
    closedir(dp);


    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(unsigned int i=0;i<files.size();i++)
	{
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}


struct PrepImageItem
{
	int id;
	bool isQueud;
    ImageAndExposure<float>* pt;

	inline PrepImageItem(int _id)
	{
		id=_id;
		isQueud = false;
		pt=0;
	}

	inline void release()
	{
		if(pt!=0) delete pt;
		pt=0;
	}
};




class ImageFolderReader
{
public:
	ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		this->path = path;
		this->calibfile = calibFile;

#if HAS_ZIPLIB
		ziparchive=0;
		databuffer=0;
#endif

		isZipped = (path.length()>4 && path.substr(path.length()-4) == ".zip");





		if(isZipped)
		{
#if HAS_ZIPLIB
			int ziperror=0;
			ziparchive = zip_open(path.c_str(),  ZIP_RDONLY, &ziperror);
			if(ziperror!=0)
			{
				printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
				exit(1);
			}

			files.clear();
			int numEntries = zip_get_num_entries(ziparchive, 0);
			for(int k=0;k<numEntries;k++)
			{
				const char* name = zip_get_name(ziparchive, k,  ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if(nstr == "." || nstr == "..") continue;
				files.push_back(name);
			}

			printf("got %d entries and %d files!\n", numEntries, (int)files.size());
			std::sort(files.begin(), files.end());
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
		else
			getdir (path, files);


        undistort = Undistort<float>::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width=undistort->getSize()[0];
		height=undistort->getSize()[1];

                // load timestamps if possible.

                loadTimestamps();
                loadGroundtruth();
                printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());

	}
	~ImageFolderReader()
	{
#if HAS_ZIPLIB
		if(ziparchive!=0) zip_close(ziparchive);
		if(databuffer!=0) delete databuffer;
#endif


		delete undistort;
	};

	Eigen::VectorXf getOriginalCalib()
	{
		return undistort->getOriginalParameter().cast<float>();
	}
	Eigen::Vector2i getOriginalDimensions()
	{
		return  undistort->getOriginalSize();
	}

	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

        void setGlobalCalibration(float baseline)
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		getCalibMono(K, w_out, h_out);
		setGlobalCalib(w_out, h_out, K);
                Vec3f bl;
                bl << -baseline, 0, 0;
                for (int level = 0; level < pyrLevelsUsed; ++ level)
                {
                    bfG[level] = bl(0) *fxG[level];
                    ibfG[level] = 1.0/bfG[level];
                    KtStereoG[level] = KG[level]*bl;
                }

                STEREO_PRE_RTll = Mat33f::Identity();
                STEREO_PRE_tTll = bl;
                STEREO_distanceLL = fabs(baseline);

                STEREO_PRE_KRKiTll = Mat33f::Identity();
                STEREO_PRE_RKiTll =  K.inverse();
                STEREO_PRE_KtTll = K * STEREO_PRE_tTll;

	}
        void setGlobalCalibration(int w,int h,Eigen::Matrix3f K,float baseline)
        {
                setGlobalCalib(w, h, K);
                Vec3f bl;
                bl << -baseline, 0, 0;
                for (int level = 0; level < pyrLevelsUsed; ++ level)
                {
                    bfG[level] = bl(0) *fxG[level];
                    ibfG[level] = 1.0/bfG[level];
                    KtStereoG[level] = KG[level]*bl;
                }
                STEREO_PRE_RTll = Mat33f::Identity();
                STEREO_PRE_tTll = bl;
                STEREO_distanceLL = fabs(baseline);

                STEREO_PRE_KRKiTll = Mat33f::Identity();
                STEREO_PRE_RKiTll =  K.inverse();
                STEREO_PRE_KtTll = K * STEREO_PRE_tTll;

        }


	int getNumImages()
	{
		return files.size();
	}

	double getTimestamp(int id)
	{
		if(timestamps.size()==0) return id*0.1f;
		if(id >= (int)timestamps.size()) return 0;
		if(id < 0) return 0;
		return timestamps[id];
	}
        std::vector<double> getGroundtruth(int id)
        {
            if(groundtruth.size()==0) return std::vector<double>();
            if(id >= (int)groundtruth.size()) return std::vector<double>();
            if(id < 0) return std::vector<double>();
            return groundtruth[id];
        }


	void prepImage(int id, bool as8U=false)
	{

	}


	MinimalImageB* getImageRaw(int id)
	{
			return getImageRaw_internal(id,0);
	}

    ImageAndExposure<float>* getImage(int id, bool forceLoadDirectly=false)
	{
		return getImage_internal(id, 0);
	}

    ImageAndExposure<Vec3f>* getColorImage(int id, bool forceLoadDirectly=false)
    {
        return getColorImage_internal(id, 0);
    }


	inline float* getPhotometricGamma()
	{
		if(undistort==0 || undistort->photometricUndist==0) return 0;
		return undistort->photometricUndist->getG();
	}


	// undistorter. [0] always exists, [1-2] only when MT is enabled.
    Undistort<float>* undistort;
private:


	MinimalImageB* getImageRaw_internal(int id, int unused)
	{
		if(!isZipped)
		{
			// CHANGE FOR ZIP FILE
			return IOWrap::readImageBW_8U(files[id]);
		}
		else
		{
#if HAS_ZIPLIB
			if(databuffer==0) databuffer = new char[widthOrg*heightOrg*6+10000];
			zip_file_t* fle = zip_fopen(ziparchive, files[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*6+10000);

			if(readbytes > (long)widthOrg*heightOrg*6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,(long)widthOrg*heightOrg*6+10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg*heightOrg*30];
				fle = zip_fopen(ziparchive, files[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*30+10000);

				if(readbytes > (long)widthOrg*heightOrg*30)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,(long)widthOrg*heightOrg*30+10000);
					exit(1);
				}
			}

			return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
	}


    MinimalImageB3* getColorImageRaw_internal(int id, int unused)
    {
        if(!isZipped)
        {
            // CHANGE FOR ZIP FILE
            return IOWrap::readImageRGB_8U(files[id]);
        }
        else
        {
#if HAS_ZIPLIB
            if(databuffer==0) databuffer = new char[widthOrg*heightOrg*6+10000];
            zip_file_t* fle = zip_fopen(ziparchive, files[id].c_str(), 0);
            long readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*6+10000);

            if(readbytes > (long)widthOrg*heightOrg*6)
            {
                printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,(long)widthOrg*heightOrg*6+10000, files[id].c_str());
                delete[] databuffer;
                databuffer = new char[(long)widthOrg*heightOrg*30];
                fle = zip_fopen(ziparchive, files[id].c_str(), 0);
                readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*30+10000);

                if(readbytes > (long)widthOrg*heightOrg*30)
                {
                    printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,(long)widthOrg*heightOrg*30+10000);
                    exit(1);
                }
            }

            return IOWrap::readStreamRGB_8U(databuffer, readbytes);
#else
            printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            exit(1);
#endif
        }
    }


    ImageAndExposure<float>* getImage_internal(int id, int unused)
    {
		MinimalImageB* minimg = getImageRaw_internal(id, 0);
        ImageAndExposure<float>* ret2 = undistort->undistort(
				minimg,
				(exposures.size() == 0 ? 1.0f : exposures[id]),
				(timestamps.size() == 0 ? 0.0 : timestamps[id]));
        delete minimg;
		return ret2;
	}

    ImageAndExposure<Vec3f>* getColorImage_internal(int id, int unused)
    {
        MinimalImageB3* minimg = getColorImageRaw_internal(id, 0);
        ImageAndExposure<Vec3f>* ret2 = undistort->undistort3d(
                minimg,
                (exposures.size() == 0 ? 1.0f : exposures[id]),
                (timestamps.size() == 0 ? 0.0 : timestamps[id]));
        delete minimg;
        return ret2;
    }

	inline void loadTimestamps()
	{
		std::ifstream tr;
		std::string timesFile = path.substr(0,path.find_last_of('/')) + "/times.txt";
		tr.open(timesFile.c_str());
                std::cout<<"miao"<<std::endl;
		while(!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int id;
			double stamp;
			float exposure = 0;

			if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
                        else if(1 == sscanf(buf, "%lf", &stamp))
                        {
                            timestamps.push_back(stamp);
                        }
		}
		tr.close();

		// check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int)exposures.size()==(int)getNumImages()) ;
		for(int i=0;i<(int)exposures.size();i++)
		{
			if(exposures[i] == 0)
			{
				// fix!
				float sum=0,num=0;
				if(i>0 && exposures[i-1] > 0) {sum += exposures[i-1]; num++;}
				if(i+1<(int)exposures.size() && exposures[i+1] > 0) {sum += exposures[i+1]; num++;}

				if(num>0)
					exposures[i] = sum/num;
			}

			if(exposures[i] == 0) exposuresGood=false;
		}


		if((int)getNumImages() != (int)timestamps.size())
		{
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if((int)getNumImages() != (int)exposures.size() || !exposuresGood)
		{
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
	}
        inline void loadGroundtruth()
        {
            std::ifstream tr;
            std::string timesFile = path.substr(0,path.find_last_of('/')) + "/gt.txt";
            tr.open(timesFile.c_str());
            while(!tr.eof() && tr.good())
            {
                    std::string line;
                    char buf[1000];
                    tr.getline(buf, 1000);

                    std::vector<double> gtVec;
                    double gt[12];

                    if(12 == sscanf(buf, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &(gt[0]),&(gt[1]),&(gt[2]),&(gt[3]),&(gt[4]),&(gt[5])
                                    ,&(gt[6]),&(gt[7]),&(gt[8]),&(gt[9]),&(gt[10]),&(gt[11])))
                    {
                        for(int i=0;i<12;i++)gtVec.push_back(gt[i]);
                        groundtruth.push_back(gtVec);
                    }
            }
            tr.close();

            if((int)getNumImages() != (int)groundtruth.size())
            {
                    groundtruth.clear();
            }

            printf("got %d images and %d groundtruth.!\n", (int)getNumImages(), (int)groundtruth.size());
        }




    std::vector<ImageAndExposure<Vec3f>*> preloadedImages;
	std::vector<std::string> files;
	std::vector<double> timestamps;
	std::vector<float> exposures;
        std::vector<std::vector<double> > groundtruth;

	int width, height;
	int widthOrg, heightOrg;

	std::string path;
	std::string calibfile;

	bool isZipped;

#if HAS_ZIPLIB
	zip_t* ziparchive;
	char* databuffer;
#endif
};

