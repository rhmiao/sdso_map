#include "imageMatch.h"

#include "ExtraImgProcess/decompHomgraphy.h"
#include "ExtraImgProcess/drawMatches.h"
#include "ExtraImgProcess/ASiftDetector.h"
#include <opencv2/core/eigen.hpp>

void ImageMatch::crossCheckMatching( const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                        std::vector<cv::DMatch>& filteredMatches12, int knn)
{
   filteredMatches12.clear();
   //std::cout<<"rows1"<<descriptors1.rows<<std::endl;
  // std::cout<<"rows2"<<descriptors2.rows<<std::endl;
   std::vector<std::vector<cv::DMatch> > matches12, matches21;
   descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn);
  //ratioTest(matches12);
   descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
   //ratioTest(matches21);
   for( size_t m = 0; m < matches12.size(); m++ )
   {
       bool findCrossCheck = false;
       for( size_t fk = 0; fk < matches12[m].size(); fk++ )
       {
           cv::DMatch forward = matches12[m][fk];

           for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
           {
               cv::DMatch backward = matches21[forward.trainIdx][bk];
               if( backward.trainIdx == forward.queryIdx )
               {
                   filteredMatches12.push_back(forward);
                   findCrossCheck = true;
                   break;
               }
           }
           if( findCrossCheck ) break;
       }
   }
   //std::cout<<"result"<<filteredMatches12.size()<<std::endl;
}

int ImageMatch::ratioTest(std::vector<std::vector<cv::DMatch> > &matches )
{
 float ratio = 0.67;
 int removed=0;
 // for all matches
 for (std::vector<std::vector<cv::DMatch> >::iterator  matchIterator= matches.begin();
      matchIterator!= matches.end(); ++matchIterator)
 {
   // if 2 NN has been identified
   if (matchIterator->size() > 1)
   {/*std::cout<<"ration"<<(*matchIterator)[0].distance/
               (*matchIterator)[1].distance<<std::endl;*/
     // check distance ratio
     if ((*matchIterator)[0].distance/
       (*matchIterator)[1].distance > ratio) {
         matchIterator->clear(); // remove match
         removed++;
     }
   } else { // does not have 2 neighbours
     matchIterator->clear(); // remove match
     removed++;
   }
 }
 return removed;
}

cv::Mat ImageMatch::match(const cv::Mat& descriptors1, const cv::Mat& descriptors2,std::vector<cv::KeyPoint>& KeyPointsRef,std::vector<cv::KeyPoint>& KeyPointsNew,
           std::vector<cv::KeyPoint>& pointsRefFix ,std::vector<cv::KeyPoint>& pointsNewFix )
{
    if(descriptors1.type()!=descriptors2.type())
        return cv::Mat::eye(3,3,CV_32F);
    crossCheckMatching( descriptors1, descriptors2,filteredMatches);

    std::vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs[i] = filteredMatches[i].queryIdx;
        trainIdxs[i] = filteredMatches[i].trainIdx;
    }

    cv::Mat mask;
    std::vector<cv::Point2f> pointsRef; cv::KeyPoint::convert(KeyPointsRef, pointsRef, queryIdxs);
    std::vector<cv::Point2f> pointsNew; cv::KeyPoint::convert(KeyPointsNew, pointsNew, trainIdxs);
    cv::Mat H12 = cv::findHomography( cv::Mat(pointsRef), cv::Mat(pointsNew), mask,CV_RANSAC, 3 );
    //cv::Mat F12 = cv::findFundamentalMat(  cv::Mat(pointsRef), cv::Mat(pointsNew), mask,CV_RANSAC, 3 );
   pointsRefFix.clear();
   pointsNewFix.clear();
    for(int i=0;i<pointsRef.size();i++)
        if(mask.at<uchar>(i,0)==1)
        {
            pointsRefFix.push_back(KeyPointsRef[queryIdxs[i]]);
            pointsNewFix.push_back(KeyPointsNew[trainIdxs[i]]);
        }
    return H12;
}

cv::Mat ImageMatch::match(const cv::Mat& descriptors1, const cv::Mat& descriptors2,std::vector<cv::KeyPoint>& KeyPointsRef,std::vector<cv::KeyPoint>& KeyPointsNew,
           std::vector<cv::Point2f>& pointsRefFix ,std::vector<cv::Point2f>& pointsNewFix )
{
    if(descriptors1.type()!=descriptors2.type())
        return cv::Mat::eye(3,3,CV_32F);
    crossCheckMatching( descriptors1, descriptors2,filteredMatches);

    std::vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs[i] = filteredMatches[i].queryIdx;
        trainIdxs[i] = filteredMatches[i].trainIdx;
    }

    cv::Mat mask;
    std::vector<cv::Point2f> pointsRef; cv::KeyPoint::convert(KeyPointsRef, pointsRef, queryIdxs);
    std::vector<cv::Point2f> pointsNew; cv::KeyPoint::convert(KeyPointsNew, pointsNew, trainIdxs);
    cv::Mat H12 = cv::findHomography( cv::Mat(pointsRef), cv::Mat(pointsNew), mask,CV_RANSAC, 3 );
    //cv::Mat F12 = cv::findFundamentalMat(  cv::Mat(pointsRef), cv::Mat(pointsNew), mask,CV_RANSAC, 3 );
   pointsRefFix.clear();
   pointsNewFix.clear();
    for(int i=0;i<pointsRef.size();i++)
        if(mask.at<uchar>(i,0)==1)
        {
            pointsRefFix.push_back(pointsRef[i]);
            pointsNewFix.push_back(pointsNew[i]);
        }
    return H12;
}

cv::Mat ImageMatch::match(const cv::Mat& descriptors1, const cv::Mat& descriptors2,std::vector<cv::KeyPoint>& KeyPointsRef,std::vector<cv::KeyPoint>& KeyPointsNew)
{
    if(descriptors1.type()!=descriptors2.type())
        return cv::Mat::eye(3,3,CV_32F);
    crossCheckMatching( descriptors1, descriptors2,filteredMatches);

    std::vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs[i] = filteredMatches[i].queryIdx;
        trainIdxs[i] = filteredMatches[i].trainIdx;
    }

    cv::Mat mask;
    std::vector<cv::Point2f> pointsRef; cv::KeyPoint::convert(KeyPointsRef, pointsRef, queryIdxs);
    std::vector<cv::Point2f> pointsNew; cv::KeyPoint::convert(KeyPointsNew, pointsNew, trainIdxs);
    cv::Mat H12;
    if(pointsRef.size()<4)H12 = cv::Mat::eye(3,3,CV_32F);
    else H12 = cv::findHomography( cv::Mat(pointsRef), cv::Mat(pointsNew), mask,CV_RANSAC, 3 );
    //cv::Mat F12 = cv::findFundamentalMat(  cv::Mat(pointsRef), cv::Mat(pointsNew), mask,CV_RANSAC, 3 );

    return H12;
}

bool ImageMatch::matchForLoopclose(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2,int wl,int hl,int lvl,std::vector<std::vector<float>> idepthPairVec)
{
     for(int i=0;i<idepthPairVec.size();++i)idepthPairVec[i].clear();
     idepthPairVec.clear();
     Eigen::Vector3f* colorRef = Frame1->dIp[lvl];
     Eigen::Vector3f* colorNew = Frame2->dIp[lvl];
     cv::Mat imgRef=cv::Mat(hl,wl,CV_8U);
     cv::Mat imgNew=cv::Mat(hl,wl,CV_8U);
     for(int i=0;i<hl;++i)
         for(int j=0;j<wl;++j){
             imgRef.at<uchar>(i,j)=(uchar)(colorRef[i*wl+j][0]);
             imgNew.at<uchar>(i,j)=(uchar)(colorNew[i*wl+j][0]);
         }

    std::vector<cv::KeyPoint> KeyPointsRef(Frame1->keyPoints);
    std::vector<cv::Point2f> points1,points2;
    for(cv::KeyPoint kpt: KeyPointsRef)points1.push_back(kpt.pt);

    cv::Mat H=match(Frame1->descriptors, Frame2->descriptors,Frame1->keyPoints,Frame2->keyPoints);
//    cv::Mat F12 = cv::findFundamentalMat(  cv::Mat(pointsRef), cv::Mat(pointsNew), mask2,CV_RANSAC, 3 );
    cv::Mat opticalStatus,opticalErr;

    cv::perspectiveTransform(points1,points2,H);
    cv::calcOpticalFlowPyrLK(imgRef,imgNew,points1,points2,opticalStatus,opticalErr,Size(11,11),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),1);

    std::vector<uchar> inliersMask(points1.size());

    Mat F=findFundamentalMat(points1, points2,FM_RANSAC,1,0.99,inliersMask);

    std::vector<cv::Point3f> pointsRef3d;
    std::vector<cv::Point2f> pointsRefFix0;
    std::vector<cv::Point2f> pointsNewFix0;
    std::vector<cv::Point2f> pointsRefFix;
    std::vector<cv::Point2f> pointsNewFix;
    std::vector<cv::KeyPoint> keyPointsRefFix;

   int countMtach=0;
    for(int i=0;i<points1.size();i++){
        if(opticalStatus.at<uchar>(i,0)&&opticalErr.at<float>(i,0)<100&&inliersMask[i]){
            pointsRefFix0.push_back(points1[i]);
            pointsNewFix0.push_back(points2[i]);
            keyPointsRefFix.push_back(KeyPointsRef[i]);
            cv::Point3f pt3;
            pt3.z=1.0/(KeyPointsRef[i].response+0.001);
            pt3.x=(points1[i].x-dso::cxG[lvl])*dso::fxiG[lvl]*pt3.z;
            pt3.y=(points1[i].y-dso::cyG[lvl])*dso::fyiG[lvl]*pt3.z;
            pointsRef3d.push_back(pt3);
            countMtach+=1;
//            std::cout<<"depth:"<<KeyPointsRef[queryIdxs[i]].pt.x<<" "<<KeyPointsRef[queryIdxs[i]].pt.y<<" "<<pt3.x<<" "<<pt3.y<<" "<<pt3.z<<std::endl;
        }
    }

    cv::Mat intrinsicl=cv::Mat::zeros(3,3,CV_32F);
    cv::Mat nodistortionl=cv::Mat::zeros(1,5,CV_32F);
    cv::Mat PnPinliers;

    intrinsicl.at<float>(0,0) = dso::fxG[lvl];
    intrinsicl.at<float>(1,1)= dso::fyG[lvl];
    intrinsicl.at<float>(0,2)= dso::cxG[lvl];
    intrinsicl.at<float>(1,2)= dso::cyG[lvl];
    intrinsicl.at<float>(2,2)= 1;

    if(pointsNewFix0.size()<5)return false;
#if CV_VERSION_MAJOR == 3
    cv::solvePnPRansac(pointsRef3d,pointsNewFix0,intrinsicl,nodistortionl,rvec,tvec,false, 100, 1, 0.99, PnPinliers);
#else
    cv::solvePnPRansac(pointsRef3d,pointsNewFix0,intrinsicl,nodistortionl,rvec,tvec,false, 100, 1, 2*pointsNewFix0.size()/3, PnPinliers);
#endif


    bool isTestSuccess;
    if(pointsNewFix0.size()>100)isTestSuccess = PnPinliers.rows>50;
    else isTestSuccess = PnPinliers.rows >=0.6*pointsNewFix0.size();

    if(PnPinliers.rows>0){
        for(int i=0;i<PnPinliers.rows;i++){
            pointsRefFix.push_back(pointsRefFix0[PnPinliers.at<uchar>(i,0)]);
            pointsNewFix.push_back(pointsNewFix0[PnPinliers.at<uchar>(i,0)]);
        }

        cv::Mat output2;
        drawMatches(  imgRef, pointsRefFix0,imgNew, pointsNewFix0,
                      output2, Scalar::all(-1), Scalar::all(-1),
                      DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::imshow("Loopclose homo Matches",output2);

        cv::Mat output;
        drawMatches(  imgRef, pointsRefFix,imgNew, pointsNewFix,
                      output, Scalar::all(-1), Scalar::all(-1),
                      DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        cv::imshow("Loopclose Matches",output);
    }

    return isTestSuccess;
}

void  ImageMatch:: calcFeature(dso::FrameHessian* Frame,int wl,int hl,int lvl)
{
    Eigen::Vector3f* colorRef = Frame->dIp[lvl];

    std::vector<cv::KeyPoint> KeyPointsRef,KeyPointsRefGFTT;

    cv::Mat recordDescriptorsRef;
    //sift
    cv::Mat imgRef=cv::Mat(hl,wl,CV_8U);
    for(int i=0;i<hl;++i)
        for(int j=0;j<wl;++j){
            imgRef.at<uchar>(i,j)=(uchar)(colorRef[i*wl+j][0]);
        }

//    detector->detect(imgRef,KeyPointsRef);
    detectorGFTT->detect(imgRef,KeyPointsRefGFTT);
    KeyPointsRef.insert(KeyPointsRef.end(),KeyPointsRefGFTT.begin(),KeyPointsRefGFTT.end());
    computeDescriptors->compute( imgRef, KeyPointsRef, recordDescriptorsRef );

    (Frame->keyPoints).assign(KeyPointsRef.begin(),KeyPointsRef.end());
    recordDescriptorsRef.copyTo(Frame->descriptors);
    Frame->descriptorIsOK=true;
}

void ImageMatch::calcHWithFeature(dso::FrameHessian* firstFrame, dso::FrameHessian* newFrame,int wl,int hl,int lvl,dso::Mat33 K_Eigen, dso::SE3 &refToNew,bool doMatch)
{
    int64 start=0,end=0;
    start = cv::getTickCount();
    Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
    Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

    std::vector<cv::KeyPoint> KeyPointsRef,KeyPointsNew;
    std::vector<cv::KeyPoint> KeyPointsRefGFTT,KeyPointsNewGFTT;
    cv::Mat recordDescriptorsRef,recordDescriptorsNew;
    //sift
    cv::Mat imgRef=cv::Mat(hl,wl,CV_8U);
    cv::Mat imgNew=cv::Mat(hl,wl,CV_8U);
    for(int i=0;i<hl;++i)
        for(int j=0;j<wl;++j){
            imgNew.at<uchar>(i,j)=(uchar)(colorNew[i*wl+j][0]);
            imgRef.at<uchar>(i,j)=(uchar)(colorRef[i*wl+j][0]);
        }
    if(!(firstFrame->descriptorIsOK))
    {
    detector->detect(imgRef,KeyPointsRef);
    detectorGFTT->detect(imgRef,KeyPointsRefGFTT);
    KeyPointsRef.insert(KeyPointsRef.end(),KeyPointsRefGFTT.begin(),KeyPointsRefGFTT.end());
    computeDescriptors->compute( imgRef, KeyPointsRef, recordDescriptorsRef );

    (firstFrame->keyPoints).assign(KeyPointsRef.begin(),KeyPointsRef.end());
    recordDescriptorsRef.copyTo(firstFrame->descriptors);
    firstFrame->descriptorIsOK=true;
    }
    else
    {
        KeyPointsRef.assign((firstFrame->keyPoints).begin(),(firstFrame->keyPoints).end());
        firstFrame->descriptors.copyTo(recordDescriptorsRef);
    }


    if(!(newFrame->descriptorIsOK))
    {
    detector->detect(imgNew,KeyPointsNew);
    detectorGFTT->detect(imgNew,KeyPointsNewGFTT);
    KeyPointsNew.insert(KeyPointsNew.end(),KeyPointsNewGFTT.begin(),KeyPointsNewGFTT.end());
    computeDescriptors->compute( imgNew, KeyPointsNew, recordDescriptorsNew );

    (newFrame->keyPoints).assign(KeyPointsNew.begin(),KeyPointsNew.end());
    recordDescriptorsNew.copyTo(newFrame->descriptors);
    newFrame->descriptorIsOK=true;
    }
    else
    {
        KeyPointsNew.assign((newFrame->keyPoints).begin(),(newFrame->keyPoints).end());
        newFrame->descriptors.copyTo(recordDescriptorsNew);
    }

    if(doMatch)
    {
        std::vector<cv::Point2f> pointsRefFix,pointsNewFix;
        cv::Mat H12 =match( recordDescriptorsRef, recordDescriptorsNew,
                            KeyPointsRef,KeyPointsNew,pointsRefFix,pointsNewFix);

        cv::Mat imgRefH;
        cv::warpPerspective(imgRef, imgRefH,H12,Size(imgRef.cols,imgRef.rows));

        cv::Mat output;
        std::cout << "The pointsRef: " << pointsRefFix.size()<< std::endl;
        drawMatches(  imgRef, pointsRefFix,imgNew, pointsNewFix,
                      output, Scalar::all(-1), Scalar::all(-1),
                      DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        imshow("Matches after correction",output);
        std::cout<<"H"<<H12<<std::endl;
        cv::imshow("3rd level image",imgNew);
        cv::imshow("3rd level image h",imgRefH);

        std::vector<cv::Mat> R,t,normal;
        cv::Mat R21,t21,K,Kf;
        float parallax=0;
        cv::eigen2cv(K_Eigen, K);
        K.convertTo(Kf,CV_32F);

#if CV_VERSION_MAJOR == 3
        cv::decomposeHomographyMat(H12, K,R,t,normal);
#else
        decomposeHomographyMat(H12, K,R,t,normal);
#endif


        int bestGood = 0;
        int secondBestGood = 0;
        int bestSolutionIdx = -1;

        for(int i=0;i<R.size();++i){
            cv::Mat Rf,tf;
            R[i].convertTo(Rf,CV_32F);
            t[i].convertTo(tf,CV_32F);
            if(cv::norm (tf)>1e-6)tf=tf/cv::norm (tf);
            std::cout<<"Rf"<<Rf<<std::endl;
            std::cout<<"tf"<<tf<<std::endl;
            int nGood = CheckRT(Rf,tf,pointsRefFix,pointsNewFix,Kf, 4.0,parallax);
            if(Rf.at<float>(2,2)<-0.5f)nGood=0;
            std::cout<<"nGood:"<<nGood<<std::endl;
            // 保留最优的和次优的
            if(nGood>bestGood)
            {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
            }
            else if(nGood>0.9*secondBestGood)
            {
                secondBestGood = nGood;
            }
        }

        if(bestSolutionIdx!=-1){
            R[bestSolutionIdx].copyTo(R21);
            t[bestSolutionIdx].copyTo(t21);

            if(cv::norm (t21)>1e-6)t21=0.1*t21/cv::norm (t21);
            dso::Mat33 R_eigen;cv::cv2eigen(R21, R_eigen);
            dso::Vec3 T_eigen;cv::cv2eigen(t21, T_eigen);
            std::cout<<"R21"<<R21<<std::endl;
            std::cout<<"t21"<<t21<<std::endl;
            //T_eigen.setZero(3,1);

            refToNew.setRotationMatrix(R_eigen);
            //refToNew=dso::SE3(R_eigen, T_eigen);
        }
    }
    end = getTickCount();
    std::cout << "The differences: " << 1000.0*(end - start)/getTickFrequency()<<" ms"<< std::endl;
}

int ImageMatch::trackingWithOFStereo(dso::FrameHessian* Frame1,int wl,int hl,int lvl,
                                      std::vector<cv::KeyPoint> &KeyPointsRef,std::vector<cv::Point3f> &KeyPoints3d,
                                      float &idepthScale,cv::Mat idepthMat,int idpthflag)
{
    int64 start=0,end=0;

    Eigen::Vector3f* colorRef = Frame1->dIp[lvl];
    Eigen::Vector3f* colorNew= Frame1->dIr[lvl];
//    std::vector<cv::Point2f> idepthPointsVec;

    cv::Mat imgRef=cv::Mat(hl,wl,CV_8U);
    cv::Mat imgNew=cv::Mat(hl,wl,CV_8U);
    for(int i=0;i<hl;++i)
        for(int j=0;j<wl;++j){
            imgRef.at<uchar>(i,j)=(uchar)(colorRef[i*wl+j][0]);
            imgNew.at<uchar>(i,j)=(uchar)(colorNew[i*wl+j][0]);
//            if(idepthMat.rows>0)
//            {
//                float idepthPre=idepthMat.at<float>(i,j);
//                if(idepthPre > dso::ibfG[0]*2 && idepthPre<5 ){
//                    cv::Point2f pt(i,j);idepthPointsVec.push_back(pt);
//                }
//            }
        }
    std::vector<cv::KeyPoint> KeyPointsRefTmp,KeyPointsRefGFTT;

    if(lvl==dso::featuresLvl) KeyPointsRefTmp.assign(Frame1->keyPoints.begin(), Frame1->keyPoints.end());
    else
    {
        detector->detect(imgRef,KeyPointsRefTmp);
        detectorGFTT->detect(imgRef,KeyPointsRefGFTT);
        KeyPointsRefTmp.insert(KeyPointsRefTmp.end(),KeyPointsRefGFTT.begin(),KeyPointsRefGFTT.end());
    }
    start = cv::getTickCount();
    cv::Mat opticalStatus,opticalErr;
    std::vector<cv::Point2f> points1,points2,recordPoints1,recordPoints2;

    points1.reserve(KeyPointsRefTmp.size());
    points2.reserve(KeyPointsRefTmp.size());
    recordPoints1.reserve(KeyPointsRefTmp.size());
    recordPoints2.reserve(KeyPointsRefTmp.size());

    for(cv::KeyPoint kpt: KeyPointsRefTmp)points1.push_back(kpt.pt);

    //use RANSAC fundamental to get inliersMask and remove outlier
    std::vector<uchar> inliersMask(points1.size());
    for(int i=0;i<points1.size();++i){
        cv::Point2f pt=points1[i];
        pt.x=pt.x+dso::bfG[lvl]*0.5f;
        points2.push_back(pt);
    }
    cv::calcOpticalFlowPyrLK(imgRef,imgNew,points1,points2,opticalStatus,opticalErr,Size(11,11),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),1);
    if(points1.size()<15)return 0;
    Mat F=findFundamentalMat(points1, points2,FM_RANSAC,1,0.99,inliersMask);
//    findHomography(points1, points2,inliersMask,FM_RANSAC,1);

    KeyPointsRef.clear();
    KeyPoints3d.clear();
    depth.setTo(0);
    std::vector<cv::Point2f> IntensePoints;
    std::vector<cv::Point2f> relativeScales;
    for(int i=0;i<points1.size();++i)
    {
        if(lvl==dso::featuresLvl)Frame1->keyPoints[i].response=-1;
        if(opticalStatus.at<uchar>(i,0)&&opticalErr.at<float>(i,0)<100&&inliersMask[i]
                &&points1[i].y>0&&points1[i].x>0&&points2[i].x>0&&points2[i].y>0
                &&points1[i].y<imgRef.rows&&points1[i].x<imgRef.cols
                &&points2[i].y<imgRef.rows&&points2[i].x<imgRef.cols){
            recordPoints1.push_back(points1[i]);
            recordPoints2.push_back(points2[i]);
            KeyPointsRef.push_back(KeyPointsRefTmp[i]);
            cv::Point2f intPt(imgRef.at<uchar>(int(points1[i].y),int(points1[i].x)),imgNew.at<uchar>(int(points2[i].y),int(points2[i].x)));
            IntensePoints.push_back(intPt);
            cv::Point3f pt3;
            pt3.z=dso::bfG[lvl]/(points2[i].x-points1[i].x);
            pt3.x=(points1[i].x-dso::cxG[lvl])*dso::fxiG[lvl]*pt3.z;
            pt3.y=(points1[i].y-dso::cyG[lvl])*dso::fyiG[lvl]*pt3.z;
            if(lvl==dso::featuresLvl){
                Frame1->keyPoints[i].response=1/pt3.z;
            }
            if(idpthflag&&pt3.z>0&&pt3.z<fabs(dso::bfG[0])/5&&idepthMat.at<float>(points1[i])>fabs(dso::ibfG[0])*5&&idepthMat.at<float>(points1[i])<10){
                cv::Point2f spt2f;
                spt2f.x=relativeScales.size()*1000;
                spt2f.y=pt3.z*idepthMat.at<float>(points1[i]);
                relativeScales.push_back(spt2f);
            }
            for(int j=-2;j<2;++j)
                for(int k=-2;k<2;++k)
                    if(points1[i].y+j>0&&points1[i].x+k>0&&
                       points1[i].y+j<depth.rows&&points1[i].x+k<depth.cols)depth.at<float>(int(points1[i].y)+j,int(points1[i].x)+k) = pt3.z;
            KeyPoints3d.push_back(pt3);
        }
    }
    if(KeyPointsRef.size()<10)return 0;
    //-------affine--------
    cv::Vec4f line;
    if(IntensePoints.size()>20){
        cv::fitLine(IntensePoints,line,CV_DIST_HUBER,0,0.01,0.01);
        if(fabs(line[1]/line[0]-1)<0.1){
        Frame1->leftToright_affine[0]=line[1]/line[0];
        Frame1->leftToright_affine[1]=line[3]-Frame1->leftToright_affine[0]*line[2];
        }
        else{
            Frame1->leftToright_affine[0]=1;
            Frame1->leftToright_affine[1]=0;
        }

    }
    if(idpthflag&&relativeScales.size()>10){
        std::vector<bool> inlierFlag;
        float a,b,c;
        fitLineRANSAC(relativeScales, a, b, c, inlierFlag);
        if(fabs(a)<0.1)idepthScale=fabs(b/c);
        else idepthScale=0;
    }
    //-------Delaunay--------
    cv::Size size=imgRef.size();
    cv::Rect rect(0, 0, size.width, size.height);
    subdiv.initDelaunay(rect);
    cv::Mat delaunay=cv::Mat(size,CV_32FC3,Scalar(0,0,0));

    subdiv.insert(recordPoints1);
    Scalar delaunay_color(255, 255, 255), points_color(255, 0, 0);

    draw_delaunay(delaunay, subdiv, delaunay_color);
    cv::imshow("delaunay", delaunay);
    end = cv::getTickCount();
    std::cout << "The differences of optcal flow stereo: " << 1000.0*(end - start)/cv::getTickFrequency()<<" ms"<< std::endl;
//    cv::Mat output;
//    drawMatches(  imgRef, recordPoints1,imgNew, recordPoints2,
//                  output, Scalar::all(-1), Scalar::all(-1),
//                  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

//    cv::imshow("optical flow stereo",output);

    return 1;
}

dso::SE3 ImageMatch::trackingWithOF(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2,int wl,int hl,int lvl,
                                    std::vector<cv::KeyPoint> &KeyPointsRef,std::vector<cv::Point3f> &KeyPoints3d,Eigen::Vector2f &relAff,dso::SE3 init_pose,int &flag)
{
    int64 start=0,end=0;
    start = cv::getTickCount();
    Eigen::Vector3f* colorRef = Frame1->dIp[lvl];
    Eigen::Vector3f* colorNew = Frame2->dIp[lvl];
    cv::Mat imgRef=cv::Mat(hl,wl,CV_8U);
    cv::Mat imgNew=cv::Mat(hl,wl,CV_8U);
    isBlur=false;
    for(int i=0;i<hl;++i)
        for(int j=0;j<wl;++j){
            imgRef.at<uchar>(i,j)=(uchar)(colorRef[i*wl+j][0]);
            imgNew.at<uchar>(i,j)=(uchar)(colorNew[i*wl+j][0]);
        }

//    cv::Laplacian(imgNew,lap,3);
//    cv::Mat lap_mean, lap_stddev;

//    cv::meanStdDev(lap,lap_mean,lap_stddev);
//    std::cout<<"lap_stddev:"<<lap_stddev.at<double>(0,0)<<std::endl;
//    if(lap_stddev.at<double>(0,0)<5){
//        isBlur=true;
//        return dso::SE3();
//    }

    if(KeyPointsRef.size()<10){
        flag=0;
        return dso::SE3();
    }
//    cv::Mat H=match(Frame1->descriptors, Frame2->descriptors,Frame1->keyPoints,Frame2->keyPoints);

    cv::Mat opticalStatus,opticalErr;
    std::vector<cv::Point2f> points1,points2,recordPoints1,recordPoints2;
    for(cv::KeyPoint kpt: KeyPointsRef)points1.push_back(kpt.pt);
    dso::Mat33f KR = (dso::KG[lvl]*init_pose.rotationMatrix().cast<float>());
    dso::Vec3f Kt = (dso::KG[lvl]*init_pose.translation().cast<float>());
    for(cv::Point3f pt3f: KeyPoints3d)
    {
        dso::Vec3f vpt3f(pt3f.x,pt3f.y,pt3f.z);
        dso::Vec3f pt=KR*vpt3f+Kt;
        cv::Point2f pt2f(pt[0] / pt[2],pt[1] / pt[2]);
        points2.push_back(pt2f);
    }
//    cv::perspectiveTransform(points1,points2,H);
    cv::calcOpticalFlowPyrLK(imgRef,imgNew,points1,points2,opticalStatus,opticalErr,Size(5,5),3,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),OPTFLOW_USE_INITIAL_FLOW);

    //use RANSAC fundamental to get inliersMask and remove outlier
    std::vector<uchar> inliersMask(points1.size());
    if(points1.size()<10){
        flag=0;
        return dso::SE3();
    }
//    Mat F=findFundamentalMat(points1, points2,FM_RANSAC,1,0.99,inliersMask);
    findHomography(points1, points2,inliersMask,FM_RANSAC,1);

    std::vector<cv::Point3f> KeyPoints3dRecord;
    std::vector<cv::Point2f> IntensePoints;
    for(int i=0;i<points1.size();++i)
        if(opticalStatus.at<uchar>(i,0)&&opticalErr.at<float>(i,0)<100&&inliersMask[i]
                &&points1[i].y>0&&points1[i].x>0&&points2[i].x>0&&points2[i].y>0
                &&points1[i].y<imgRef.rows&&points1[i].x<imgRef.cols
                &&points2[i].y<imgRef.rows&&points2[i].x<imgRef.cols){
            recordPoints1.push_back(points1[i]);
            recordPoints2.push_back(points2[i]);
            KeyPoints3dRecord.push_back(KeyPoints3d[i]);
            cv::Point2f intPt(imgRef.at<uchar>(int(points1[i].y),int(points1[i].x)),imgNew.at<uchar>(int(points2[i].y),int(points2[i].x)));
            IntensePoints.push_back(intPt);
        }

    if(KeyPoints3dRecord.size()<10){
        flag=0;
        return dso::SE3();
    }
//    cv::Vec4f line;
//    if(IntensePoints.size()>20){
//        cv::fitLine(IntensePoints,line,CV_DIST_HUBER,0,0.01,0.01);
//        relAff[0]=line[1]/line[0];
//        relAff[1]=line[3]-relAff[0]*line[2];
//    }
    cv::Mat PnPinliers,R,R_vector,T;
    cv::Mat intrinsicl=cv::Mat::zeros(3,3,CV_32F);
    cv::Mat nodistortionl=cv::Mat::zeros(1,5,CV_32F);

    intrinsicl.at<float>(0,0) = dso::fxG[lvl];
    intrinsicl.at<float>(1,1)= dso::fyG[lvl];
    intrinsicl.at<float>(0,2)= dso::cxG[lvl];
    intrinsicl.at<float>(1,2)= dso::cyG[lvl];
    intrinsicl.at<float>(2,2)= 1;

#if CV_VERSION_MAJOR == 3
    cv::solvePnPRansac(KeyPoints3dRecord,recordPoints2,intrinsicl,nodistortionl,R_vector,T,false, 100, 2, 0.99, PnPinliers);
#else
    cv::solvePnPRansac(KeyPoints3dRecord,recordPoints2,intrinsicl,nodistortionl,R_vector,T,false, 200, 1, recordPoints2.size()/3, PnPinliers);
#endif

    cv::Rodrigues(R_vector,R);

    Eigen::Matrix3f R_eigen;
    Eigen::Vector3f T_eigen;
    cv::cv2eigen(R,R_eigen);
    cv::cv2eigen(T,T_eigen);

    dso::SE3 trackingSe3(R_eigen.cast<double>(),T_eigen.cast<double>());
//    std::cout<<"isTestSuccess"<<PnPinliers.rows<<" "<<recordPoints2.size()<<std::endl;

    bool isTestSuccess;
    if(recordPoints2.size()>200)isTestSuccess = PnPinliers.rows>100;
    else isTestSuccess = PnPinliers.rows >=0.4*recordPoints2.size();

    end = cv::getTickCount();
    LOG(INFO) << "The differences of optcal flow: " << 1000.0*(end - start)/cv::getTickFrequency()<<" ms"<< std::endl;
    cv::Mat output;
    drawMatches(  imgRef, recordPoints1,imgNew, recordPoints2,
                  output, Scalar::all(-1), Scalar::all(-1),
                  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imshow("optical flow tracking",output);

    flag = 1;
    if(!isTestSuccess){
        flag = 0;
        std::cout<<"PnPinliers are too few!"<<std::endl;
        return dso::SE3();
    }
    else return trackingSe3;

}

void ImageMatch::getTriangleVetexInDelaunay(cv::Subdiv2D& subdiv, cv::Point2f fp,std::vector<cv::Point2f> &output)
{
    int e0=0, vertex=0;
        output.clear();
        int status=subdiv.locate(fp, e0, vertex);

        if(status==Subdiv2D::PTLOC_VERTEX){
            cv::Point2f org=subdiv.getVertex(vertex);
            output.push_back(org);
        }
        else if(status==Subdiv2D::PTLOC_ON_EDGE){
            cv::Point2f org,dst;
            if(subdiv.edgeOrg(e0, &org) > 0)output.push_back(org);
            if(org.x*org.y==0||org.x<0||org.y<0){
                output.clear();
                return;
            }
            if(subdiv.edgeDst(e0, &dst) > 0 )output.push_back(dst);
            if(dst.x*dst.y==0||dst.x<0||dst.y<0){
                output.clear();
                return;
            }
        }
        else if(status==Subdiv2D::PTLOC_INSIDE&& e0 > 0 )
        {
            int e = e0;
                cv::Point2f org,dst;
                if(subdiv.edgeOrg(e, &org) > 0)output.push_back(org);
                if(org.x*org.y==0||org.x<0||org.y<0){
                    output.clear();
                    return;
                }
                if(subdiv.edgeDst(e, &dst) > 0 )output.push_back(dst);
                if(dst.x*dst.y==0||dst.x<0||dst.y<0){
                    output.clear();
                    return;
                }
                e = subdiv.getEdge(e0, Subdiv2D::NEXT_AROUND_LEFT);
                if( e != e0){
                    if(subdiv.edgeDst(e, &dst) > 0 ){
                        if(dst!=output[0]&&dst!=output[1])output.push_back(dst);
                        else {
                            subdiv.edgeOrg(e, &dst);
                            output.push_back(dst);
                        }
                    }
                    if(dst.x*dst.y==0||dst.x<0||dst.y<0){
                        output.clear();
                        return;
                    }
                }
        }

}

float ImageMatch::getInterpDepthInDelaunay(Eigen::Vector3f target,std::vector<Eigen::Vector3f> triangle)
{
    float depth;
    for(int i=0;i<triangle.size();i++){
        if(triangle[i][2]<0||triangle[i][2]>1000){
            return 0;
        }
    }
    if(triangle.size()==3){
        Eigen::Vector3f vec1 = triangle[2] - triangle[0];
        Eigen::Vector3f vec2 = triangle[1] - triangle[0];

        Eigen::Vector3f vec3 = vec1.cross(vec2);

        if(vec1[2]==0.0f&&vec2[2]==0.0f)depth=triangle[2][2];
        else depth = (vec3.dot(triangle[0])-vec3.head(2).dot(target.head(2)))/vec3[2];
    }
    else if(triangle.size()==2){
        depth=0.5*(triangle[0][2]+triangle[1][2]);
    }
    else if(triangle.size()==1){
        depth=triangle[0][2];
    }

    return depth;

}

void ImageMatch::getScaleDepthFromStereo(dso::FrameHessian* frame,int wl,int hl,int lvl)
{
    Eigen::Vector3f* colorLeft = frame->dIp[lvl];
    Eigen::Vector3f* colorRight= frame->dIr[lvl];

    cv::Mat imgLeft=cv::Mat(hl,wl,CV_8U);
    cv::Mat imgRight=cv::Mat(hl,wl,CV_8U);
    for(int i=0;i<hl;++i)
        for(int j=0;j<wl;++j){
            imgLeft.at<uchar>(i,j)=(uchar)(colorLeft[i*wl+j][0]);
            imgRight.at<uchar>(i,j)=(uchar)(colorRight[i*wl+j][0]);
        }

    cv::Mat opticalStatus,opticalErr;
    std::vector<cv::Point2f> points1,points2,points1Record,points2Record;

    int scale=1<<lvl;
    for(dso::PointHessian* ph:frame->pointHessians){
        cv::Point2f pt2f(ph->u/scale,ph->v/scale);
        points1.push_back(pt2f);
    }
    imgLeft.convertTo(imgLeft, CV_8U,frame->leftToright_affine[0],frame->leftToright_affine[1]);

    //use RANSAC fundamental to get inliersMask and remove outlier
//    std::vector<uchar> inliersMask(points1.size());

//leftToright_affine<<1,20;
    for(int i=0;i<points1.size();++i){
        cv::Point2f pt=points1[i];
        dso::PointHessian* ph=frame->pointHessians[i];
        pt.x=traceOnRight(ph, lvl,frame->leftToright_affine);
        pt.y=ph->v/scale;
        points2.push_back(pt);
    }
   cv::calcOpticalFlowPyrLK(imgLeft,imgRight,points1,points2,opticalStatus,opticalErr,Size(11,11),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),1);

    for(int i=0;i<points1.size();++i){
        if(opticalStatus.at<uchar>(i,0)&&opticalErr.at<float>(i,0)<100)
        {
        cv::Point2f ptl=points1[i];
        cv::Point2f ptr=points2[i];
        points1Record.push_back(ptl);
        points2Record.push_back(ptr);
        dso::PointHessian* ph=frame->pointHessians[i];
        //ph->setIdepth((ptr.x-ptl.x)*dso::ibfG[lvl]);
        //std::cout<<(ptr.x-ptl.x)*dso::ibfG[lvl]<<" "<<ph->idepth<<" "<<frame->leftToright_affine<<std::endl;
        }

    }
    cv::Mat output;
    drawMatches(  imgLeft, points1,imgRight, points2,
                  output, Scalar::all(-1), Scalar::all(-1),
                  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imshow("getScaleDepthFromStereo",output);


}
//根据点集拟合直线ax+by+c=0，res为残差
void ImageMatch::calcLinePara(std::vector<cv::Point2f> pts, float &a, float &b, float &c, float &res)
{
    res = 0;
    Vec4f line;
    std::vector<cv::Point2f> ptsF;
    for (unsigned int i = 0; i < pts.size(); i++)
        ptsF.push_back(pts[i]);

    cv::fitLine(ptsF, line, CV_DIST_L2, 0, 1e-2, 1e-2);
    a = line[1];
    b = -line[0];
    c = line[0] * line[3] - line[1] * line[2];

    for (unsigned int i = 0; i < pts.size(); i++)
    {
        float resid_ = fabs(pts[i].x * a + pts[i].y * b + c);
        res += resid_;
    }
    res /= pts.size();
}

//得到直线拟合样本，即在直线采样点集上随机选2个点
bool ImageMatch::getSample(std::vector<int> set, std::vector<int> &sset)
{
    int i[2];
    if (set.size() > 2)
    {
        do
        {
            for (int n = 0; n < 2; n++)
                i[n] = int((float)rand() / (float)RAND_MAX * (set.size() - 1));
        } while (!(i[1] != i[0]));
        for (int n = 0; n < 2; n++)
        {
            sset.push_back(i[n]);
        }
    }
    else
    {
        return false;
    }
    return true;
}

//RANSAC直线拟合
void ImageMatch::fitLineRANSAC(std::vector<cv::Point2f> ptSet, float &a, float &b, float &c, std::vector<bool> &inlierFlag)
{
    float residual_error = 0.3; //内点阈值

    bool stop_loop = false;
    int maximum = 0;  //最大内点数

    //最终内点标识及其残差
    inlierFlag = std::vector<bool>(ptSet.size(), false);
    std::vector<float> resids_(ptSet.size(), 3);
    int sample_count = 0;
    int N = 500;

    float res = 0;

    // RANSAC
    srand((unsigned int)time(NULL)); //设置随机数种子
    std::vector<int> ptsID;
    for (unsigned int i = 0; i < ptSet.size(); i++)
        ptsID.push_back(i);
    while (N > sample_count && !stop_loop)
    {
        std::vector<bool> inlierstemp;
        std::vector<float> residualstemp;
        std::vector<int> ptss;
        int inlier_count = 0;
        if (!getSample(ptsID, ptss))
        {
            stop_loop = true;
            continue;
        }

        std::vector<cv::Point2f> pt_sam;
        pt_sam.push_back(ptSet[ptss[0]]);
        pt_sam.push_back(ptSet[ptss[1]]);

        // 计算直线方程
        calcLinePara(pt_sam, a, b, c, res);
        //内点检验
        for (unsigned int i = 0; i < ptSet.size(); i++)
        {
            cv::Point2f pt = ptSet[i];
            float resid_ = fabs(pt.x * a + pt.y * b + c);
            residualstemp.push_back(resid_);
            inlierstemp.push_back(false);
            if (resid_ < residual_error)
            {
                ++inlier_count;
                inlierstemp[i] = true;
            }
        }
        // 找到最佳拟合直线
        if (inlier_count >= maximum)
        {
            maximum = inlier_count;
            resids_ = residualstemp;
            inlierFlag = inlierstemp;
        }
        // 更新RANSAC迭代次数，以及内点概率
        if (inlier_count == 0)
        {
            N = 500;
        }
        else
        {
            float epsilon = 1.0 - float(inlier_count) / (float)ptSet.size(); //野值点比例
            float p = 0.99; //所有样本中存在1个好样本的概率
            float s = 2.0;
            N = int(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
        }
        ++sample_count;
    }

    //利用所有内点重新拟合直线
    std::vector<cv::Point2f> pset;
    for (unsigned int i = 0; i < ptSet.size(); i++)
    {
        if (inlierFlag[i])
            pset.push_back(ptSet[i]);
    }

    calcLinePara(pset, a, b, c, res);
}

float ImageMatch::traceOnRight(dso::PointHessian* ph,int lvl,dso::Vec2f hostToFrame_affine)
{
    dso::FrameHessian* frame = ph->host;
    // KRKi
    //    Mat33f KRKi = Mat33f::Identity().cast<float>();
    // Kt
    // T between stereo cameras;
    //    Kt = K*bl;
    //    KtStereoG
    // to simplify set aff 1, 0
    int scale=1<<lvl;
    dso::Vec3f hostToFrame_Kt= dso::KtStereoG[lvl];
    dso::Vec3f pr =  dso::Vec3f(ph->u/scale,ph->v/scale, 1);

    // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
    float dx = 1;
    float dy = 0;

    float errorInPixel = 0.2f + 2;

    int radius=int(fabs(dso::bfG[lvl]*0.3f));

    if(errorInPixel >10) errorInPixel=10;

    int numSteps = 2*radius;
    dso::Mat22f Rplane = dso::Mat22f::Identity().cast<float>();

    float ptx = std::max(5.0f,ph->u/scale+dso::bfG[lvl]*ph->idepth-radius*dx);
    float pty = ph->v/scale;


    dso::Vec2f rotatetPattern[MAX_RES_PER_POINT];
    for(int idx=0;idx<patternNum;idx++)
        rotatetPattern[idx] = Rplane * dso::Vec2f(dso::patternP[idx][0], dso::patternP[idx][1]);

    float errors[100];
    float bestU=0, bestV=0, bestEnergy=1e10;
    int bestIdx=-1;
    if(numSteps >= 100) numSteps = 99;

    float hitColor0=0,phcolor0=0,delcolor=0;
    for(int i=0;i<numSteps;i++)
    {
        float energy=0;
        for(int idx=0;idx<patternNum;idx++)
        {
            float hitColor = dso::getInterpolatedElement31(frame->dIr[lvl],
                    (float)(ptx+rotatetPattern[idx][0]),
                    (float)(pty+rotatetPattern[idx][1]),
                    dso::wG[lvl]);
            if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
            float phcolor=(float)(hostToFrame_affine[0] * ph->color[idx] + hostToFrame_affine[1]);

            if(idx!=0){
//                if(fabs((hitColor-phcolor+delcolor))>20)energy +=5000;
            }
            else{
                hitColor0=hitColor;
                phcolor0=phcolor;
                delcolor=phcolor0-hitColor0;
            }

            float residual = hitColor  - phcolor+delcolor;
            float hw = fabs(residual) < dso::setting_huberTH ? 1 : dso::setting_huberTH / fabs(residual);
            energy += hw *residual*residual*(2-hw);
        }


//        std::cout<<i<<" "<<energy<<std::endl;
        errors[i] = energy;
        if(energy < bestEnergy)
        {
            bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
        }

        ptx+=dx;
        pty+=dy;
        if(ptx < 5 || pty < 5 || ptx > dso::wG[lvl]-6 || pty > dso::hG[lvl]-6)break;
    }


    // find best score outside a +-2px radius.
    float secondBest=1e10;
    for(int i=0;i<numSteps;i++)
    {
        if((i < bestIdx-dso::setting_minTraceTestRadius || i > bestIdx+dso::setting_minTraceTestRadius) && errors[i] < secondBest)
            secondBest = errors[i];
    }

    // ============== do GN optimization ===================
    float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
    if(dso::setting_trace_GNIterations>0) bestEnergy = 1e5;
    int gnStepsGood=0, gnStepsBad=0;
    for(int it=0;it<dso::setting_trace_GNIterations;it++)
    {
        float H = 1, b=0, energy=0;
        for(int idx=0;idx<patternNum;idx++)
        {
            dso::Vec3f hitColor = dso::getInterpolatedElement33(frame->dIr[lvl],
                    (float)(bestU+rotatetPattern[idx][0]),
                    (float)(bestV+rotatetPattern[idx][1]),dso::wG[lvl]);

            if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}
            float residual = hitColor[0] - (hostToFrame_affine[0] * ph->color[idx] + hostToFrame_affine[1]);
            float dResdDist = dx*hitColor[1] + dy*hitColor[2];
            float hw = fabs(residual) < dso::setting_huberTH ? 1 : dso::setting_huberTH / fabs(residual);

            H += hw*dResdDist*dResdDist;
            b += hw*residual*dResdDist;
            energy += (ph->weights[idx])*(ph->weights[idx])*hw *residual*residual*(2-hw);
        }


        if(energy > bestEnergy)
        {
            gnStepsBad++;

            // do a smaller step from old point.
            stepBack*=0.5;
            bestU = uBak + stepBack*dx;
            bestV = vBak + stepBack*dy;
        }
        else
        {
            gnStepsGood++;

            float step = -gnstepsize*b/H;
            if(step < -0.5) step = -0.5;
            else if(step > 0.5) step=0.5;

            if(!std::isfinite(step)) step=0;

            uBak=bestU;
            vBak=bestV;
            stepBack=step;

            bestU += step*dx;
            bestV += step*dy;
            bestEnergy = energy;

        }

        if(fabsf(stepBack) < dso::setting_trace_GNThreshold) break;
    }


    float idepth_min,idepth_max;
    // ============== set new interval ===================
    if(dx*dx>dy*dy)
    {
        idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
        idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
    }
    else
    {
        idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
        idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
    }
    if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);


    return bestU;
}

void ImageMatch::calcCornersInSelectPts(dso::FrameHessian* Frame,std::vector<dso::ImmaturePoint*> phs,int lvl)
{
    int MIN_DIST=2;
    int hl=dso::hG[lvl],wl=dso::wG[lvl];
    cv::Mat mask = cv::Mat(hl, wl, CV_8UC1, cv::Scalar(0));
    int countUnMarked=0;
    for(dso::ImmaturePoint* ph:phs){
        if(ph->isCorner==-1){
            countUnMarked++;
            cv::Point cvPt(int(ph->u),int(ph->v));
            cv::circle(mask, cvPt, MIN_DIST, 255, -1);
        }
    }
    if(countUnMarked>10){
        Eigen::Vector3f* color = Frame->dIp[lvl];

        cv::Mat img=cv::Mat(hl,wl,CV_8U);
        for(int i=0;i<hl;++i)
            for(int j=0;j<wl;++j){
                img.at<uchar>(i,j)=(uchar)(color[i*wl+j][0]);
            }
        std::vector<cv::Point2f> n_pts;
        cv::goodFeaturesToTrack(img, n_pts, int(phs.size()), 0.01, MIN_DIST, mask);
        cv::Mat cornerMask = cv::Mat(hl, wl, CV_8UC1, cv::Scalar(0));
        for(cv::Point2f n_pt:n_pts)cornerMask.at<uchar>(int(n_pt.y),int(n_pt.x))=255;
        for(dso::ImmaturePoint* ph:phs){
            if(cornerMask.at<uchar>(int(ph->v),int(ph->u))==255)ph->isCorner=1;
            else ph->isCorner=0;
        }
//        for(dso::ImmaturePoint* ph:phs)cv::circle( img, cv::Point2f(ph->u,ph->v), 1, cv::Scalar(255), 1, CV_AA );
//        for(cv::Point2f n_pt:n_pts)cv::circle( img, n_pt, 3, cv::Scalar(255), 1, CV_AA );
//        cv::imshow("corners",img);
    }

}

void ImageMatch::calcCornersInSelectPts(dso::FrameHessian* Frame,std::vector<dso::PointHessian*> phs,int lvl)
{
    int MIN_DIST=2;
    int hl=dso::hG[lvl],wl=dso::wG[lvl];
    cv::Mat mask = cv::Mat(hl, wl, CV_8UC1, cv::Scalar(0));
    int countUnMarked=0;
    for(dso::PointHessian* ph:phs){
        if(ph->isCorner==-1){
            countUnMarked++;
            cv::Point cvPt(int(ph->u),int(ph->v));
            cv::circle(mask, cvPt, MIN_DIST, 255, -1);
        }
    }
    if(countUnMarked>10){
        Eigen::Vector3f* color = Frame->dIp[lvl];

        cv::Mat img=cv::Mat(hl,wl,CV_8U);
        for(int i=0;i<hl;++i)
            for(int j=0;j<wl;++j){
                img.at<uchar>(i,j)=(uchar)(color[i*wl+j][0]);
            }
        std::vector<cv::Point2f> n_pts;
        cv::goodFeaturesToTrack(img, n_pts, int(phs.size()), 0.01, MIN_DIST, mask);
        cv::Mat cornerMask = cv::Mat(hl, wl, CV_8UC1, cv::Scalar(0));
        for(cv::Point2f n_pt:n_pts)cornerMask.at<uchar>(int(n_pt.y),int(n_pt.x))=255;
        for(dso::PointHessian* ph:phs){
            if(cornerMask.at<uchar>(int(ph->v),int(ph->u))==255)ph->isCorner=1;
            else ph->isCorner=0;
        }
//        for(dso::PointHessian* ph:phs)cv::circle( img, cv::Point2f(ph->u,ph->v), 1, cv::Scalar(255), 1, CV_AA );
//        for(cv::Point2f n_pt:n_pts)cv::circle( img, n_pt, 3, cv::Scalar(255), 1, CV_AA );
//        cv::imshow("corners",img);
    }

}
