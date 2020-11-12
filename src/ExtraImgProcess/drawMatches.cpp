#include "drawMatches.h"

const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

static inline void _drawKeypoint( Mat& img, const cv::Point2f& p, const Scalar& color, int flags )
{
    CV_Assert( !img.empty() );
    Point center( cvRound(p.x * draw_multiplier), cvRound(p.y * draw_multiplier) );

    if( flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS )
    {
        int radius = cvRound(2/2 * draw_multiplier); // KeyPoint::size is a diameter

        // draw the circles around keypoints with the keypoints size
        circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );

    }
    else
    {
        // draw center with R=3
        int radius = 3 * draw_multiplier;
        circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
    }
}

void drawKeypoints( const Mat& image, const std::vector<cv::Point2f>& points, Mat& outImage,
                    const Scalar& _color, int flags )
{
    if( !(flags & DrawMatchesFlags::DRAW_OVER_OUTIMG) )
    {
        if( image.type() == CV_8UC3 )
        {
            image.copyTo( outImage );
        }
        else if( image.type() == CV_8UC1 )
        {
            cvtColor( image, outImage, CV_GRAY2BGR );
        }
        else
        {
            CV_Error( CV_StsBadArg, "Incorrect type of input image.\n" );
        }
    }

    RNG& rng=theRNG();
    bool isRandColor = _color == Scalar::all(-1);

    CV_Assert( !outImage.empty() );
    for( int i=0; i<points.size(); ++i )
    {
        Scalar color = isRandColor ? Scalar(rng(256), rng(256), rng(256)) : _color;
        _drawKeypoint( outImage, points[i], color, flags );
    }
}

static void _prepareImgAndDrawKeypoints( const Mat& img1, const std::vector<cv::Point2f>& keypoints1,
                                         const Mat& img2, const std::vector<cv::Point2f>& keypoints2,
                                         Mat& outImg, Mat& outImg1, Mat& outImg2,
                                         const Scalar& singlePointColor, int flags )
{
    Size size( img1.cols + img2.cols, MAX(img1.rows, img2.rows) );
    if( flags & DrawMatchesFlags::DRAW_OVER_OUTIMG )
    {
        if( size.width > outImg.cols || size.height > outImg.rows )
            CV_Error( CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
    }
    else
    {
        outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
        outImg = Scalar::all(0);
        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );

        if( img1.type() == CV_8U )
            cvtColor( img1, outImg1, CV_GRAY2BGR );
        else
            img1.copyTo( outImg1 );

        if( img2.type() == CV_8U )
            cvtColor( img2, outImg2, CV_GRAY2BGR );
        else
            img2.copyTo( outImg2 );
    }

    // draw keypoints
    if( !(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) )
    {
        Mat _outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
        drawKeypoints( _outImg1, keypoints1, _outImg1, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );

        Mat _outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
        drawKeypoints( _outImg2, keypoints2, _outImg2, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );
    }
}

static inline void _drawMatch( Mat& outImg, Mat& outImg1, Mat& outImg2 ,
                          const cv::Point2f& kp1, const cv::Point2f& kp2, const Scalar& matchColor, int flags )
{
    RNG& rng = theRNG();
    bool isRandMatchColor = matchColor == Scalar::all(-1);
    Scalar color = isRandMatchColor ? Scalar( rng(256), rng(256), rng(256) ) : matchColor;

    _drawKeypoint( outImg1, kp1, color, flags );
    _drawKeypoint( outImg2, kp2, color, flags );

    Point2f dpt2 = Point2f( std::min(kp2.x+outImg1.cols, float(outImg.cols-1)), kp2.y );

    line( outImg,
          Point(cvRound(kp1.x*draw_multiplier), cvRound(kp1.y*draw_multiplier)),
          Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
          color, 1, CV_AA, draw_shift_bits );
}

void drawMatches( const Mat& img1, const std::vector<cv::Point2f>& keypoints1,
                  const Mat& img2, const std::vector<cv::Point2f>& keypoints2,
                   Mat& outImg,
                  const Scalar& matchColor, const Scalar& singlePointColor, int flags )
{

    Mat outImg1, outImg2;
    _prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2,
                                 outImg, outImg1, outImg2, singlePointColor, flags );

    // draw matches
    for( size_t m = 0; m < keypoints1.size(); m++ )
    {
            _drawMatch( outImg, outImg1, outImg2, keypoints1[m], keypoints2[m], matchColor, flags );
    }
}

void draw_point(Mat& img, Point2f fp, Scalar color)
{
    circle(img, fp, 2, color, CV_FILLED, CV_AA, 0);
}

// Draw delaunay triangles
void draw_delaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{

    std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    std::vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}

//Draw voronoi diagram
void draw_voronoi(Mat& img, Subdiv2D& subdiv)
{
    std::vector<std::vector<Point2f> > facets;
    std::vector<Point2f> centers;
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

    std::vector<Point> ifacet;
    std::vector<std::vector<Point> > ifacets(1);

    for (size_t i = 0; i < facets.size(); i++)
    {
        ifacet.resize(facets[i].size());
        for (size_t j = 0; j < facets[i].size(); j++)
            ifacet[j] = facets[i][j];

        Scalar color;
        color[0] = std::rand() & 255;
        color[1] = std::rand() & 255;
        color[2] = std::rand() & 255;
        fillConvexPoly(img, ifacet, color, 8, 0);

        ifacets[0] = ifacet;
        polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);
        circle(img, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);
    }

}
void locate_point( Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color )
{
    int e0=0, vertex=0;

    subdiv.locate(fp, e0, vertex);

    if( e0 > 0 )
    {
        int e = e0;
        do
        {
            Point2f org, dst;
            if( subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0 )
                line( img, org, dst, active_color, 3, CV_AA, 0 );

            e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_LEFT);
        }
        while( e != e0 );
    }

    draw_point( img, fp, active_color );
}
