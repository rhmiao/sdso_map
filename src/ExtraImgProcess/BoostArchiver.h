/*
 * map save/load extension for DSO
 * This header contains boost headers needed by serialization
 *
 * object to save:
 *   - KeyFrame
 *   - KeyFrameDatabase
 *   - Map
 *   - MapPoint
 */
#ifndef BOOST_ARCHIVER_H
#define BOOST_ARCHIVER_H
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
// set serialization needed by KeyFrame::mspChildrens ...
#include <boost/serialization/map.hpp>
// map serialization needed by KeyFrame::mConnectedKeyFrameWeights ...
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/base_object.hpp>
// base object needed by DBoW3::BowVector and DBoW3::FeatureVector
#include <opencv2/core/core.hpp>
#if CV_VERSION_MAJOR == 3
#include <opencv2/features2d/features2d.hpp>
#else
#include <opencv2/nonfree/features2d.hpp>
#endif

#include "BowVector.h"
#include "FeatureVector.h"

#include "util/NumType.h"

BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)
namespace boost{
    namespace serialization {

    /* serialization for DBoW3 BowVector */
    template<class Archive>
    void serialize(Archive &ar, DBoW3::BowVector &BowVec, const unsigned int file_version)
    {
        ar & boost::serialization::base_object<DBoW3::BowVector>(BowVec);
    }
    /* serialization for DBoW3 FeatureVector */
    template<class Archive>
    void serialize(Archive &ar, DBoW3::FeatureVector &FeatVec, const unsigned int file_version)
    {
        ar & boost::serialization::base_object<DBoW3::FeatureVector>(FeatVec);
    }

    /* serialization for CV KeyPoint */
    template<class Archive>
    void serialize(Archive &ar, ::cv::KeyPoint &kf, const unsigned int file_version)
    {
        ar & kf.angle;
        ar & kf.class_id;
        ar & kf.octave;
        ar & kf.response;
        ar & kf.response;
        ar & kf.pt.x;
        ar & kf.pt.y;
    }
    /* serialization for CV Mat */
    template<class Archive>
    void save(Archive &ar, const ::cv::Mat &m, const unsigned int file_version)
    {
        cv::Mat m_ = m;
        if (!m.isContinuous())
            m_ = m.clone();
        size_t elem_size = m_.elemSize();
        size_t elem_type = m_.type();
        ar & m_.cols;
        ar & m_.rows;
        ar & elem_size;
        ar & elem_type;

        const size_t data_size = m_.cols * m_.rows * elem_size;

        ar & boost::serialization::make_array(m_.ptr(), data_size);
    }
    template<class Archive>
    void load(Archive & ar, ::cv::Mat& m, const unsigned int version)
    {
        int cols, rows;
        size_t elem_size, elem_type;

        ar & cols;
        ar & rows;
        ar & elem_size;
        ar & elem_type;

        m.create(rows, cols, elem_type);
        size_t data_size = m.cols * m.rows * elem_size;

        ar & boost::serialization::make_array(m.ptr(), data_size);
    }

    /* serialization for eigen Mat */
    template<  class Archive,  class S,  int Rows_,  int Cols_,  int Ops_,  int MaxRows_, int MaxCols_>
    inline void save(Archive & ar,  const Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & g,  const unsigned int version)
    {
        int rows = g.rows();
        int cols = g.cols();

        ar & rows;
        ar & cols;
        ar & boost::serialization::make_array(g.data(), rows * cols);
    }

    template<   class Archive,  class S, int Rows_,int Cols_,int Ops_, int MaxRows_,  int MaxCols_>
    inline void load(Archive & ar, Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & g, const unsigned int version)
    {
        int rows, cols;
        ar & rows;
        ar & cols;
        g.resize(rows, cols);
        ar & boost::serialization::make_array(g.data(), rows * cols);
    }

    template<   class Archive, class S,  int Rows_, int Cols_, int Ops_, int MaxRows_, int MaxCols_>
    inline void serialize( Archive & ar, Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & g, const unsigned int version)
    {
        split_free(ar, g, version);
    }


    }

}
// TODO: boost::iostream zlib compressed binary format
#endif // BOOST_ARCHIVER_H
