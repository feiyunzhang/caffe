#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
MapDataLayer<Dtype>::~MapDataLayer<Dtype>(){
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template<typename Dtype>
TransformationParameter MapDataLayer<Dtype>::label_trans_param(
      const TransformationParameter& trans_param){
  // Initialize label_transformer and set scale to 1
  // and clear mean file
  int crop_size = trans_param.crop_size();
  bool mirror = trans_param.mirror();

  TransformationParameter label_transform_param;
  label_transform_param.set_scale(1);
  label_transform_param.set_crop_size(crop_size);
  label_transform_param.set_mirror(mirror);
  return label_transform_param;
}

template <typename Dtype>
void MapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first" << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }

  // Read a data point and use it to initialize the top blob.
  BlobProtoVector maps;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    maps.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    maps.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
  CHECK(maps.blobs_size() == 2) << "MapDataLayer accepts BlobProtoVector with"
                                << " 2 BlobProtos: data and label.";
  BlobProto dataMap = maps.blobs(0);
  BlobProto labelMap = maps.blobs(1);

  // do not support mirror and crop for the moment
  int crop_size = this->layer_param_.transform_param().crop_size();
  bool mirror = this->layer_param_.transform_param().mirror();
  CHECK(crop_size == 0) << "MapDataLayer does not support cropping.";
  CHECK(!mirror) << "MapDataLayer does not support mirroring";

  // reshape data map
  (*top)[0]->Reshape(
      this->layer_param_.data_param().batch_size(), dataMap.channels(),
      dataMap.height(), dataMap.width());
  this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
      dataMap.channels(), dataMap.height(), dataMap.width());
  // reshape label map
  (*top)[1]->Reshape(
      this->layer_param_.data_param().batch_size(), labelMap.channels(),
      labelMap.height(), labelMap.width());
  this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
      labelMap.channels(), labelMap.height(), labelMap.width());
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // data map size
  this->datum_channels_ = dataMap.channels();
  this->datum_height_ = dataMap.height();
  this->datum_width_ = dataMap.width();
  this->datum_size_ = dataMap.channels() * dataMap.height() * dataMap.width();
  int label_size = labelMap.channels() * labelMap.height() * labelMap.width();
  label_mean_ = new Dtype[label_size]();
}

Datum BlobProto2Datum(const BlobProto& blob){
  Datum datum;
  datum.set_channels(blob.channels());
  datum.set_height(blob.height());
  datum.set_width(blob.width());
  datum.mutable_float_data()->CopyFrom(blob.data());
  return datum;
}

template<typename Dtype>
void MapDataLayer<Dtype>::InternalThreadEntry() {
  BlobProtoVector maps;
  Datum dataMap, labelMap;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      maps.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      maps.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Data transformer only accepts Datum
    dataMap = BlobProto2Datum(maps.blobs(0));
    labelMap = BlobProto2Datum(maps.blobs(1));

    // Apply data and label transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, dataMap, this->mean_, top_data);
    this->label_transformer_.Transform(item_id, labelMap, this->label_mean_, top_label);

    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

INSTANTIATE_CLASS(MapDataLayer);

} // caffe