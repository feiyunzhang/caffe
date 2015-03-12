//
//  convert_leveldb_to_lmdb.cpp
//
//  Created by Kai Kang on 12/3/15.
//
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>

#include <iostream>
#include <sys/stat.h>

using std::string;


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a leveldb to lmdb.\n"
                          "Usage:\n"
                          "    convert_leveldb_to_lmdb INDB SAVEDB\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_leveldb_to_lmdb");
    return 1;
  }

  const char* leveldb_path = argv[1];
  const char* lmdb_path = argv[2];

  // Open leveldb
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // leveldb
  leveldb::DB* in_db;
  leveldb::Iterator* in_iter;
  leveldb::Options options;
  options.create_if_missing = false;

  LOG(INFO) << "Opening leveldb " << leveldb_path;
  leveldb::Status status = leveldb::DB::Open(options, leveldb_path, &in_db);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << leveldb_path << std::endl << status.ToString();
  in_iter = in_db->NewIterator(leveldb::ReadOptions());

  LOG(INFO) << "Opening lmdb " << lmdb_path;
  CHECK_EQ(mkdir(lmdb_path, 0744), 0) << "mkdir " << lmdb_path << " failed.";
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
  << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env, lmdb_path, 0, 0664), MDB_SUCCESS)
  << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
  << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
  << "mdb_open failed. Does the lmdb already exist?";

  // Copy dataset
  int count = 0;
  for (in_iter->SeekToFirst(); in_iter->Valid(); in_iter->Next()) {
    string value = in_iter->value().ToString();
    string keystr = in_iter->key().ToString();
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    mdb_key.mv_size = keystr.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
    << "mdb_put failed";

    if (++count % 1000 == 0) {
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
      << "mdb_txn_commit failed";
      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    LOG(INFO) << "Processed " << count << " files.";
  }
  assert(in_iter->status().ok());  // Check for any errors found during the scan
  mdb_close(mdb_env, mdb_dbi);
  mdb_env_close(mdb_env);
  delete in_iter;
  return 0;
}

