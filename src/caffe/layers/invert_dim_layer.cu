#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/invert_dim_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void invert_dim(const int count, const int invert_dim,
    const int outer_dim, const int inner_dim,
    const Dtype* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, count) {

    int n = index % inner_dim;
    int i = index / inner_dim % invert_dim;
    int o = index / outer_dim;
    //int output_index = o * outer_dim + (invert_dim - 1 - i) * inner_dim + n;
    //printf("Index %d, o %d, i %d, n %d, output index %d\n", index, o, i, n, output_index);

    output[o * outer_dim + (invert_dim - 1- i) * inner_dim + n] = input[index];
  }
}

template <typename Dtype>
void InvertDimLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();

  //printf("count %d, invert_dim %d, outer_dim %d, inner_dim %d\n", count, invert_dim_, outer_dim_, inner_dim_);
  invert_dim<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, invert_dim_, outer_dim_, inner_dim_, bottom_data, top_data);
}

template <typename Dtype>
void InvertDimLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  int count = bottom[0]->count();

  invert_dim<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, invert_dim_, outer_dim_, inner_dim_, top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(InvertDimLayer);


}  // namespace caffe
