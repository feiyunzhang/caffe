#include <algorithm>
#include <vector>

#include "caffe/layers/invert_dim_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InvertDimLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  invert_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.invert_param().axis());
  invert_dim_ = bottom[0]->shape(invert_axis_);
  outer_dim_ = bottom[0]->count(invert_axis_);
  inner_dim_ = bottom[0]->count(invert_axis_ + 1);
}

template <typename Dtype>
void InvertDimLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void InvertDimLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(InvertDimLayer);
#endif

INSTANTIATE_CLASS(InvertDimLayer);
REGISTER_LAYER_CLASS(InvertDim);

}  // namespace caffe
