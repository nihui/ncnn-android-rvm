// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef RVM_H
#define RVM_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct InterFeatures
{
    ncnn::Mat r1;
    ncnn::Mat r2;
    ncnn::Mat r3;
    ncnn::Mat r4;
};

class RVM
{
public:
    RVM();
    ~RVM();

    int load(const char* parampath, const char* modelpath, bool use_gpu = false);
    int load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu = false);

    void set_model_type(int model_type);
    void set_target_size(int target_size);
    void set_intra_inter(int intra_inter);
    void set_postproc_mode(bool segmentation, bool refine_deep, bool refine_fast);

    int detect(const cv::Mat& rgb, InterFeatures& feats, cv::Mat& fgr, cv::Mat& pha, cv::Mat& seg);
    int draw(cv::Mat& rgb, const cv::Mat& fgr, const cv::Mat& pha, const cv::Mat& seg);

protected:
    ncnn::Net rvm;
    int model_type;
    int target_size;
    int intra_inter;
    bool segmentation;
    bool refine_deep;
    bool refine_fast;
};

#endif // RVM_H
