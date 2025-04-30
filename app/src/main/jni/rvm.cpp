// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

// ncnn model exported from https://github.com/PeterL1n/RobustVideoMatting
//
// import torch
// from torch import nn
// from model import MattingNetwork
// from model.fast_guided_filter import FastGuidedFilterRefiner
// from model.deep_guided_filter import DeepGuidedFilterRefiner
//
// class Model(nn.Module):
//     def __init__(self):
//         super().__init__()
//
//         self.rvm = MattingNetwork('mobilenetv3').eval()
//         self.rvm.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
//
//         self.refiner_deep = DeepGuidedFilterRefiner()
//         self.refiner_fast = FastGuidedFilterRefiner()
//
//     def forward_first_frame(self, src):
//         return self.rvm(src)
//
//     def forward(self, src, src_sm, r1, r2, r3, r4):
//
//         f1, f2, f3, f4 = self.rvm.backbone(src_sm)
//         f4 = self.rvm.aspp(f4)
//         hid, *rec = self.rvm.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
//
//         # downsample
//         fgr_residual, pha = self.rvm.project_mat(hid).split([3, 1], dim=-3)
//         fgr = fgr_residual + src_sm
//
//         # downsample + refiner_deep
//         fgr_residual_deep, pha_deep = self.refiner_deep(src, src_sm, fgr_residual, pha, hid)
//         fgr_deep = fgr_residual_deep + src
//
//         # downsample + refiner_fast
//         fgr_residual_fast, pha_fast = self.refiner_fast(src, src_sm, fgr_residual, pha, hid)
//         fgr_fast = fgr_residual_fast + src
//
//         # downsample + segmentation
//         seg = self.rvm.project_seg(hid)
//
//         return fgr, pha, fgr_deep, pha_deep, fgr_fast, pha_fast, seg, *rec
//
// import pnnx
//
// model = Model().eval()
//
// x = torch.rand(1, 3, 512, 512)
// x2 = torch.rand(1, 3, 256, 256)
// x2_hr = torch.rand(1, 3, 1024, 1024)
//
// # generate feats via forward_first_frame, with different shapes
// fgr, pha, r1, r2, r3, r4 = model.forward_first_frame(x)
// fgr2, pha2, r12, r22, r32, r42 = model.forward_first_frame(x2)
//
// # export with dynamic shape
// pnnx.export(model, "rvm_mobilenetv3.pt", (x, x, r1, r2, r3, r4), (x2_hr, x2, r12, r22, r32, r42))
//
// and then fix refiner_fast fp16 overflow issue in ncnn.param via appending 31=1 layer feat mask
//
// BinaryOp   div_58    2 1 401 399 402 0=3 31=1
//

#include "rvm.h"

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

RVM::RVM()
{
    model_type = 0;
    target_size = 512;
    intra_inter = 0;
    segmentation = false;
    refine_deep = true;
    refine_fast = false;
}

RVM::~RVM()
{
}

int RVM::load(const char* parampath, const char* modelpath, bool use_gpu)
{
    rvm.clear();

    // rvm.opt = ncnn::Option();

    rvm.opt.use_fp16_packed = false;
    rvm.opt.use_fp16_storage = false;
    rvm.opt.use_fp16_arithmetic = false;

#if NCNN_VULKAN
    rvm.opt.use_vulkan_compute = use_gpu;
#endif

    rvm.load_param(parampath);
    rvm.load_model(modelpath);

    return 0;
}

int RVM::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    rvm.clear();

    // rvm.opt = ncnn::Option();

    rvm.opt.use_fp16_packed = false;
    rvm.opt.use_fp16_storage = false;
    rvm.opt.use_fp16_arithmetic = false;

#if NCNN_VULKAN
    rvm.opt.use_vulkan_compute = use_gpu;
#endif

    rvm.load_param(mgr, parampath);
    rvm.load_model(mgr, modelpath);

    return 0;
}

void RVM::set_model_type(int _model_type)
{
    model_type = _model_type;
}

void RVM::set_target_size(int _target_size)
{
    target_size = _target_size;
}

void RVM::set_intra_inter(int _intra_inter)
{
    intra_inter = _intra_inter;
}

void RVM::set_postproc_mode(bool _segmentation, bool _refine_deep, bool _refine_fast)
{
    segmentation = _segmentation;
    refine_deep = _refine_deep;
    refine_fast = _refine_fast;
}

int RVM::detect(const cv::Mat& rgb, InterFeatures& feats, cv::Mat& fgr, cv::Mat& pha, cv::Mat& seg)
{
    const int w = rgb.cols;
    const int h = rgb.rows;

    const int max_stride = 16;

    bool refine_deep = true;
    // bool refine_fast = true;

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    ncnn::Mat in_pad;
    ncnn::Mat in_small_pad;

    int wpad = 0;
    int hpad = 0;

    bool downsample = std::max(w, h) > target_size;
    if (downsample)
    {
        // letterbox pad to multiple of max_stride
        int w2 = w;
        int h2 = h;
        float scale = 1.f;
        if (w > h)
        {
            scale = (float)target_size / w;
            w2 = target_size;
            h2 = h2 * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h2 = target_size;
            w2 = w2 * scale;
        }

        ncnn::Mat in_small = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, w, h, w2, h2);

        // letterbox pad to target_size rectangle
        int w2pad = (w2 + max_stride - 1) / max_stride * max_stride - w2;
        int h2pad = (h2 + max_stride - 1) / max_stride * max_stride - h2;
        ncnn::copy_make_border(in_small, in_small_pad, h2pad / 2, h2pad - h2pad / 2, w2pad / 2, w2pad - w2pad / 2, ncnn::BORDER_CONSTANT, 114.f);

        in_small_pad.substract_mean_normalize(0, norm_vals);

        int w3 = w;
        int h3 = h;
        if (w > h)
        {
            w3 = w;
            h3 = in_small_pad.h / scale;
            wpad = 0;
            hpad = h3 - h;
        }
        else
        {
            h3 = h;
            w3 = in_small_pad.w / scale;
            wpad = w3 - w;
            hpad = 0;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, w, h);

        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        in_pad.substract_mean_normalize(0, norm_vals);
    }
    else
    {
        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, w, h);

        // letterbox pad to target_size rectangle
        wpad = (w + max_stride - 1) / max_stride * max_stride - w;
        hpad = (h + max_stride - 1) / max_stride * max_stride - h;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        in_pad.substract_mean_normalize(0, norm_vals);

        in_small_pad = in_pad;
    }

    bool is_first_frame = false;
    if (feats.r1.empty() && feats.r2.empty() && feats.r3.empty() && feats.r4.empty())
    {
        is_first_frame = true;
    }

    if (model_type == 0)
    {
        // rvm_mobilenetv3
        feats.r1.create(in_small_pad.w / 2, in_small_pad.h / 2, 16);
        feats.r2.create(in_small_pad.w / 4, in_small_pad.h / 4, 20);
        feats.r3.create(in_small_pad.w / 8, in_small_pad.h / 8, 40);
        feats.r4.create(in_small_pad.w / 16, in_small_pad.h / 16, 64);
    }
    else // if (model_type == 1)
    {
        // rvm_resnet50
        feats.r1.create(in_small_pad.w / 2, in_small_pad.h / 2, 16);
        feats.r2.create(in_small_pad.w / 4, in_small_pad.h / 4, 32);
        feats.r3.create(in_small_pad.w / 8, in_small_pad.h / 8, 64);
        feats.r4.create(in_small_pad.w / 16, in_small_pad.h / 16, 128);
    }

    if (intra_inter == 0 || is_first_frame)
    {
        feats.r1.fill(0.f);
        feats.r2.fill(0.f);
        feats.r3.fill(0.f);
        feats.r4.fill(0.f);
    }

    ncnn::Extractor ex = rvm.create_extractor();

    ex.input("in0", in_pad);
    ex.input("in1", in_small_pad);

    ex.input("in2", feats.r1);
    ex.input("in3", feats.r2);
    ex.input("in4", feats.r3);
    ex.input("in5", feats.r4);

    ncnn::Mat out_fgr;
    ncnn::Mat out_pha;
    ncnn::Mat out_seg;

    if (segmentation)
    {
        // segmentation
        ex.extract("out6", out_seg);
    }
    else if (downsample)
    {
        if (refine_deep)
        {
            // downsample + refine deep
            ex.extract("out2", out_fgr);
            ex.extract("out3", out_pha);
        }
        else // if (refine_fast)
        {
            // downsample + refine fast
            ex.extract("out4", out_fgr);
            ex.extract("out5", out_pha);
        }
    }
    else
    {
        // no downsample
        ex.extract("out0", out_fgr);
        ex.extract("out1", out_pha);
    }

    if (intra_inter == 1)
    {
        // feats
        ex.extract("out7", feats.r1, 1);
        ex.extract("out8", feats.r2, 1);
        ex.extract("out9", feats.r3, 1);
        ex.extract("out10", feats.r4, 1);
    }

    const float denorm_vals[3] = {255.f, 255.f, 255.f};

    if (segmentation)
    {
        out_seg.substract_mean_normalize(0, denorm_vals);
        seg.create(in_pad.h, in_pad.w, CV_8UC1);
        out_seg.to_pixels_resize(seg.data, ncnn::Mat::PIXEL_GRAY, in_pad.w, in_pad.h);

        // cut letterbox pad
        seg = seg(cv::Rect(wpad / 2, hpad / 2, w, h));
    }
    else
    {
        out_fgr.substract_mean_normalize(0, denorm_vals);
        fgr.create(out_fgr.h, out_fgr.w, CV_8UC3);
        out_fgr.to_pixels(fgr.data, ncnn::Mat::PIXEL_RGB);

        out_pha.substract_mean_normalize(0, denorm_vals);
        pha.create(out_pha.h, out_pha.w, CV_8UC1);
        out_pha.to_pixels(pha.data, ncnn::Mat::PIXEL_GRAY);

        // cut letterbox pad
        fgr = fgr(cv::Rect(wpad / 2, hpad / 2, w, h));
        pha = pha(cv::Rect(wpad / 2, hpad / 2, w, h));
    }

    return 0;
}

int RVM::draw(cv::Mat& rgb, const cv::Mat& fgr, const cv::Mat& pha, const cv::Mat& seg)
{
    const int w = rgb.cols;
    const int h = rgb.rows;

    if (!fgr.empty() && !pha.empty())
    {
        // composite
        for (int y = 0; y < h; y++)
        {
            const uchar* pf = fgr.ptr<const uchar>(y);
            const uchar* pa = pha.ptr<const uchar>(y);
            uchar* p = rgb.ptr<uchar>(y);
            for (int x = 0; x < w; x++)
            {
                const float alpha = pa[0] / 255.f;
                p[0] = cv::saturate_cast<uchar>(pf[0] * alpha + (1 - alpha) * 120);
                p[1] = cv::saturate_cast<uchar>(pf[1] * alpha + (1 - alpha) * 255);
                p[2] = cv::saturate_cast<uchar>(pf[2] * alpha + (1 - alpha) * 155);
                pf += 3;
                pa += 1;
                p += 3;
            }
        }
    }
    else if (!seg.empty())
    {
        // composite seg
        for (int y = 0; y < h; y++)
        {
            uchar* p = rgb.ptr<uchar>(y);
            const uchar* ps = seg.ptr<const uchar>(y);
            for (int x = 0; x < w; x++)
            {
                const float alpha = ps[0] / 255.f;
                p[0] = cv::saturate_cast<uchar>(p[0] * alpha + (1 - alpha) * 120);
                p[1] = cv::saturate_cast<uchar>(p[1] * alpha + (1 - alpha) * 255);
                p[2] = cv::saturate_cast<uchar>(p[2] * alpha + (1 - alpha) * 155);
                ps += 1;
                p += 3;
            }
        }
    }

    return 0;
}
