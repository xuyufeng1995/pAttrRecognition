//
// Created by xuyufeng1 on 2021/6/8.
//

#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace nvinfer1;

static Logger gLogger;

static const int INPUT_H = 256;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 1*8*4*2;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights:" << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }

        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps) {
    /// adjustedScale = scale / sqrt(variance + epsilon)
    /// batchNorm = (input + bias - (adjustedScale * mean)) * adjustedScale

    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len:" << len << std::endl;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    /// output=(input * scale + shift)^{power}
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);

    return scale_1;
}

IActivationLayer* basicConv2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, DimsHW ksize, int s, DimsHW p, std::string lname) {

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, ksize, weightMap[lname + ".weight"], weightMap[lname + ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(p);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;
}

IConcatenationLayer* inceptionA(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, const int conv_out[7]) {
    IActivationLayer* relu1 = basicConv2d(network, weightMap, input, conv_out[0], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_1x1");

    IActivationLayer* relu2 = basicConv2d(network, weightMap, input, conv_out[1], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_3x3_reduce");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), conv_out[2], DimsHW(3, 3), 1, DimsHW(1, 1), lname + "_3x3");

    IActivationLayer* relu3 = basicConv2d(network, weightMap, input, conv_out[3], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_double_3x3_reduce");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), conv_out[4], DimsHW(3, 3), 1, DimsHW(1, 1), lname + "_double_3x3_1");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), conv_out[5], DimsHW(3, 3), 1, DimsHW(1, 1), lname + "_double_3x3_2");

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setPaddingNd(DimsHW{1, 1});
    pool1->setAverageCountExcludesPadding(false);
    IActivationLayer* relu4 = basicConv2d(network, weightMap, *pool1->getOutput(0), conv_out[6], DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "_pool_proj");

    ITensor* inputTensors[] = {relu1->getOutput(0), relu2->getOutput(0), relu3->getOutput(0), relu4->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 4);
    assert(cat1);

    return cat1;
}

IConcatenationLayer* inceptionB(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, const int conv_out[5]) {
    IActivationLayer* relu1 = basicConv2d(network, weightMap, input, conv_out[0], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_3x3_reduce");
    relu1 = basicConv2d(network, weightMap, *relu1->getOutput(0), conv_out[1], DimsHW(3, 3), 2, DimsHW(1, 1), lname + "_3x3");

    IActivationLayer* relu2 = basicConv2d(network, weightMap, input, conv_out[2], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_double_3x3_reduce");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), conv_out[3], DimsHW(3, 3), 1, DimsHW(1, 1), lname + "_double_3x3_1");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), conv_out[4], DimsHW(3, 3), 2, DimsHW(1, 1), lname + "_double_3x3_2");

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    /// 添加设置ceil=True, 第二种方法，
    pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    ITensor* inputTensors[] = {relu1->getOutput(0), relu2->getOutput(0), pool1->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 3);
    assert(cat1);

    return cat1;
}

IConcatenationLayer* inceptionC(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, const int conv_out[7]) {
    IActivationLayer* relu1 = basicConv2d(network, weightMap, input, conv_out[0], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_1x1");

    IActivationLayer* relu2 = basicConv2d(network, weightMap, input, conv_out[1], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_3x3_reduce");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), conv_out[2], DimsHW(3, 3), 1, DimsHW(1, 1), lname + "_3x3");

    IActivationLayer* relu3 = basicConv2d(network, weightMap, input, conv_out[3], DimsHW(1, 1), 1, DimsHW(0, 0), lname + "_double_3x3_reduce");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), conv_out[4], DimsHW(3, 3), 1, DimsHW(1, 1), lname + "_double_3x3_1");
    relu3 = basicConv2d(network, weightMap, *relu3->getOutput(0), conv_out[5], DimsHW(3, 3), 1, DimsHW(1, 1), lname + "_double_3x3_2");

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setPaddingNd(DimsHW{1, 1});
    //pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
    IActivationLayer* relu4 = basicConv2d(network, weightMap, *pool1->getOutput(0), conv_out[6], DimsHW{1, 1}, 1, DimsHW{0, 0}, lname + "_pool_proj");

    ITensor* inputTensors[] = {relu1->getOutput(0), relu2->getOutput(0), relu3->getOutput(0), relu4->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 4);
    assert(cat1);

    return cat1;
}

IActivationLayer* ChannelAttn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int channel, int pooling_size) {
    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{pooling_size, pooling_size / 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setAverageCountExcludesPadding(false);

    IConvolutionLayer* conv1 = network->addConvolutionNd(*pool1->getOutput(0), channel / 16, DimsHW{1, 1}, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});

    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), channel, DimsHW{1, 1}, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});

    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kSIGMOID);
    assert(relu2);

    return relu2;
}

IPluginV2Layer* SpatialTransformBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int channel, int pooling_size) {
    int index_per_class[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 9, 2, 10, 11, 12, 13, 2, 14, 15};
    int attribute_per_class_merged[16] = {2, 2, 12, 3, 2, 2, 3, 4, 2, 3, 4, 3, 3, 5, 3, 4};
    ITensor* inputTensors[19];

    //for (int i = 0; i < 19; i++) {
        /// stn_feature = features * self.att_list[i](features) + features
        IActivationLayer* relu1 = ChannelAttn(network,weightMap, input, lname + ".att_list." + std::to_string(0), channel, pooling_size);
        IElementWiseLayer* elem1 = network->addElementWise(input, *relu1->getOutput(0), ElementWiseOperation::kPROD);
        elem1 = network->addElementWise(*elem1->getOutput(0), input, ElementWiseOperation::kSUM);

        /// theta_i = self.stn_list[i](F.avg_pool2d(stn_feature, stn_feature.size()[2:]).view(bs, -1)).view(-1, 4)
        IPoolingLayer* pool1 = network->addPoolingNd(*elem1->getOutput(0), PoolingType::kAVERAGE, DimsHW{pooling_size, pooling_size / 2});
        assert(pool1);

        IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool1->getOutput(0), 4, weightMap[lname + ".stn_list." + std::to_string(index_per_class[0]) + ".weight"], weightMap[lname + ".stn_list." + std::to_string(index_per_class[0]) + ".bias"]);
        assert(fc1);

        /// theta_i = self.transform_theta(theta_i, i)
        auto creator = getPluginRegistry()->getPluginCreator("AffineGridLayer_TRT", "1");
        PluginField plugin_fields[1];
        int featureMapShape[3] = {pooling_size, pooling_size / 2, 2};
        plugin_fields[0].data = featureMapShape;
        plugin_fields[0].length = 3;
        plugin_fields[0].name = "featureMapShape";
        plugin_fields[0].type = PluginFieldType::kFLOAT32;
        PluginFieldCollection plugin_data;
        plugin_data.nbFields = 1;
        plugin_data.fields = plugin_fields;
        IPluginV2 *pluginobj = creator->createPlugin("MaxLayer_TRT", &plugin_data);
        ITensor* maxLayer[] = {fc1->getOutput(0)};
        auto out = network->addPluginV2(maxLayer, 1, *pluginobj);
        return out;


        /// pred = self.gap_list[i](sub_feature).view(bs, -1)
        IPoolingLayer* pool2 = network->addPoolingNd(*elem1->getOutput(0), PoolingType::kAVERAGE, DimsHW{pooling_size, pooling_size / 2});
        assert(pool2);
        pool2->setStrideNd(DimsHW{1, 1});
        pool2->setAverageCountExcludesPadding(false);

        /// pred = self.fc_list[self.index_per_class[i]](pred)
        IFullyConnectedLayer* fc2 = network->addFullyConnected(*pool2->getOutput(0), attribute_per_class_merged[index_per_class[0]], weightMap[lname + ".fc_list." + std::to_string(index_per_class[0]) + ".weight"], weightMap[lname + ".fc_list." + std::to_string(index_per_class[0]) + ".bias"]);
        assert(fc2);

    //    inputTensors[i] = fc2->getOutput(0);
    //}

    //IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 19);
    //assert(cat1);

    //return cat1;
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    //create input tensor of shape {1, 1, 256, 128} with name INPUT_BLOB_NAME
    ITensor* data= network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../inception.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IActivationLayer* relu1 = basicConv2d(network, weightMap, *data,64, DimsHW{7, 7}, 2, DimsHW{3, 3}, "main_branch.conv1_7x7_s2");
    /// 添加padding层来抵消ceil=True, 第一种方法，
    IPaddingLayer* pad1 = network->addPaddingNd(*relu1->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});
    assert(pad1);
    IPoolingLayer* pool1 = network->addPoolingNd(*pad1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    IActivationLayer* relu2 = basicConv2d(network, weightMap, *pool1->getOutput(0), 64, DimsHW{1, 1}, 1, DimsHW{0, 0}, "main_branch.conv2_3x3_reduce");
    relu2 = basicConv2d(network, weightMap, *relu2->getOutput(0), 192, DimsHW{3, 3}, 1, DimsHW{1, 1}, "main_branch.conv2_3x3");
    IPaddingLayer* pad2 = network->addPaddingNd(*relu2->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});
    assert(pad1);
    IPoolingLayer* pool2 = network->addPoolingNd(*pad2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool2->setStrideNd(DimsHW{2, 2});

    int outSize3a[7] = {64, 64, 64, 64, 96, 96, 32};
    IConcatenationLayer* inception_3a = inceptionA(network, weightMap, *pool2->getOutput(0), "main_branch.inception_3a", outSize3a);
    int outSize3b[7] = {64, 64, 96, 64, 96, 96, 64};
    IConcatenationLayer* inception_3b = inceptionA(network, weightMap, *inception_3a->getOutput(0), "main_branch.inception_3b", outSize3b);
    int outSize3c[5] = {128, 160, 64, 96, 96};
    IConcatenationLayer* inception_3c = inceptionB(network, weightMap, *inception_3b->getOutput(0), "main_branch.inception_3c", outSize3c);

    int outSize4a[7] = {224, 64, 96, 96, 128, 128, 128};
    IConcatenationLayer* inception_4a = inceptionA(network, weightMap, *inception_3c->getOutput(0), "main_branch.inception_4a", outSize4a);
    int outSize4b[7] = {192, 96, 128, 96, 128, 128, 128};
    IConcatenationLayer* inception_4b = inceptionA(network, weightMap, *inception_4a->getOutput(0), "main_branch.inception_4b", outSize4b);
    int outSize4c[7] = {160, 128, 160, 128, 160, 160, 128};
    IConcatenationLayer* inception_4c = inceptionA(network, weightMap, *inception_4b->getOutput(0), "main_branch.inception_4c", outSize4c);
    int outSize4d[7] = {96, 128, 192, 160, 192, 192, 128};
    IConcatenationLayer* inception_4d = inceptionA(network, weightMap, *inception_4c->getOutput(0), "main_branch.inception_4d", outSize4d);
    int outSize4e[5] = {128, 192, 192, 256, 256};
    IConcatenationLayer* inception_4e = inceptionB(network, weightMap, *inception_4d->getOutput(0), "main_branch.inception_4e", outSize4e);

    int outSize5a[7] = {352, 192, 320, 160, 224, 224, 128};
    IConcatenationLayer* inception_5a = inceptionA(network, weightMap, *inception_4e->getOutput(0), "main_branch.inception_5a", outSize5a);
    int outSize5b[7] = {352, 192, 320, 192, 224, 224, 128};
    IConcatenationLayer* inception_5b = inceptionC(network, weightMap, *inception_5a->getOutput(0), "main_branch.inception_5b", outSize5b);

    /// AdaptivePooling与Max/AvgPooling相互转换
    /// stride = floor ( (input_size / (output_size−1) )
    /// kernel_size = input_size − (output_size−1) * stride
    IPoolingLayer* pool3 = network->addPoolingNd(*inception_5b->getOutput(0), PoolingType::kAVERAGE, DimsHW{8, 4});
    assert(pool3);
    pool3->setStrideNd(DimsHW{1, 1});
    pool3->setAverageCountExcludesPadding(false);

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool3->getOutput(0), 93, weightMap["finalfc.weight"], weightMap["finalfc.bias"]);
    assert(fc1);

    /// Lateral layers
    IConvolutionLayer* latlayer_5b = network->addConvolutionNd(*inception_5b->getOutput(0), 128, DimsHW{1, 1}, weightMap["latlayer_5b.weight"], weightMap["latlayer_5b.bias"]);
    assert(latlayer_5b);
    latlayer_5b->setStrideNd(DimsHW{1, 1});

    IConvolutionLayer* latlayer_4d = network->addConvolutionNd(*inception_4d->getOutput(0), 128, DimsHW{1, 1}, weightMap["latlayer_4d.weight"], weightMap["latlayer_4d.bias"]);
    assert(latlayer_4d);
    latlayer_4d->setStrideNd(DimsHW{1, 1});
    /// upsample  x: [1, 128, 8, 4]   y:[1, 128, 16, 8]   ------------>  [1, 128, 16, 8]
    IResizeLayer* upsample1 = network->addResize(*latlayer_5b->getOutput(0));
    assert(upsample1);
    upsample1->setResizeMode(ResizeMode::kLINEAR);
    upsample1->setAlignCorners(false);
    upsample1->setOutputDimensions(latlayer_4d->getOutput(0)->getDimensions());
    std::cout << upsample1->getOutput(0)->getDimensions().d[0] << ", " << upsample1->getOutput(0)->getDimensions().d[1] << ", " << upsample1->getOutput(0)->getDimensions().d[2] << std::endl;
    ITensor* fusion_4d[] = {upsample1->getOutput(0), latlayer_4d->getOutput(0)};
    IConcatenationLayer* cat2 = network->addConcatenation(fusion_4d, 2);

    IConvolutionLayer* latlayer_3b = network->addConvolutionNd(*inception_3b->getOutput(0), 128, DimsHW{1, 1}, weightMap["latlayer_3b.weight"], weightMap["latlayer_3b.bias"]);
    assert(latlayer_3b);
    latlayer_3b->setStrideNd(DimsHW{1, 1});
    /// upsample    x:[1, 256, 16, 8]   y:[1, 128, 32, 16] ------> [1, 256, 32, 16]
    IResizeLayer* upsample2 = network->addResize(*cat2->getOutput(0));
    assert(upsample2);
    upsample2->setResizeMode(ResizeMode::kLINEAR);
    upsample2->setAlignCorners(false);
    upsample2->setOutputDimensions(Dims3{256,32,16});
    std::cout << upsample2->getOutput(0)->getDimensions().d[0] << ", " << upsample2->getOutput(0)->getDimensions().d[1] << ", " << upsample2->getOutput(0)->getDimensions().d[2] << std::endl;
    ITensor* fusion_3b[] = {upsample2->getOutput(0), latlayer_3b->getOutput(0)};
    IConcatenationLayer* cat3 = network->addConcatenation(fusion_3b, 2);


    auto cat1 = SpatialTransformBlock(network, weightMap, *latlayer_5b->getOutput(0), "st_5b", 128, 8);
//    cat2 = SpatialTransformBlock(network, weightMap, *cat2->getOutput(0), "st_4d", 128 * 2, 16);
//    cat3 = SpatialTransformBlock(network, weightMap, *cat3->getOutput(0), "st_3b", 128 * 3, 32);
//
//    ITensor* outputTensors[] = {cat3->getOutput(0), cat2->getOutput(0), cat1->getOutput(0), fc1->getOutput(0)};
//    IConcatenationLayer* cat4 = network->addConcatenation(outputTensors, 4);
//
//    auto creator = getPluginRegistry()->getPluginCreator("MaxLayer_TRT", "1");
//    const PluginFieldCollection* pluginData = creator->getFieldNames();
//    IPluginV2 *pluginobj = creator->createPlugin("MaxLayer_TRT", pluginData);
//    ITensor* maxLayer[] = {cat4->getOutput(0)};
//    auto out = network->addPluginV2(maxLayer, 1, *pluginobj);

    cat1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*cat1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 23);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // release host memory
    for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine
    (*modelStream) = engine->serialize();

    // close everything down
    engine->destroy();
    config->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    cudaMalloc(&buffers[inputIndex], sizeof(float) * 3 * INPUT_W * INPUT_H * batchSize);
    cudaMalloc(&buffers[outputIndex], sizeof(float) * OUTPUT_SIZE * batchSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(buffers[inputIndex], input, sizeof(float) * 3 * INPUT_W * INPUT_H * batchSize, cudaMemcpyHostToDevice, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[outputIndex], sizeof(float) * OUTPUT_SIZE * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;

    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("inception.engine", std::ios::binary);
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        std::cout << "build success" << std::endl;
    //    return 0;
    //} else if (std::string(argv[1]) == "-d") {
        std::ifstream file("inception.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    cv::Mat img = cv::imread("/home/xuyufeng/projects/python/rider_convert/data/D2A40B75-B292-EF6F-42E9-88CD04000000.jpg");
    cv::Mat re(INPUT_H, INPUT_W, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = re.data + row * re.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    static float prob[OUTPUT_SIZE];
    std::cout << setiosflags(std::ios::fixed) << std::setprecision(4);
    for (int i = 0; i < 1; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);

        std::ofstream outFile("trt_out.txt");
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            outFile << prob[k] << " ";
            if ((k + 1) % 4 == 0) outFile << std::endl;
        }
        outFile.close();
        std::cout << std::endl;
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::cout << "success!!!!!!!!!" << std::endl;
    return 0;
}

