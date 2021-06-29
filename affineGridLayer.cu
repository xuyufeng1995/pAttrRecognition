//
// Created by xuyufeng1 on 2021/6/25.
//
#include "affineGridLayer.h"
#include <assert.h>
#include <math.h>


namespace Tn
{
    template<typename T>
    void write(char* &buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char* &buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}


namespace nvinfer1 {
    AffineGridPlugin::AffineGridPlugin(int outputHeight, int outputWidth, int outputChannel)
    {
        mHeight = outputHeight;
        mWidth = outputWidth;
        mChannel = outputChannel;
    }

    AffineGridPlugin::~AffineGridPlugin()
    {

    }

    AffineGridPlugin::AffineGridPlugin(const void *data, size_t length) {
        using namespace Tn;
        const char* d = reinterpret_cast<const char*>(data), *a = d;
        read(d, mHeight);
        read(d, mWidth);
        read(d, mChannel);
        read(d, mThreadCount);

        assert(d == a + length);
    }

    void AffineGridPlugin::serialize(void *buffer) const {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mHeight);
        write(d, mWidth);
        write(d, mChannel);
        write(d, mThreadCount);

        assert(d == a + getSerializationSize());
    }

    size_t AffineGridPlugin::getSerializationSize() const {
        return sizeof(mHeight) + sizeof(mWidth) + sizeof(mChannel) + sizeof(mThreadCount);
    }

    int AffineGridPlugin::initialize() {
        return 0;
    }

    Dims AffineGridPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
        return Dims3(mHeight, mWidth, mChannel);
    };

    void AffineGridPlugin::setPluginNamespace(const char *pluginNamespace) {
        mPluginNamespace = pluginNamespace;
    }

    const char* AffineGridPlugin::getPluginNamespace() const {
        return mPluginNamespace;
    }

    DataType AffineGridPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
        return DataType::kFLOAT;
    }

    bool AffineGridPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                        int nbInputs) const {
        return false;
    }

    bool AffineGridPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void AffineGridPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) {
        assert(nbInput == 1);
        assert(nbOutput == 1);

    }

    void AffineGridPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) {

    }

    void AffineGridPlugin::detachFromContext() {

    }

    const char* AffineGridPlugin::getPluginType() const {
        return "AffineGridLayer_TRT";
    }

    const char* AffineGridPlugin::getPluginVersion() const {
        return "1";
    }

    void AffineGridPlugin::destroy() {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* AffineGridPlugin::clone() const {
        AffineGridPlugin *p = new AffineGridPlugin(mHeight, mWidth, mChannel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __global__ void affineGrid(const float *input, float *output, const int height, const int width) {
        float* grid;
        grid = new float[height * width * 3];
        for (int i = 0; i < width; i++) {
            float tmp = -1.0 + (2.0 * i + 1.0) / width;
            for (int j = 0; j < height; j++) {
                grid[i * 3 + j * 3 * width] = tmp;     // x
            }
        }

        for (int i = 0; i < height; i++) {
            float tmp = -1.0 + (2.0 * i + 1.0) / height;
            for (int j = 0; j < width; j++) {
                grid[i * width * 3 + j * 3 + 1] = tmp;   // y
                grid[i * width * 3 + j * 3 + 2] = 1.0;   // z = 1.0
            }
        }

        float theta[6] = {0};
        theta[0] = 1.0f / (1.0f + exp(-input[0]));
        theta[3] = 1.0f / (1.0f + exp(-input[1]));
        theta[4] = tanh(input[2]);
        theta[5] = tanh(input[3]);

        for (int i = 0; i < width * height; i++) {
            for (int j = 0; j < 2; j++) {
                float sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += grid[i * 3 + k] * theta[k * 2 + j];
                }
                output[i * 2 + j] = sum;
            }
        }

        delete[] grid;
    }


    void AffineGridPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize) {

        affineGrid<<<1, 1>>>(inputs[0], output, mHeight, mWidth);
    }

    int AffineGridPlugin::enqueue(int batchSize, const void* const *inputs, void** outputs, void* workspace, cudaStream_t stream) {
        forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection AffineGridPluginCreator::mFC{};
    std::vector<PluginField> AffineGridPluginCreator::mPluginAttributes;

    AffineGridPluginCreator::AffineGridPluginCreator() {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* AffineGridPluginCreator::getPluginName() const {
        return "AffineGridLayer_TRT";
    }

    const char* AffineGridPluginCreator::getPluginVersion() const {
        return "1";
    }

    const PluginFieldCollection* AffineGridPluginCreator::getFieldNames() {
        return &mFC;
    }

    IPluginV2IOExt* AffineGridPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) {
        assert(fc->nbFields == 1);
        assert(strcmp(fc->fields[0].name, "featureMapShape") == 0);

        int *p_featureMapShape = (int*)(fc->fields[0].data);
        int outputHeight = p_featureMapShape[0];
        int outputWidth = p_featureMapShape[1];
        int outputChannel = p_featureMapShape[2];
        AffineGridPlugin* obj = new AffineGridPlugin(outputHeight, outputWidth, outputChannel);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* AffineGridPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                               size_t serialLength) {
        AffineGridPlugin* obj = new AffineGridPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}
