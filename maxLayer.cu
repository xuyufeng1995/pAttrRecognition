#include <assert.h>
#include <iostream>
#include "maxLayer.h"

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

using namespace Inception;

namespace nvinfer1
{
    MaxLayerPlugin::MaxLayerPlugin()
    {
        mClassCount = CLASS_NUM;
    }

    MaxLayerPlugin::~MaxLayerPlugin()
    {

    }

    MaxLayerPlugin::MaxLayerPlugin(const void *data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);

        assert(d == a + length);
    }

    void MaxLayerPlugin::serialize(void *buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);

        assert(d == a + getSerializationSize());
    }

    size_t MaxLayerPlugin::getSerializationSize() const
    {
        return sizeof(mClassCount) + sizeof(mThreadCount);
    }

    int MaxLayerPlugin::initialize()
    {
        return 0;
    }

    Dims MaxLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        // output the result to channel;
        return Dims3(1, 1, mClassCount);
    }

    // set plugin namespace
    void MaxLayerPlugin::setPluginNamespace(const char *pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* MaxLayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType MaxLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool MaxLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if output tensor is broadcast across a batch.
    bool MaxLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void MaxLayerPlugin::configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput)
    {

    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void MaxLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {

    }

    // Detach the plugin object from its execution context.
    void MaxLayerPlugin::detachFromContext()
    {

    }

    const char* MaxLayerPlugin::getPluginType() const
    {
        return "MaxLayer_TRT";
    }

    const char* MaxLayerPlugin::getPluginVersion() const
    {
        return "1";
    }

    void MaxLayerPlugin::destroy()
    {
        delete this;
    }

    // clone the plugin
    IPluginV2IOExt* MaxLayerPlugin::clone() const
    {
        MaxLayerPlugin *p = new MaxLayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __global__ void Max(const float *input, float *output, int classes) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= classes) return;

        output[idx] = input[idx];
        for (int i = 1; i < 4; i++) {
            if (output[idx] < input[idx + i * classes]) {
                output[idx] = input[idx + i * classes];
            }
        }

    }

    void MaxLayerPlugin::forwardGpu(const float *const *inputs, float *output, cudaStream_t stream, int batchSize)
    {
        int outputElem = mClassCount;
        for (int idx = 0; idx < batchSize; ++idx) {
            cudaMemset(output + idx * outputElem, 0, sizeof(float));
        }

        Max<<<1, mClassCount>>>(inputs[0], output, mClassCount);
    }

    int MaxLayerPlugin::enqueue(int batchSize, const void* const *inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        assert(batchSize == 1);
        forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);

        return 0;
    }

    PluginFieldCollection MaxLayerPluginCreator::mFC{};
    std::vector<PluginField> MaxLayerPluginCreator::mPluginAttributes;

    MaxLayerPluginCreator::MaxLayerPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* MaxLayerPluginCreator::getPluginName() const
    {
        return "MaxLayer_TRT";
    }

    const char* MaxLayerPluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection* MaxLayerPluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt* MaxLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        MaxLayerPlugin* obj = new MaxLayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* MaxLayerPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        MaxLayerPlugin* obj = new MaxLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
