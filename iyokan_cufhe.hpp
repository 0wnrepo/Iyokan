#ifndef VIRTUALSECUREPLATFORM_IYOKAN_CUFHE_HPP
#define VIRTUALSECUREPLATFORM_IYOKAN_CUFHE_HPP

#include <cufhe.h>
#include <cufhe_gpu.cuh>

#include "iyokan.hpp"

#define CUFHE_GPU_NUM (8)

struct CUFHEWorkerInfo {
    std::shared_ptr<cufhe::Stream> stream;
};

using TaskCUFHEGate = Task<cufhe::Ctxt, cufhe::Ctxt, CUFHEWorkerInfo>;

inline void copyCtxt(cufhe::Ctxt& dst, const cufhe::Ctxt& src,
                     std::shared_ptr<cufhe::Stream> stream = nullptr)
{
    if (stream){
        cufhe::mCopy(dst, src, *stream);
    cufhe::Synchronize(CUFHE_GPU_NUM);
    }
    else{
        cufhe::mCopySync(dst, src);
    }
}

class TaskCUFHEGateMem : public TaskCUFHEGate {
public:
    TaskCUFHEGateMem(int numInputs) : TaskCUFHEGate(numInputs)
    {
    }

    void set(const cufhe::Ctxt& newval)
    {
        copyCtxt(output(), newval);
    }

    const cufhe::Ctxt& get() const
    {
        return output();
    }
};

class TaskCUFHEGateDFF : public TaskCUFHEGateMem {
protected:
    void startAsyncImpl(CUFHEWorkerInfo) override
    {
    }

public:
    TaskCUFHEGateDFF() : TaskCUFHEGateMem(1)
    {
    }

    bool areInputsReady() const override
    {
        // Since areInputsReady() is called after calling of tick(), the
        // input should already be in val_.
        return true;
    }

    void tick() override
    {
        TaskCUFHEGateMem::tick();
        copyCtxt(output(), input(0));
    }

    bool hasFinished() const override
    {
        // Since hasFinished() is called after calling of tick(), the
        // input should already be in val_.
        return true;
    }
};

class TaskCUFHEGateWIRE : public TaskCUFHEGateMem {
private:
    CUFHEWorkerInfo wi_;

private:
    void startAsyncImpl(CUFHEWorkerInfo wi) override
    {
        wi_ = std::move(wi);

        switch (getInputSize()) {
        case 0:
            // Nothing to do!
            break;
        case 1:
            copyCtxt(output(), input(0), wi_.stream);
            break;
        default:
            assert(false);
        }
    }

public:
    TaskCUFHEGateWIRE(bool inputNeeded) : TaskCUFHEGateMem(inputNeeded ? 1 : 0)
    {
    }

    bool hasFinished() const override
    {
        return getInputSize() == 0 || cufhe::StreamQuery(*wi_.stream);
    }
};

#define DEFINE_TASK_GATE(name, numInputs, expr)          \
    class TaskCUFHEGate##name : public TaskCUFHEGate {   \
    private:                                             \
        CUFHEWorkerInfo wi_;                             \
                                                         \
    private:                                             \
        void startAsyncImpl(CUFHEWorkerInfo wi) override \
        {                                                \
            wi_ = std::move(wi);                         \
            auto st = wi_.stream;                        \
            (expr);                                      \
        }                                                \
                                                         \
    public:                                              \
        TaskCUFHEGate##name() : TaskCUFHEGate(numInputs) \
        {                                                \
        }                                                \
        bool hasFinished() const override                \
        {                                                \
            return cufhe::StreamQuery(*wi_.stream);      \
        }                                                \
    };
DEFINE_TASK_GATE(AND, 2, cufhe::mAnd(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(NAND, 2, cufhe::mNand(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(ANDNOT, 2, cufhe::mAndYN(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(OR, 2, cufhe::mOr(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(NOR, 2, cufhe::mNor(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(ORNOT, 2, cufhe::mOrYN(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(XOR, 2, cufhe::mXor(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(XNOR, 2, cufhe::mXnor(output(), input(0), input(1), *st));
DEFINE_TASK_GATE(MUX, 3,
                 cufhe::mMux(output(), input(2), input(1), input(0), *st));
DEFINE_TASK_GATE(NOT, 1, cufhe::mNot(output(), input(0), *st));
#undef DEFINE_TASK_GATE

class CUFHENetworkBuilder
    : public NetworkBuilder<TaskCUFHEGate, TaskCUFHEGateMem, TaskCUFHEGateDFF,
                            TaskCUFHEGateWIRE, CUFHEWorkerInfo> {
private:
#define DEFINE_GATE_IMPL(name)                           \
    std::shared_ptr<TaskCUFHEGate> name##Impl() override \
    {                                                    \
        return std::make_shared<TaskCUFHEGate##name>();  \
    }
    DEFINE_GATE_IMPL(AND);
    DEFINE_GATE_IMPL(NAND);
    DEFINE_GATE_IMPL(ANDNOT);
    DEFINE_GATE_IMPL(OR);
    DEFINE_GATE_IMPL(NOR);
    DEFINE_GATE_IMPL(ORNOT);
    DEFINE_GATE_IMPL(XOR);
    DEFINE_GATE_IMPL(XNOR);
    DEFINE_GATE_IMPL(MUX);
    DEFINE_GATE_IMPL(NOT);
#undef DEFINE_GATE_IMPL
};

using CUFHENetwork = CUFHENetworkBuilder::NetworkType;

class CUFHEWorker : public Worker<CUFHEWorkerInfo> {
private:
    CUFHEWorkerInfo wi_;

private:
    CUFHEWorkerInfo getWorkerInfo() override
    {
        return wi_;
    }

public:
    CUFHEWorker(ReadyQueue<CUFHEWorkerInfo>& readyQueue,
                size_t& numFinishedTargets,
                std::shared_ptr<ProgressGraphMaker> graph, int gpuNum)
        : Worker(readyQueue, numFinishedTargets, graph)
    {
        wi_.stream = std::make_shared<cufhe::Stream>(gpuNum, 0);
        wi_.stream->Create();
    }
};

void doCUFHE(const Options& opt);
void processAllGates(CUFHENetwork& net, int numWorkers,
                     std::shared_ptr<ProgressGraphMaker> graph = nullptr);
#endif
