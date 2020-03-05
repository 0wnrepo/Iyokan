#include "iyokan.hpp"

#include "kvsp-packet.hpp"

//
#include <fstream>

template <class NetworkBuilder, class TaskNetwork>
auto get(TaskNetwork& net, const std::string& kind, const std::string& portName,
         int portBit)
{
    return net.template get<typename NetworkBuilder::ParamTaskTypeWIRE>(
        kind, portName, portBit);
}

// Assume variable names 'NetworkBuilder' and 'net'
#define ASSERT_OUTPUT_EQ(portName, portBit, expected)                          \
    assert(getOutput(get<NetworkBuilder>(net, "output", portName, portBit)) == \
           (expected))
#define SET_INPUT(portName, portBit, val) \
    setInput(get<NetworkBuilder>(net, "input", portName, portBit), val)

template <class NetworkBuilder>
void testNOT()
{
    NetworkBuilder builder;
    builder.INPUT(0, "A", 0);
    builder.NOT(1);
    builder.OUTPUT(2, "out", 0);
    builder.connect(0, 1);
    builder.connect(1, 2);

    TaskNetwork net = std::move(builder);
    auto out = get<NetworkBuilder>(net, "output", "out", 0);

    std::array<std::tuple<int, int>, 8> invals{{{0, 1}, {0, 1}}};
    for (int i = 0; i < 2; i++) {
        // Set inputs.
        SET_INPUT("A", 0, std::get<0>(invals[i]));

        processAllGates(net);

        // Check if results are okay.
        assert(getOutput(out) == std::get<1>(invals[i]));

        net.tick();
    }
}

template <class NetworkBuilder>
void testMUX()
{
    NetworkBuilder builder;
    builder.INPUT(0, "A", 0);
    builder.INPUT(1, "B", 0);
    builder.INPUT(2, "S", 0);
    builder.MUX(3);
    builder.OUTPUT(4, "out", 0);
    builder.connect(0, 3);
    builder.connect(1, 3);
    builder.connect(2, 3);
    builder.connect(3, 4);

    TaskNetwork net = std::move(builder);

    std::array<std::tuple<int, int, int, int>, 8> invals{{/*A,B, S, O*/
                                                          {0, 0, 0, 0},
                                                          {0, 0, 1, 0},
                                                          {0, 1, 0, 0},
                                                          {0, 1, 1, 1},
                                                          {1, 0, 0, 1},
                                                          {1, 0, 1, 0},
                                                          {1, 1, 0, 1},
                                                          {1, 1, 1, 1}}};
    for (int i = 0; i < 8; i++) {
        // Set inputs.
        SET_INPUT("A", 0, std::get<0>(invals[i]));
        SET_INPUT("B", 0, std::get<1>(invals[i]));
        SET_INPUT("S", 0, std::get<2>(invals[i]));

        processAllGates(net);

        // Check if results are okay.
        ASSERT_OUTPUT_EQ("out", 0, std::get<3>(invals[i]));

        net.tick();
    }
}

template <class NetworkBuilder>
void testBinopGates()
{
    NetworkBuilder builder;
    builder.INPUT(0, "in0", 0);
    builder.INPUT(1, "in1", 0);

    int nextId = 10;

    std::unordered_map<std::string, std::array<uint8_t, 4 /* 00, 01, 10, 11 */
                                               >>
        id2res;

#define DEFINE_BINOP_GATE_TEST(name, e00, e01, e10, e11) \
    do {                                                 \
        int gateId = nextId++;                           \
        int outputId = nextId++;                         \
        builder.name(gateId);                            \
        builder.OUTPUT(outputId, "out_" #name, 0);       \
        builder.connect(0, gateId);                      \
        builder.connect(1, gateId);                      \
        builder.connect(gateId, outputId);               \
        id2res["out_" #name] = {e00, e01, e10, e11};     \
    } while (false);
    DEFINE_BINOP_GATE_TEST(AND, 0, 0, 0, 1);
    DEFINE_BINOP_GATE_TEST(NAND, 1, 1, 1, 0);
    DEFINE_BINOP_GATE_TEST(ANDNOT, 0, 0, 1, 0);
    DEFINE_BINOP_GATE_TEST(OR, 0, 1, 1, 1);
    DEFINE_BINOP_GATE_TEST(ORNOT, 1, 0, 1, 1);
    DEFINE_BINOP_GATE_TEST(XOR, 0, 1, 1, 0);
    DEFINE_BINOP_GATE_TEST(XNOR, 1, 0, 0, 1);
#undef DEFINE_BINOP_GATE_TEST

    TaskNetwork net = std::move(builder);

    std::array<std::pair<int, int>, 4> invals{{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
    for (int i = 0; i < 4; i++) {
        // Set inputs.
        SET_INPUT("in0", 0, invals[i].first ? 1 : 0);
        SET_INPUT("in1", 0, invals[i].second ? 1 : 0);

        processAllGates(net);

        // Check if results are okay.
        for (auto&& [portName, res] : id2res)
            ASSERT_OUTPUT_EQ(portName, 0, res[i]);

        net.tick();
    }
}

template <class NetworkBuilder>
void testFromJSONtest_pass_4bit()
{
    const std::string fileName = "test/test-pass-4bit.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto net = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(net.isValid());

    SET_INPUT("io_in", 0, 0);
    SET_INPUT("io_in", 1, 1);
    SET_INPUT("io_in", 2, 1);
    SET_INPUT("io_in", 3, 0);

    processAllGates(net);

    ASSERT_OUTPUT_EQ("io_out", 0, 0);
    ASSERT_OUTPUT_EQ("io_out", 1, 1);
    ASSERT_OUTPUT_EQ("io_out", 2, 1);
    ASSERT_OUTPUT_EQ("io_out", 3, 0);
}

template <class NetworkBuilder>
void testFromJSONtest_and_4bit()
{
    const std::string fileName = "test/test-and-4bit.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto net = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(net.isValid());

    SET_INPUT("io_inA", 0, 0);
    SET_INPUT("io_inA", 1, 0);
    SET_INPUT("io_inA", 2, 1);
    SET_INPUT("io_inA", 3, 1);
    SET_INPUT("io_inB", 0, 0);
    SET_INPUT("io_inB", 1, 1);
    SET_INPUT("io_inB", 2, 0);
    SET_INPUT("io_inB", 3, 1);

    processAllGates(net);

    ASSERT_OUTPUT_EQ("io_out", 0, 0);
    ASSERT_OUTPUT_EQ("io_out", 1, 0);
    ASSERT_OUTPUT_EQ("io_out", 2, 0);
    ASSERT_OUTPUT_EQ("io_out", 3, 1);
}

template <class NetworkBuilder>
void testFromJSONtest_and_4_2bit()
{
    const std::string fileName = "test/test-and-4_2bit.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto net = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(net.isValid());

    SET_INPUT("io_inA", 0, 1);
    SET_INPUT("io_inA", 1, 0);
    SET_INPUT("io_inA", 2, 1);
    SET_INPUT("io_inA", 3, 1);
    SET_INPUT("io_inB", 0, 1);
    SET_INPUT("io_inB", 1, 1);
    SET_INPUT("io_inB", 2, 1);
    SET_INPUT("io_inB", 3, 1);

    processAllGates(net);

    ASSERT_OUTPUT_EQ("io_out", 0, 1);
    ASSERT_OUTPUT_EQ("io_out", 1, 0);
}

template <class NetworkBuilder>
void testFromJSONtest_mux_4bit()
{
    const std::string fileName = "test/test-mux-4bit.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto net = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(net.isValid());

    SET_INPUT("io_inA", 0, 0);
    SET_INPUT("io_inA", 1, 0);
    SET_INPUT("io_inA", 2, 1);
    SET_INPUT("io_inA", 3, 1);
    SET_INPUT("io_inB", 0, 0);
    SET_INPUT("io_inB", 1, 1);
    SET_INPUT("io_inB", 2, 0);
    SET_INPUT("io_inB", 3, 1);

    SET_INPUT("io_sel", 0, 0);
    processAllGates(net);
    ASSERT_OUTPUT_EQ("io_out", 0, 0);
    ASSERT_OUTPUT_EQ("io_out", 1, 0);
    ASSERT_OUTPUT_EQ("io_out", 2, 1);
    ASSERT_OUTPUT_EQ("io_out", 3, 1);
    net.tick();

    SET_INPUT("io_sel", 0, 1);
    processAllGates(net);
    ASSERT_OUTPUT_EQ("io_out", 0, 0);
    ASSERT_OUTPUT_EQ("io_out", 1, 1);
    ASSERT_OUTPUT_EQ("io_out", 2, 0);
    ASSERT_OUTPUT_EQ("io_out", 3, 1);
}

template <class NetworkBuilder>
void testFromJSONtest_addr_4bit()
{
    const std::string fileName = "test/test-addr-4bit.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto net = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(net.isValid());

    SET_INPUT("io_inA", 0, 0);
    SET_INPUT("io_inA", 1, 0);
    SET_INPUT("io_inA", 2, 1);
    SET_INPUT("io_inA", 3, 1);
    SET_INPUT("io_inB", 0, 0);
    SET_INPUT("io_inB", 1, 1);
    SET_INPUT("io_inB", 2, 0);
    SET_INPUT("io_inB", 3, 1);

    processAllGates(net);

    ASSERT_OUTPUT_EQ("io_out", 0, 0);
    ASSERT_OUTPUT_EQ("io_out", 1, 1);
    ASSERT_OUTPUT_EQ("io_out", 2, 1);
    ASSERT_OUTPUT_EQ("io_out", 3, 0);
}

template <class NetworkBuilder>
void testFromJSONtest_register_4bit()
{
    const std::string fileName = "test/test-register-4bit.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto net = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(net.isValid());

    SET_INPUT("io_in", 0, 0);
    SET_INPUT("io_in", 1, 0);
    SET_INPUT("io_in", 2, 1);
    SET_INPUT("io_in", 3, 1);

    // 1: Reset all DFFs.
    SET_INPUT("reset", 0, 1);
    processAllGates(net);
    net.tick();

    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(13)->task())) == 0);
    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(14)->task())) == 0);
    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(15)->task())) == 0);
    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(16)->task())) == 0);

    // 2: Store values into DFFs.
    SET_INPUT("reset", 0, 0);
    processAllGates(net);
    net.tick();

    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(13)->task())) == 0);
    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(14)->task())) == 0);
    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(15)->task())) == 1);
    assert(getOutput(std::dynamic_pointer_cast<
                     typename NetworkBuilder::ParamTaskTypeMem>(
               net.node(16)->task())) == 1);

    ASSERT_OUTPUT_EQ("io_out", 0, 0);
    ASSERT_OUTPUT_EQ("io_out", 1, 0);
    ASSERT_OUTPUT_EQ("io_out", 2, 0);
    ASSERT_OUTPUT_EQ("io_out", 3, 0);

    // 3: Get outputs.
    SET_INPUT("reset", 0, 0);
    processAllGates(net);
    net.tick();

    ASSERT_OUTPUT_EQ("io_out", 0, 0);
    ASSERT_OUTPUT_EQ("io_out", 1, 0);
    ASSERT_OUTPUT_EQ("io_out", 2, 1);
    ASSERT_OUTPUT_EQ("io_out", 3, 1);
}

template <class NetworkBuilder>
void testSequentialCircuit()
{
    /*
                    B               D
       reset(0) >---> ANDNOT(4) >---> DFF(2)
                        ^ A            v Q
                        |              |
                        *--< NOT(3) <--*-----> OUTPUT(1)
                                    A
    */

    NetworkBuilder builder;
    builder.INPUT(0, "reset", 0);
    builder.OUTPUT(1, "out", 0);
    builder.DFF(2);
    builder.NOT(3);
    builder.ANDNOT(4);
    builder.connect(2, 1);
    builder.connect(4, 2);
    builder.connect(2, 3);
    builder.connect(3, 4);
    builder.connect(0, 4);

    TaskNetwork net = std::move(builder);
    assert(net.isValid());

    auto dff =
        std::dynamic_pointer_cast<typename NetworkBuilder::ParamTaskTypeMem>(
            net.node(2)->task());
    auto out = get<NetworkBuilder>(net, "output", "out", 0);

    // 1:
    SET_INPUT("reset", 0, 1);
    processAllGates(net);

    // 2:
    net.tick();
    assert(getOutput(dff) == 0);
    SET_INPUT("reset", 0, 0);
    processAllGates(net);
    ASSERT_OUTPUT_EQ("out", 0, 0);

    // 3:
    net.tick();
    assert(getOutput(dff) == 1);
    processAllGates(net);
    ASSERT_OUTPUT_EQ("out", 0, 1);

    // 4:
    net.tick();
    assert(getOutput(dff) == 0);
    processAllGates(net);
    ASSERT_OUTPUT_EQ("out", 0, 0);
}

template <class NetworkBuilder>
void testFromJSONtest_counter_4bit()
{
    const std::string fileName = "test/test-counter-4bit.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto net = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(net.isValid());

    std::vector<std::array<int, 4>> outvals{{{0, 0, 0, 0},
                                             {1, 0, 0, 0},
                                             {0, 1, 0, 0},
                                             {1, 1, 0, 0},
                                             {0, 0, 1, 0},
                                             {1, 0, 1, 0},
                                             {0, 1, 1, 0},
                                             {1, 1, 1, 0},
                                             {0, 0, 0, 1},
                                             {1, 0, 0, 1},
                                             {0, 1, 0, 1},
                                             {1, 1, 0, 1},
                                             {0, 0, 1, 1},
                                             {1, 0, 1, 1},
                                             {0, 1, 1, 1},
                                             {1, 1, 1, 1}}};

    SET_INPUT("reset", 0, 1);
    processAllGates(net);

    SET_INPUT("reset", 0, 0);
    for (size_t i = 0; i < outvals.size(); i++) {
        net.tick();
        processAllGates(net);
        ASSERT_OUTPUT_EQ("io_out", 0, outvals[i][0]);
        ASSERT_OUTPUT_EQ("io_out", 1, outvals[i][1]);
        ASSERT_OUTPUT_EQ("io_out", 2, outvals[i][2]);
        ASSERT_OUTPUT_EQ("io_out", 3, outvals[i][3]);
    }
}

template <class NetworkBuilder, class TaskROM, class NormalT, class RAMNetwork,
          class ROMNetwork>
void testFromJSONdiamond_core_wo_ram_rom(RAMNetwork ramA, RAMNetwork ramB,
                                         ROMNetwork rom)
{
    assert(rom.isValid());

    const std::string fileName = "test/diamond-core-wo-ram-rom.json";
    std::ifstream ifs{fileName};
    assert(ifs);

    auto core = readNetworkFromJSON<NetworkBuilder>(ifs);
    assert(core.isValid());

    auto net =
        core.template merge<NormalT>(rom,
                                     {
                                         {"io_romAddr", 0, "addr", 0},
                                         {"io_romAddr", 1, "addr", 1},
                                         {"io_romAddr", 2, "addr", 2},
                                         {"io_romAddr", 3, "addr", 3},
                                         {"io_romAddr", 4, "addr", 4},
                                         {"io_romAddr", 5, "addr", 5},
                                         {"io_romAddr", 6, "addr", 6},
                                     },
                                     {
                                         {"rdata", 0, "io_romData", 0},
                                         {"rdata", 1, "io_romData", 1},
                                         {"rdata", 2, "io_romData", 2},
                                         {"rdata", 3, "io_romData", 3},
                                         {"rdata", 4, "io_romData", 4},
                                         {"rdata", 5, "io_romData", 5},
                                         {"rdata", 6, "io_romData", 6},
                                         {"rdata", 7, "io_romData", 7},
                                         {"rdata", 8, "io_romData", 8},
                                         {"rdata", 9, "io_romData", 9},
                                         {"rdata", 10, "io_romData", 10},
                                         {"rdata", 11, "io_romData", 11},
                                         {"rdata", 12, "io_romData", 12},
                                         {"rdata", 13, "io_romData", 13},
                                         {"rdata", 14, "io_romData", 14},
                                         {"rdata", 15, "io_romData", 15},
                                         {"rdata", 16, "io_romData", 16},
                                         {"rdata", 17, "io_romData", 17},
                                         {"rdata", 18, "io_romData", 18},
                                         {"rdata", 19, "io_romData", 19},
                                         {"rdata", 20, "io_romData", 20},
                                         {"rdata", 21, "io_romData", 21},
                                         {"rdata", 22, "io_romData", 22},
                                         {"rdata", 23, "io_romData", 23},
                                         {"rdata", 24, "io_romData", 24},
                                         {"rdata", 25, "io_romData", 25},
                                         {"rdata", 26, "io_romData", 26},
                                         {"rdata", 27, "io_romData", 27},
                                         {"rdata", 28, "io_romData", 28},
                                         {"rdata", 29, "io_romData", 29},
                                         {"rdata", 30, "io_romData", 30},
                                         {"rdata", 31, "io_romData", 31},
                                     })
            .template merge<NormalT>(ramA,
                                     {
                                         {"io_memA_writeEnable", 0, "wren", 0},
                                         {"io_memA_address", 0, "addr", 0},
                                         {"io_memA_address", 1, "addr", 1},
                                         {"io_memA_address", 2, "addr", 2},
                                         {"io_memA_address", 3, "addr", 3},
                                         {"io_memA_address", 4, "addr", 4},
                                         {"io_memA_address", 5, "addr", 5},
                                         {"io_memA_address", 6, "addr", 6},
                                         {"io_memA_address", 7, "addr", 7},
                                         {"io_memA_in", 0, "wdata", 0},
                                         {"io_memA_in", 1, "wdata", 1},
                                         {"io_memA_in", 2, "wdata", 2},
                                         {"io_memA_in", 3, "wdata", 3},
                                         {"io_memA_in", 4, "wdata", 4},
                                         {"io_memA_in", 5, "wdata", 5},
                                         {"io_memA_in", 6, "wdata", 6},
                                         {"io_memA_in", 7, "wdata", 7},
                                     },
                                     {
                                         {"rdata", 0, "io_memA_out", 0},
                                         {"rdata", 1, "io_memA_out", 1},
                                         {"rdata", 2, "io_memA_out", 2},
                                         {"rdata", 3, "io_memA_out", 3},
                                         {"rdata", 4, "io_memA_out", 4},
                                         {"rdata", 5, "io_memA_out", 5},
                                         {"rdata", 6, "io_memA_out", 6},
                                         {"rdata", 7, "io_memA_out", 7},
                                     })
            .template merge<NormalT>(ramB,
                                     {
                                         {"io_memB_writeEnable", 0, "wren", 0},
                                         {"io_memB_address", 0, "addr", 0},
                                         {"io_memB_address", 1, "addr", 1},
                                         {"io_memB_address", 2, "addr", 2},
                                         {"io_memB_address", 3, "addr", 3},
                                         {"io_memB_address", 4, "addr", 4},
                                         {"io_memB_address", 5, "addr", 5},
                                         {"io_memB_address", 6, "addr", 6},
                                         {"io_memB_address", 7, "addr", 7},
                                         {"io_memB_in", 0, "wdata", 0},
                                         {"io_memB_in", 1, "wdata", 1},
                                         {"io_memB_in", 2, "wdata", 2},
                                         {"io_memB_in", 3, "wdata", 3},
                                         {"io_memB_in", 4, "wdata", 4},
                                         {"io_memB_in", 5, "wdata", 5},
                                         {"io_memB_in", 6, "wdata", 6},
                                         {"io_memB_in", 7, "wdata", 7},
                                     },
                                     {
                                         {"rdata", 0, "io_memB_out", 0},
                                         {"rdata", 1, "io_memB_out", 1},
                                         {"rdata", 2, "io_memB_out", 2},
                                         {"rdata", 3, "io_memB_out", 3},
                                         {"rdata", 4, "io_memB_out", 4},
                                         {"rdata", 5, "io_memB_out", 5},
                                         {"rdata", 6, "io_memB_out", 6},
                                         {"rdata", 7, "io_memB_out", 7},
                                     });
    assert(net.isValid());

    // 0: 74 80                        	lsi	ra, 24
    // 2: 15 00 00                     	lw	ra, 0(ra)
    // 5: 34 31                        	lsi	sp, 3
    // 7: 1d 01 00                     	sw	sp, 0(ra)
    // a: 15 00 00                     	lw	ra, 0(ra)
    // d: 0e 00                        	js	0
    setROM(*net.template get<TaskROM>("rom", "all", 0),
           std::vector<uint8_t>{0x74, 0x80, 0x15, 0x00, 0x00, 0x34, 0x31, 0x1d,
                                0x01, 0x00, 0x15, 0x00, 0x00, 0x0e, 0x00});

    setRAM(net, std::vector<uint8_t>{
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
                    0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
                });

    SET_INPUT("reset", 0, 1);
    processAllGates(net);

    SET_INPUT("reset", 0, 0);
    for (int i = 0; i < 11; i++) {
        net.tick();
        processAllGates(net);
    }

    ASSERT_OUTPUT_EQ("io_finishFlag", 0, 1);

    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x00, 1);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x01, 1);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x02, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x03, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x04, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x05, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x06, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x07, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x08, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x09, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x0a, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x0b, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x0c, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x0d, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x0e, 0);
    ASSERT_OUTPUT_EQ("io_regOut_x0", 0x0f, 0);
}
//
#include "iyokan_plain.hpp"

void processAllGates(PlainNetwork& net,
                     std::shared_ptr<ProgressGraphMaker> graph = nullptr)
{
    processAllGates(net, std::thread::hardware_concurrency(), graph);
}

void setInput(std::shared_ptr<TaskPlainGateMem> task, int val)
{
    task->set(val);
}

void setROM(TaskPlainROM& rom, const std::vector<uint8_t>& src)
{
    for (int i = 0; i < 512 / 4; i++) {
        int val = 0;
        for (int j = 3; j >= 0; j--) {
            size_t offset = i * 4 + j;
            val = (val << 8) | (offset < src.size() ? src[offset] : 0x00);
        }
        rom.set4le(i << 2, val);
    }
}

void setRAM(PlainNetwork& net, const std::vector<uint8_t>& src)
{
    auto ramA = net.get<TaskPlainRAM>("ram", "A", 0),
         ramB = net.get<TaskPlainRAM>("ram", "B", 0);
    for (size_t i = 0; i < src.size(); i++)
        (i % 2 == 1 ? ramA : ramB)->set(i / 2, src[i]);
}

int getOutput(std::shared_ptr<TaskPlainGateMem> task)
{
    return task->get();
}

void testProgressGraphMaker()
{
    /*
                    B               D
       reset(0) >---> ANDNOT(4) >---> DFF(2)
                        ^ A            v Q
                        |              |
                        *--< NOT(3) <--*-----> OUTPUT(1)
                                    A
    */

    PlainNetworkBuilder builder;
    builder.INPUT(0, "reset", 0);
    builder.OUTPUT(1, "out", 0);
    builder.DFF(2);
    builder.NOT(3);
    builder.ANDNOT(4);
    builder.connect(2, 1);
    builder.connect(4, 2);
    builder.connect(2, 3);
    builder.connect(3, 4);
    builder.connect(0, 4);

    PlainNetwork net = std::move(builder);
    assert(net.isValid());

    auto graph = std::make_shared<ProgressGraphMaker>();

    processAllGates(net, graph);

    std::stringstream ss;
    graph->dumpDOT(ss);
    std::string dot = ss.str();
    assert(dot.find("n0 [label = \"{INPUT|reset[0]}\"]") != std::string::npos);
    assert(dot.find("n1 [label = \"{OUTPUT|out[0]}\"]") != std::string::npos);
    assert(dot.find("n2 [label = \"{DFF}\"]") != std::string::npos);
    assert(dot.find("n3 [label = \"{NOT}\"]") != std::string::npos);
    assert(dot.find("n4 [label = \"{ANDNOT}\"]") != std::string::npos);
    assert(dot.find("n2 -> n1") != std::string::npos);
    assert(dot.find("n4 -> n2") != std::string::npos);
    assert(dot.find("n2 -> n3") != std::string::npos);
    assert(dot.find("n0 -> n4") != std::string::npos);
    assert(dot.find("n3 -> n4") != std::string::npos);
}

void testDoPlainWithRAMROM()
{
    using namespace utility;

    // Prepare request packet
    writeToArchive("_test_plain_req_packet00", parseELF("test/test00.elf"));

    Options opt;
    opt.blueprint = NetworkBlueprint{"test/cahp-diamond.toml"};
    opt.inputFile = "_test_plain_req_packet00";
    opt.outputFile = "_test_plain_res_packet00";
    opt.numCycles = 8;
    opt.quiet = true;

    doPlain(opt);

    auto resPacket = readFromArchive<PlainPacket>("_test_plain_res_packet00");
    assert(u8vec2i(resPacket.bits.at("finflag")) == 1);
    assert(u8vec2i(resPacket.bits.at("reg_x0")) == 42);
    assert(resPacket.ram.at("ramB").at(12) == 42);
}

#include "iyokan_tfhepp.hpp"

class TFHEppTestHelper {
private:
    std::shared_ptr<TFHEpp::SecretKey> sk_;
    std::shared_ptr<TFHEpp::GateKey> gk_;
    std::shared_ptr<TFHEpp::CircuitKey> ck_;
    TFHEpp::TLWElvl0 zero_, one_;

private:
    TFHEppTestHelper()
    {
        sk_ = std::make_shared<TFHEpp::SecretKey>();
        gk_ = std::make_shared<TFHEpp::GateKey>(*sk_);
        zero_ = TFHEpp::bootsSymEncrypt({0}, *sk_).at(0);
        one_ = TFHEpp::bootsSymEncrypt({1}, *sk_).at(0);
    }

public:
    static TFHEppTestHelper& instance()
    {
        static TFHEppTestHelper inst;
        return inst;
    }

    void prepareCircuitKey()
    {
        ck_ = std::make_shared<TFHEpp::CircuitKey>(*sk_);
    }

    TFHEppWorkerInfo wi() const
    {
        return TFHEppWorkerInfo{TFHEpp::lweParams{}, gk_, ck_};
    }

    const std::shared_ptr<TFHEpp::SecretKey>& sk() const
    {
        return sk_;
    }

    const TFHEpp::TLWElvl0& zero() const
    {
        return zero_;
    }

    const TFHEpp::TLWElvl0& one() const
    {
        return one_;
    }

    std::string getELFAsPacketFile(const std::string& elfFilePath) const
    {
        // Read packet
        auto reqPacket = parseELF(elfFilePath).encrypt(*sk());
        // Write packet into temporary file
        static const std::string outFilePath = "_test_req_packet00";
        writeToArchive(outFilePath, reqPacket);
        return outFilePath;
    }
};

void processAllGates(TFHEppNetwork& net,
                     std::shared_ptr<ProgressGraphMaker> graph = nullptr)
{
    processAllGates(net, std::thread::hardware_concurrency(),
                    TFHEppTestHelper::instance().wi(), graph);
}

void setInput(std::shared_ptr<TaskTFHEppGateMem> task, int val)
{
    auto& h = TFHEppTestHelper::instance();
    task->set(val ? h.one() : h.zero());
}

void setROM(TaskTFHEppROMUX& rom, const std::vector<uint8_t>& src)
{
    auto& h = TFHEppTestHelper::instance();
    auto params = h.wi().params;

    for (size_t i = 0; i < 512 / (params.N / 8); i++) {
        TFHEpp::Polynomiallvl1 pmu;
        for (size_t j = 0; j < params.N; j++) {
            size_t offset = i * params.N + j;
            size_t byteOffset = offset / 8, bitOffset = offset % 8;
            uint8_t val = byteOffset < src.size()
                              ? (src[byteOffset] >> bitOffset) & 1u
                              : 0;
            pmu[j] = val ? params.μ : -params.μ;
        }
        rom.set128le(
            i * (params.N / 8),
            TFHEpp::trlweSymEncryptlvl1(pmu, params.αbk, h.sk()->key.lvl1));
    }
}

void setRAM(TFHEppNetwork& net, const std::vector<uint8_t>& src)
{
    auto& h = TFHEppTestHelper::instance();
    auto params = h.wi().params;

    for (size_t i = 0; i < src.size(); i++) {
        for (int bit = 0; bit < 8; bit++) {
            TFHEpp::Polynomiallvl1 pmu = {};
            uint8_t val = i < src.size() ? (src[i] >> bit) & 1u : 0;
            pmu[0] = val ? params.μ : -params.μ;
            TFHEpp::TRLWElvl1 trlwe =
                TFHEpp::trlweSymEncryptlvl1(pmu, params.α, h.sk()->key.lvl1);

            net.get<TaskTFHEppRAMUX>("ram", (i % 2 == 1 ? "A" : "B"), bit)
                ->set(i / 2, trlwe);
        }
    }
}

int getOutput(std::shared_ptr<TaskTFHEppGateMem> task)
{
    return TFHEpp::bootsSymDecrypt({task->get()},
                                   *TFHEppTestHelper::instance().sk())[0];
}

void testTFHEppSerialization()
{
    auto& h = TFHEppTestHelper::instance();
    const std::shared_ptr<const TFHEpp::SecretKey>& sk = h.sk();
    const std::shared_ptr<const TFHEpp::GateKey>& gk = h.wi().gateKey;

    // Test for secret key
    {
        // Dump
        writeToArchive("_test_sk", *sk);
        // Load
        auto sk2 = std::make_shared<TFHEpp::SecretKey>();
        readFromArchive<TFHEpp::SecretKey>(*sk2, "_test_sk");

        auto zero = TFHEpp::bootsSymEncrypt({0}, *sk2).at(0);
        auto one = TFHEpp::bootsSymEncrypt({1}, *sk2).at(0);
        TFHEpp::TLWElvl0 res;
        TFHEpp::HomANDNY(res, zero, one, *gk);
        assert(TFHEpp::bootsSymDecrypt({res}, *sk2).at(0) == 1);
    }

    // Test for gate key
    {
        std::stringstream ss{std::ios::binary | std::ios::out | std::ios::in};

        // Dump
        writeToArchive(ss, *gk);
        // Load
        auto gk2 = std::make_shared<TFHEpp::GateKey>();
        readFromArchive<TFHEpp::GateKey>(*gk2, ss);

        auto zero = TFHEpp::bootsSymEncrypt({0}, *sk).at(0);
        auto one = TFHEpp::bootsSymEncrypt({1}, *sk).at(0);
        TFHEpp::TLWElvl0 res;
        TFHEpp::HomANDNY(res, zero, one, *gk2);
        assert(TFHEpp::bootsSymDecrypt({res}, *sk).at(0) == 1);
    }

    // Test for TLWE level 0
    {
        std::stringstream ss{std::ios::binary | std::ios::out | std::ios::in};

        {
            auto zero = TFHEpp::bootsSymEncrypt({0}, *sk).at(0);
            auto one = TFHEpp::bootsSymEncrypt({1}, *sk).at(0);
            writeToArchive(ss, zero);
            writeToArchive(ss, one);
            ss.seekg(0);
        }

        {
            TFHEpp::TLWElvl0 res, zero, one;
            readFromArchive(zero, ss);
            readFromArchive(one, ss);
            TFHEpp::HomANDNY(res, zero, one, *gk);
            assert(TFHEpp::bootsSymDecrypt({res}, *sk).at(0) == 1);
        }
    }
}

void testDoTFHEWithRAMROM()
{
    using namespace utility;
    auto& h = TFHEppTestHelper::instance();

    Options opt;
    opt.blueprint = NetworkBlueprint{"test/cahp-diamond.toml"};
    opt.inputFile = h.getELFAsPacketFile("test/test00.elf");
    opt.outputFile = "_test_res_packet00";
    opt.numCycles = 8;

    doTFHE(opt);

    writeToArchive("_test_sk", *h.sk());
    auto resPacket = readFromArchive<TFHEPacket>("_test_res_packet00");
    auto plainResPacket = resPacket.decrypt(*h.sk());

    assert(u8vec2i(plainResPacket.bits.at("finflag")) == 1);
    assert(u8vec2i(plainResPacket.bits.at("reg_x0")) == 42);
    assert(plainResPacket.ram.at("ramB").at(12) == 42);
}

#ifdef IYOKAN_CUDA_ENABLED
#include "iyokan_cufhe.hpp"

class CUFHETestHelper {
private:
    std::shared_ptr<cufhe::PriKey> sk_;
    std::shared_ptr<cufhe::PubKey> gk_;
    cufhe::Ctxt zero_, one_;

private:
    CUFHETestHelper()
    {
        cufhe::SetSeed();

        sk_ = std::make_shared<cufhe::PriKey>();
        gk_ = std::make_shared<cufhe::PubKey>();
        cufhe::KeyGen(*gk_, *sk_);

        cufhe::Ptxt p;
        p = 0;
        cufhe::Encrypt(zero_, p, *sk_);
        p = 1;
        cufhe::Encrypt(one_, p, *sk_);
    }

public:
    class CUFHEManager {
    public:
        CUFHEManager()
        {
            cufhe::Initialize(*CUFHETestHelper::instance().gk_);
        }

        ~CUFHEManager()
        {
            cufhe::CleanUp();
        }
    };

public:
    static CUFHETestHelper& instance()
    {
        static CUFHETestHelper inst;
        return inst;
    }

    const std::shared_ptr<cufhe::PriKey>& sk() const
    {
        return sk_;
    }

    const cufhe::Ctxt& zero() const
    {
        return zero_;
    }

    const cufhe::Ctxt& one() const
    {
        return one_;
    }
};

void processAllGates(CUFHENetwork& net,
                     std::shared_ptr<ProgressGraphMaker> graph = nullptr)
{
    processAllGates(net, 240, graph);
}

void setInput(std::shared_ptr<TaskCUFHEGateMem> task, int val)
{
    auto& h = CUFHETestHelper::instance();
    task->set(val ? h.one() : h.zero());
}

int getOutput(std::shared_ptr<TaskCUFHEGateMem> task)
{
    cufhe::Ptxt p;
    cufhe::Decrypt(p, task->get(), *CUFHETestHelper::instance().sk());
    return p.get();
}

void testDoCUFHEWithRAMROM()
{
    using namespace utility;
    auto& h = TFHEppTestHelper::instance();

    Options opt;
    opt.blueprint = NetworkBlueprint{"test/cahp-diamond.toml"};
    opt.inputFile = h.getELFAsPacketFile("test/test00.elf");
    opt.outputFile = "_test_res_packet00";
    opt.numCycles = 8;

    doCUFHE(opt);

    writeToArchive("_test_sk", *h.sk());
    auto resPacket = readFromArchive<TFHEPacket>("_test_res_packet00");
    auto plainResPacket = resPacket.decrypt(*h.sk());
    assert(u8vec2i(plainResPacket.bits.at("finflag")) == 1);
    assert(u8vec2i(plainResPacket.bits.at("reg_x0")) == 42);
    assert(plainResPacket.ram.at("ramB").at(12) == 42);
}

void testBridgeBetweenCUFHEAndTFHEpp()
{
    auto& ht = TFHEppTestHelper::instance();

    NetworkBuilderBase<CUFHEWorkerInfo> b0;
    NetworkBuilderBase<TFHEppWorkerInfo> b1;
    auto t0 =
        b0.addINPUT<TaskCUFHEGateWIRE>(detail::genid(), 0, "in", 0, false);
    auto t1 = std::make_shared<TaskCUFHE2TFHEpp>();
    b1.addTask(NodeLabel{detail::genid(), "cufhe2tfhepp", ""}, 0, t1);
    auto t2 = std::make_shared<TaskTFHEpp2CUFHE>();
    b1.addTask(NodeLabel{detail::genid(), "tfhepp2cufhe", ""}, 0, t2);
    auto t3 =
        b0.addOUTPUT<TaskCUFHEGateWIRE>(detail::genid(), 0, "out", 0, true);
    b0.connectTasks(t1, t2);

    auto net0 = std::make_shared<TaskNetwork<CUFHEWorkerInfo>>(std::move(b0));
    auto net1 = std::make_shared<TaskNetwork<TFHEppWorkerInfo>>(std::move(b1));
    auto bridge0 = connectWithBridge(t0, t1);
    auto bridge1 = connectWithBridge(t2, t3);

    CUFHENetworkRunner runner{1, 1, ht.wi()};
    runner.addNetwork(net0);
    runner.addNetwork(net1);
    runner.addBridge(bridge0);
    runner.addBridge(bridge1);

    t0->set(*tfhepp2cufhe(ht.one()));
    runner.run();
    assert(cufhe2tfhepp(t3->get()) == ht.one());

    net0->tick();
    net1->tick();
    bridge0->tick();
    bridge1->tick();

    t0->set(*tfhepp2cufhe(ht.zero()));
    runner.run();
    assert(cufhe2tfhepp(t3->get()) == ht.zero());
}
#endif

void testBlueprint()
{
    using namespace blueprint;

    NetworkBlueprint blueprint{"test/cahp-diamond.toml"};

    {
        const auto& files = blueprint.files();
        assert(files.size() == 1);
        assert(files[0].type == File::TYPE::IYOKANL1_JSON);
        assert(files[0].path == "test/diamond-core-wo-ram-rom.json");
        assert(files[0].name == "core");
    }

    {
        const auto& roms = blueprint.builtinROMs();
        assert(roms.size() == 1);
        assert(roms[0].name == "rom");
        assert(roms[0].inAddrWidth == 7);
        assert(roms[0].outRdataWidth == 32);
    }

    {
        const auto& rams = blueprint.builtinRAMs();
        assert(rams.size() == 2);
        assert(rams[0].name == "ramA" && rams[1].name == "ramB" ||
               rams[1].name == "ramA" && rams[0].name == "ramB");
        assert(rams[0].inAddrWidth == 8);
        assert(rams[0].inWdataWidth == 8);
        assert(rams[0].outRdataWidth == 8);
        assert(rams[1].inAddrWidth == 8);
        assert(rams[1].inWdataWidth == 8);
        assert(rams[1].outRdataWidth == 8);
    }

    {
        const auto& edges = blueprint.edges();
        auto assertIn = [&edges](std::string fNodeName, std::string fPortName,
                                 std::string tNodeName, std::string tPortName,
                                 int size) {
            for (int i = 0; i < size; i++) {
                auto v =
                    std::make_pair(Port{fNodeName, {"output", fPortName, i}},
                                   Port{tNodeName, {"input", tPortName, i}});
                auto it = std::find(edges.begin(), edges.end(), v);
                assert(it != edges.end());
            }
        };
        assertIn("core", "io_romAddr", "rom", "addr", 7);
        assertIn("rom", "rdata", "core", "io_romData", 32);
        assertIn("core", "io_memA_writeEnable", "ramA", "wren", 1);
        assertIn("core", "io_memA_address", "ramA", "addr", 8);
        assertIn("core", "io_memA_in", "ramA", "wdata", 8);
        assertIn("ramA", "rdata", "core", "io_memA_out", 8);
        assertIn("core", "io_memB_writeEnable", "ramB", "wren", 1);
        assertIn("core", "io_memB_address", "ramB", "addr", 8);
        assertIn("core", "io_memB_in", "ramB", "wdata", 8);
        assertIn("ramB", "rdata", "core", "io_memB_out", 8);
    }

    {
        const Port& port = blueprint.at("reset").value();
        assert(port.nodeName == "core");
        assert(port.portLabel.kind == "input");
        assert(port.portLabel.portName == "reset");
        assert(port.portLabel.portBit == 0);
    }

    {
        const Port& port = blueprint.at("finflag").value();
        assert(port.nodeName == "core");
        assert(port.portLabel.kind == "output");
        assert(port.portLabel.portName == "io_finishFlag");
        assert(port.portLabel.portBit == 0);
    }

    for (int ireg = 0; ireg < 16; ireg++) {
        for (int ibit = 0; ibit < 16; ibit++) {
            const Port& port =
                blueprint.at(utility::fok("reg_x", ireg), ibit).value();
            assert(port.nodeName == "core");
            assert(port.portLabel.portName ==
                   utility::fok("io_regOut_x", ireg));
            assert(port.portLabel.portBit == ibit);
        }
    }
}

int main(int argc, char** argv)
{
    AsyncThread::setNumThreads(std::thread::hardware_concurrency());

    testNOT<PlainNetworkBuilder>();
    testMUX<PlainNetworkBuilder>();
    testBinopGates<PlainNetworkBuilder>();
    testFromJSONtest_pass_4bit<PlainNetworkBuilder>();
    testFromJSONtest_and_4bit<PlainNetworkBuilder>();
    testFromJSONtest_and_4_2bit<PlainNetworkBuilder>();
    testFromJSONtest_mux_4bit<PlainNetworkBuilder>();
    testFromJSONtest_addr_4bit<PlainNetworkBuilder>();
    testFromJSONtest_register_4bit<PlainNetworkBuilder>();
    testSequentialCircuit<PlainNetworkBuilder>();
    testFromJSONtest_counter_4bit<PlainNetworkBuilder>();
    testFromJSONdiamond_core_wo_ram_rom<PlainNetworkBuilder, TaskPlainROM,
                                        uint8_t>(makePlainRAMNetwork("A"),
                                                 makePlainRAMNetwork("B"),
                                                 makePlainROMNetwork());
    testDoPlainWithRAMROM();

    testNOT<TFHEppNetworkBuilder>();
    testMUX<TFHEppNetworkBuilder>();
    testBinopGates<TFHEppNetworkBuilder>();
    testFromJSONtest_pass_4bit<TFHEppNetworkBuilder>();
    testFromJSONtest_pass_4bit<TFHEppNetworkBuilder>();
    testFromJSONtest_and_4bit<TFHEppNetworkBuilder>();
    testFromJSONtest_and_4_2bit<TFHEppNetworkBuilder>();
    testFromJSONtest_mux_4bit<TFHEppNetworkBuilder>();
    testFromJSONtest_addr_4bit<TFHEppNetworkBuilder>();
    testFromJSONtest_register_4bit<TFHEppNetworkBuilder>();
    testSequentialCircuit<TFHEppNetworkBuilder>();
    testFromJSONtest_counter_4bit<TFHEppNetworkBuilder>();
    testTFHEppSerialization();

#ifdef IYOKAN_CUDA_ENABLED
    {
        CUFHETestHelper::CUFHEManager man;

        testNOT<CUFHENetworkBuilder>();
        testMUX<CUFHENetworkBuilder>();
        testBinopGates<CUFHENetworkBuilder>();
        testFromJSONtest_pass_4bit<CUFHENetworkBuilder>();
        testFromJSONtest_and_4bit<CUFHENetworkBuilder>();
        testFromJSONtest_and_4_2bit<CUFHENetworkBuilder>();
        testFromJSONtest_mux_4bit<CUFHENetworkBuilder>();
        testFromJSONtest_addr_4bit<CUFHENetworkBuilder>();
        testFromJSONtest_register_4bit<CUFHENetworkBuilder>();
        testSequentialCircuit<CUFHENetworkBuilder>();
        testFromJSONtest_counter_4bit<CUFHENetworkBuilder>();
        testBridgeBetweenCUFHEAndTFHEpp();
    }
#endif

    testProgressGraphMaker();
    testBlueprint();

    if (argc >= 2 && strcmp(argv[1], "slow") == 0) {
        TFHEppTestHelper::instance().prepareCircuitKey();

        testFromJSONdiamond_core_wo_ram_rom<TFHEppNetworkBuilder,
                                            TaskTFHEppROMUX, TFHEpp::TLWElvl0>(
            makeTFHEppRAMNetwork("A"), makeTFHEppRAMNetwork("B"),
            makeTFHEppROMNetwork());
        testDoTFHEWithRAMROM();

#ifdef IYOKAN_CUDA_ENABLED
        testDoCUFHEWithRAMROM();
#endif
    }
}
