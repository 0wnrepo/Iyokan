#ifndef VIRTUALSECUREPLATFORM_IYOKAN_PACKET_HPP
#define VIRTUALSECUREPLATFORM_IYOKAN_PACKET_HPP

#include <fstream>
#include <iomanip>
#include <memory>
#include <regex>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <tfhe++.hpp>

#include "error.hpp"
#include "utility.hpp"

enum class Bit : bool {};
inline constexpr Bit operator~(Bit l) noexcept
{
    return Bit(~static_cast<int>(l));
}
inline constexpr Bit operator|(Bit l, Bit r) noexcept
{
    return Bit(static_cast<int>(l) | static_cast<int>(r));
}
inline constexpr Bit operator&(Bit l, Bit r) noexcept
{
    return Bit(static_cast<int>(l) & static_cast<int>(r));
}
inline constexpr Bit operator^(Bit l, Bit r) noexcept
{
    return Bit(static_cast<int>(l) ^ static_cast<int>(r));
}
inline constexpr Bit operator|=(Bit& l, Bit r) noexcept
{
    return l = l | r;
}
inline constexpr Bit operator&=(Bit& l, Bit r) noexcept
{
    return l = l & r;
}
inline constexpr Bit operator^=(Bit& l, Bit r) noexcept
{
    return l = l ^ r;
}
inline Bit operator"" _b(unsigned long long x)
{
    return Bit(x != 0);
}

inline uint64_t bitvec2i(const std::vector<Bit>& src, int start = 0,
                         int end = -1)
{
    if (end == -1)
        end = src.size();
    assert(end - start < 64);
    uint64_t ret = 0;
    for (size_t i = start; i < end; i++)
        ret |= (static_cast<size_t>(src.at(i)) << (i - start));
    return ret;
}

namespace TFHEpp {
template <class Archive>
void serialize(Archive& ar, lweParams& src)
{
    ar(src.n, src.α, src.Nbit, src.N, src.l, src.Bgbit, src.Bg, src.αbk, src.t,
       src.basebit, src.αks, src.μ, src.nbarbit, src.nbar, src.lbar,
       src.Bgbitbar, src.Bgbar, src.αbklvl02, src.tbar, src.basebitlvl21,
       src.αprivks, src.μbar);
}

template <class Archive>
void serialize(Archive& ar, lweKey& src)
{
    ar(src.lvl0, src.lvl1, src.lvl2);
}

template <class Archive>
void serialize(Archive& ar, SecretKey& src)
{
    ar(src.key, src.params);
}

template <class Archive>
void serialize(Archive& ar, GateKey& src)
{
    ar(src.ksk, src.bkfftlvl01);
}

template <class Archive>
void serialize(Archive& ar, CircuitKey& src)
{
    ar(src.privksk, src.bkfftlvl02);
}
}  // namespace TFHEpp

inline std::vector<TFHEpp::TLWElvl0> encryptBits(const TFHEpp::SecretKey& key,
                                                 const std::vector<Bit>& src)
{
    std::vector<uint8_t> in;
    in.reserve(src.size());
    for (auto&& bit : src)
        in.push_back(bit == 1_b ? 1 : 0);
    return TFHEpp::bootsSymEncrypt(in, key);
}

inline std::vector<TFHEpp::TRLWElvl1> encryptROM(const TFHEpp::SecretKey& key,
                                                 const std::vector<Bit>& src)
{
    const TFHEpp::lweParams& params = key.params;
    std::vector<TFHEpp::TRLWElvl1> ret;

    TFHEpp::Polynomiallvl1 pmu = {};
    for (size_t i = 0; i < src.size(); i++) {
        pmu[i % params.N] = src[i] == 1_b ? params.μ : -params.μ;
        if (i % params.N == params.N - 1)
            ret.push_back(
                TFHEpp::trlweSymEncryptlvl1(pmu, params.αbk, key.key.lvl1));
    }

    return ret;
}

inline std::vector<TFHEpp::TLWElvl0> encryptROMInTLWE(
    const TFHEpp::SecretKey& key, const std::vector<Bit>& src)
{
    return encryptBits(key, src);
}

inline std::vector<TFHEpp::TRLWElvl1> encryptRAM(const TFHEpp::SecretKey& key,
                                                 const std::vector<Bit>& src)
{
    const TFHEpp::lweParams& params = key.params;
    std::vector<TFHEpp::TRLWElvl1> ret;

    for (auto&& bit : src) {
        TFHEpp::Polynomiallvl1 pmu = {};
        pmu[0] = bit == 1_b ? params.μ : -params.μ;
        ret.push_back(TFHEpp::trlweSymEncryptlvl1(pmu, params.α, key.key.lvl1));
    }

    return ret;
}

inline std::vector<TFHEpp::TLWElvl0> encryptRAMInTLWE(
    const TFHEpp::SecretKey& key, const std::vector<Bit>& src)
{
    return encryptBits(key, src);
}

inline std::vector<uint8_t> decrypt(const TFHEpp::SecretKey& key,
                                    const std::vector<TFHEpp::TLWElvl0>& src)
{
    std::vector<uint8_t> ret;
    for (auto it = src.begin(); it != src.end();) {
        uint8_t byte = 0;
        for (uint32_t i = 0; i < 8; i++, ++it) {
            assert(it != src.end());
            uint8_t val = TFHEpp::bootsSymDecrypt(std::vector{*it}, key).at(0);
            byte |= (val & 1u) << i;
        }
        ret.push_back(byte);
    }
    return ret;
}

inline std::vector<Bit> decryptBits(const TFHEpp::SecretKey& key,
                                    const std::vector<TFHEpp::TLWElvl0>& src)
{
    auto bitvals = TFHEpp::bootsSymDecrypt(src, key);
    std::vector<Bit> bits;
    for (auto&& bitval : bitvals)
        bits.push_back(bitval != 0 ? 1_b : 0_b);
    return bits;
}

inline std::vector<Bit> decryptRAM(const TFHEpp::SecretKey& key,
                                   const std::vector<TFHEpp::TRLWElvl1>& src)
{
    std::vector<Bit> ret;
    for (auto&& encbit : src) {
        uint8_t bitval =
            TFHEpp::trlweSymDecryptlvl1(encbit, key.key.lvl1).at(0);
        ret.push_back(bitval != 0 ? 1_b : 0_b);
    }

    return ret;
}

inline std::vector<Bit> decryptRAMInTLWE(
    const TFHEpp::SecretKey& key, const std::vector<TFHEpp::TLWElvl0>& src)
{
    return decryptBits(key, src);
}

inline std::vector<Bit> decryptROM(const TFHEpp::SecretKey& key,
                                   const std::vector<TFHEpp::TRLWElvl1>& src)
{
    std::vector<Bit> ret;
    for (auto&& encblk : src) {
        auto blk = TFHEpp::trlweSymDecryptlvl1(encblk, key.key.lvl1);
        for (uint8_t bitval : blk)
            ret.push_back(bitval != 0 ? 1_b : 0_b);
    }

    return ret;
}

inline std::vector<Bit> decryptROMInTLWE(
    const TFHEpp::SecretKey& key, const std::vector<TFHEpp::TLWElvl0>& src)
{
    return decryptBits(key, src);
}

struct TFHEPacket;

struct PlainPacket {
    std::unordered_map<std::string, std::vector<Bit>> ram;
    std::unordered_map<std::string, std::vector<Bit>> rom;
    std::unordered_map<std::string, std::vector<Bit>> bits;
    std::optional<int> numCycles;

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(ram, rom, bits, numCycles);
    }

    inline TFHEPacket encrypt(const TFHEpp::SecretKey& key) const;
};

struct TFHEPacket {
    std::shared_ptr<TFHEpp::GateKey> gk;
    std::shared_ptr<TFHEpp::CircuitKey> ck;
    std::unordered_map<std::string, std::vector<TFHEpp::TRLWElvl1>> ram;
    std::unordered_map<std::string, std::vector<TFHEpp::TLWElvl0>> ramInTLWE;
    std::unordered_map<std::string, std::vector<TFHEpp::TRLWElvl1>> rom;
    std::unordered_map<std::string, std::vector<TFHEpp::TLWElvl0>> romInTLWE;
    std::unordered_map<std::string, std::vector<TFHEpp::TLWElvl0>> bits;
    std::optional<int> numCycles;

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(gk, ck, ram, ramInTLWE, rom, romInTLWE, bits, numCycles);
    }

    inline PlainPacket decrypt(const TFHEpp::SecretKey& key) const;
};

TFHEPacket PlainPacket::encrypt(const TFHEpp::SecretKey& key) const
{
    TFHEPacket tfhe{std::make_shared<TFHEpp::GateKey>(key),
                    std::make_shared<TFHEpp::CircuitKey>(key),
                    {},
                    {},
                    {},
                    {},
                    {},
                    numCycles};
    // Encrypt RAM
    for (auto&& [name, src] : ram) {
        if (auto [it, inserted] = tfhe.ram.emplace(name, encryptRAM(key, src));
            !inserted)
            error::die("Invalid PlainPacket. Duplicate ram's key: ", name);
        if (auto [it, inserted] =
                tfhe.ramInTLWE.emplace(name, encryptRAMInTLWE(key, src));
            !inserted)
            error::die("Invalid PlainPacket. Duplicate ram's key: ", name);
    }

    // Encrypt ROM
    for (auto&& [name, src] : rom) {
        if (auto [it, inserted] = tfhe.rom.emplace(name, encryptROM(key, src));
            !inserted)
            error::die("Invalid PlainPacket. Duplicate rom's key: ", name);
        if (auto [it, inserted] =
                tfhe.romInTLWE.emplace(name, encryptROMInTLWE(key, src));
            !inserted)
            error::die("Invalid PlainPacket. Duplicate rom's key: ", name);
    }

    // Encrypt bits
    for (auto&& [name, src] : bits) {
        auto [it, inserted] = tfhe.bits.emplace(name, encryptBits(key, src));
        if (!inserted)
            error::die("Invalid PlainPacket. Duplicate bits's key: ", name);
    }

    return tfhe;
}

PlainPacket TFHEPacket::decrypt(const TFHEpp::SecretKey& key) const
{
    PlainPacket plain{{}, {}, {}, numCycles};

    // Decrypt RAM
    for (auto&& [name, trlwes] : ram)
        plain.ram.emplace(name, decryptRAM(key, trlwes));
    for (auto&& [name, tlwes] : ramInTLWE)
        plain.ram.emplace(name, decryptRAMInTLWE(key, tlwes));

    // Decrypt ROM
    for (auto&& [name, trlwes] : rom)
        plain.rom.emplace(name, decryptROM(key, trlwes));
    for (auto&& [name, tlwes] : romInTLWE)
        plain.rom.emplace(name, decryptROMInTLWE(key, tlwes));

    // Decrypt bits
    for (auto&& [name, tlwes] : bits) {
        auto [it, inserted] = plain.bits.emplace(name, decryptBits(key, tlwes));
        if (!inserted)
            error::die("Invalid TFHEPacket. Duplicate bits's key: ", name);
    }

    return plain;
}

template <class T>
void readFromArchive(T& res, std::istream& is)
{
    cereal::PortableBinaryInputArchive ar{is};
    ar(res);
}

template <class T>
void readFromArchive(T& res, const std::string& path)
{
    try {
        std::ifstream ifs{path, std::ios::binary};
        assert(ifs && "Can't open the file to read from; maybe not found?");
        readFromArchive<T>(res, ifs);
    }
    catch (std::exception& ex) {
        error::die("Invalid archive: ", path);
    }
}

template <class T>
T readFromArchive(std::istream& is)
{
    T ret;
    readFromArchive(ret, is);
    return ret;
}

template <class T>
T readFromArchive(const std::string& path)
{
    T ret;
    readFromArchive(ret, path);
    return ret;
}

template <class T>
void writeToArchive(std::ostream& os, const T& src)
{
    cereal::PortableBinaryOutputArchive ar{os};
    ar(src);
}

template <class T>
void writeToArchive(const std::string& path, const T& src)
{
    try {
        std::ofstream ofs{path, std::ios::binary};
        assert(ofs && "Can't open the file to write in; maybe not allowed?");
        return writeToArchive(ofs, src);
    }
    catch (std::exception& ex) {
        error::die("Unable to write into archive: ", path);
    }
}

#endif
