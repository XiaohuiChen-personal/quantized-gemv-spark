// Host-side helpers: BF16/FP16 bit conversions, file loading, the data manifest.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------- number formats

// BF16 is the top 16 bits of an IEEE float32, so widening is a shift.
inline float bf16_to_float(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Round-to-nearest-even narrowing (inputs here are finite).
inline uint16_t float_to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// IEEE binary16 -> float, including subnormals (scales can be subnormal).
inline float half_to_float(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    float v;
    if (exp == 0)
        v = std::ldexp((float)man, -24);                       // subnormal or zero
    else if (exp == 31)
        v = man ? NAN : INFINITY;
    else
        v = std::ldexp((float)(man | 0x400u), (int)exp - 25);  // 1.man x 2^(exp-15)
    return sign ? -v : v;
}

// ---------------------------------------------------------------- files

inline std::vector<uint8_t> read_file(const std::string& path, size_t expected_bytes)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    const size_t n = (size_t)f.tellg();
    if (n != expected_bytes) {
        fprintf(stderr, "%s: %zu bytes, expected %zu\n", path.c_str(), n, expected_bytes);
        exit(EXIT_FAILURE);
    }
    std::vector<uint8_t> buf(n);
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)n)) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    return buf;
}

// ---------------------------------------------------------------- manifest

// One line of data/manifest.tsv (written by scripts/export_weights.py):
// id  tensor_name  M  K  role
struct MatrixInfo {
    std::string id, tensor, role;
    int M = 0, K = 0;
};

inline std::vector<MatrixInfo> read_manifest(const std::string& path)
{
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open %s (run `make data` first)\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    std::vector<MatrixInfo> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        MatrixInfo m;
        if (!(ss >> m.id >> m.tensor >> m.M >> m.K >> m.role)) {
            fprintf(stderr, "bad manifest line: %s\n", line.c_str());
            exit(EXIT_FAILURE);
        }
        out.push_back(m);
    }
    return out;
}
