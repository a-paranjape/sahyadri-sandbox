// make_testset: writes a version-3 snapcomp snapshot with the reference library (pkdgrav3 io/snapcomp),
// for testing the C99 reader io/pkc_core.c. Not part of Rockstar.
//
// usage: make_testset out_prefix N ids(lattice|raw|delta) writers pot(0|1) block_size [seed]
//   N^3 particles on a lattice in a 50 Mpc/h box, displaced by a few cells, 9 Sahyadri subsamples,
//   w0-wa cosmology (Omega_lambda = 0, Omega_DE = 0.69, w0 = -0.9, wa = 0.1), z = 0.5.
#include "snapcomp.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace snapcomp;

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: make_testset out_prefix N ids(lattice|raw|delta) writers pot(0|1) block_size [seed]\n");
        return 2;
    }
    const std::string out = argv[1];
    const uint64_t N = std::stoull(argv[2]);
    const std::string idmode = argv[3];
    const uint32_t W = uint32_t(std::stoul(argv[4]));
    const bool pot = std::stoi(argv[5]) != 0;
    const uint32_t B = uint32_t(std::stoul(argv[6]));
    const uint64_t seed = argc > 7 ? std::stoull(argv[7]) : 1;
    const uint64_t n = N * N * N;
    const double L = 50.0, z = 0.5, a = 1.0 / (1.0 + z);
    const double Om = 0.31, Ode = 0.69, w0 = -0.9, wa = 0.1, hh = 0.68;
    const double percents[9] = {1, 3, 5, 9, 10, 12, 15, 20, 25};

    Header h;
    h.n_total = n;
    h.lattice_n = idmode == "lattice" ? N : 0;
    choose_grid(L * 1000.0, 5.0, n, h.ncell, h.sub); // 5 kpc/h position step
    h.vel_step = 2.44;
    h.vel_to_kms = 1.0;
    h.box_size = L;
    h.redshift = z;
    h.time = 0.123;
    h.omega_m = Om;
    h.omega_l = 0.0;
    h.h = hh;
    h.n_subsamples = 9;
    for (int k = 0; k < 9; ++k) h.percents[k] = percents[k];
    h.subsample_seed = 7;
    h.n_writers = W;
    h.block_size = B;
    h.step = 100;
    h.has_cosmology = true;
    h.omega_b = 0.049, h.omega_rad = 0.0, h.omega_de = Ode, h.w0 = w0, h.wa = wa;
    h.sigma8 = 0.81, h.n_s = 0.965, h.running = 0, h.normalization = 0, h.pivot = 0.05;
    if (pot) h.pot_step = 100.0, h.pot_to_kms2 = 1.0;
    // E(z) as pkdgrav3 computes it for this background
    const double rhoDE = Ode * std::pow(a, -3.0 * (1.0 + w0 + wa)) * std::exp(-3.0 * wa * (1.0 - a));
    const double Ez = std::sqrt(Om / (a * a * a) + rhoDE);
    char info[512];
    std::snprintf(info, sizeof info, "{\"code\":\"make_testset\",\"cosmology\":{\"Omega_m\":%.10g,\"Omega_DE\":%.10g,\"w0\":%.10g,\"wa\":%.10g,\"E_z\":%.10g}}",
                  Om, Ode, w0, wa, Ez);
    h.info = info;

    // particles
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> disp(0.0, 2.0 / double(N)), vel(0.0, 300.0), phi(-1e5, 3e4);
    std::vector<double> pos(3 * n);
    std::vector<float> v(3 * n), P(n);
    std::vector<uint64_t> ids(n);
    std::vector<uint32_t> writer(n);
    for (uint64_t iz = 0, i = 0; iz < N; ++iz)
        for (uint64_t iy = 0; iy < N; ++iy)
            for (uint64_t ix = 0; ix < N; ++ix, ++i) {
                const uint64_t q[3] = {ix, iy, iz};
                for (int d = 0; d < 3; ++d) {
                    double x = (double(q[d]) + 0.5) / double(N) - 0.5 + disp(rng);
                    x -= std::floor(x + 0.5); // wrap into [-0.5, 0.5)
                    pos[3 * i + d] = x;
                    v[3 * i + d] = float(vel(rng));
                }
                P[i] = float(phi(rng));
                ids[i] = idmode == "raw" ? rng() >> 1 : i; // "delta": index IDs without lattice_n
                writer[i] = uint32_t(ix * W / N);
            }

    SubsampleTable table(h.percents, 9, h.subsample_seed);
    std::vector<int> sub(n);
    for (uint64_t i = 0; i < n; ++i) sub[i] = table(ids[i]);

    uint64_t written = 0;
    for (uint32_t w = 0; w < W; ++w)
        for (int k = 0; k < 9; ++k) {
            Header hk = h;
            hk.subsample = uint32_t(k);
            hk.writer = w;
            char pct[16];
            std::snprintf(pct, sizeof pct, "%02d", int(percents[k]));
            FileWriter fw(out + ".s" + pct + "." + std::to_string(w) + ".pkc", hk);
            BlockEncoder enc(hk);
            std::vector<double> bp;
            std::vector<float> bv, bP;
            std::vector<uint64_t> bi;
            auto flush = [&]() {
                if (bi.empty()) return;
                auto& bytes = enc.encode(bi.size(), bp.data(), bv.data(), bi.data(), pot ? bP.data() : nullptr);
                fw.append(bytes.data(), bytes.size(), uint32_t(bi.size()));
                written += bi.size();
                bp.clear(), bv.clear(), bP.clear(), bi.clear();
            };
            for (uint64_t i = 0; i < n; ++i) {
                if (writer[i] != w || sub[i] != k) continue;
                for (int d = 0; d < 3; ++d) bp.push_back(pos[3 * i + d]), bv.push_back(v[3 * i + d]);
                bP.push_back(P[i]);
                bi.push_back(ids[i]);
                if (bi.size() == B) flush();
            }
            flush();
            fw.close();
        }
    std::printf("wrote %s: %llu particles, %u writers, ids %s, potential %d, block %u, grid %ux%u, E_z %.10g\n", out.c_str(),
                (unsigned long long)written, W, idmode.c_str(), int(pot), B, h.ncell, h.sub, Ez);
    return written == n ? 0 : 1;
}
