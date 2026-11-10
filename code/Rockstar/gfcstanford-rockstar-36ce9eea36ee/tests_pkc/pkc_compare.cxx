// pkc_compare: tests the C99 reader io/pkc_core.c against the reference library (pkdgrav3 io/snapcomp).
// Not part of Rockstar.
//
// usage: pkc_compare decode  <prefix>                  every block of every file: C == C++ (X, V, ids)
//        pkc_compare shares  <prefix> NB [NB ...]      every ID once over the NB shares; balance
//        pkc_compare subsel  <prefix> pct[,pct...]     selected subsamples only, membership by ID hash
//        pkc_compare cosmo   <prefix> Ol W0 WA Ez      Rockstar cosmology and E_z from the header
//        pkc_compare gadget  <prefix> <gadget_prefix>  per ID: Rockstar's floats from pkc_to_gadget files
//                                                      == the native reader's floats (bit for bit)
//        pkc_compare blocks  <prefix> <gadget_prefix> K Gadget file k (pkc_to_gadget --blocks K) holds the IDs
//                                                      of native share k, in the same order
//        pkc_compare open    <prefix>                  open + decode everything; prints the error, exit 1
#include "snapcomp.h"

extern "C" {
#include "pkc_core.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static char err[PKC_ERRLEN];

static void need(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

static pkc_set open_set(const std::string& prefix, const std::vector<double>& want = {}) {
    pkc_set s;
    need(pkc_set_open(prefix.c_str(), want.data(), int(want.size()), &s, err) == 0, err);
    return s;
}

static int cmd_decode(const std::string& prefix) {
    pkc_set s = open_set(prefix);
    pkc_decoder d;
    pkc_decoder_init(&d);
    pkc_block b = {};
    uint64_t blocks = 0, parts = 0, coders[3] = {0, 0, 0};
    for (int f = 0; f < s.n_files; ++f) {
        snapcomp::FileReader ref(s.files[f].path);
        need(ref.n_blocks() == s.files[f].h.n_blocks, "block count differs");
        snapcomp::Block rb;
        for (size_t i = 0; i < ref.n_blocks(); ++i) {
            ref.read_block(i, rb);
            need(pkc_read_block(&s.files[f], i, &d, &b, err) == 0, err);
            need(b.n == rb.size() && b.id_coder == rb.id_coder, "block size or coder differs");
            need(std::memcmp(b.X, rb.X.data(), 3 * b.n * sizeof(int64_t)) == 0, "X differs");
            need(std::memcmp(b.V, rb.V.data(), 3 * b.n * sizeof(int64_t)) == 0, "V differs");
            need(std::memcmp(b.id, rb.id.data(), b.n * sizeof(uint64_t)) == 0, "ids differ");
            blocks++, parts += b.n, coders[b.id_coder]++;
        }
    }
    std::printf("decode OK: %d files, %llu blocks, %llu particles identical to snapcomp; ID coders raw/delta/lagr %llu/%llu/%llu\n",
                s.n_files, (unsigned long long)blocks, (unsigned long long)parts, (unsigned long long)coders[0],
                (unsigned long long)coders[1], (unsigned long long)coders[2]);
    pkc_block_free(&b);
    pkc_decoder_free(&d);
    pkc_set_close(&s);
    return 0;
}

static std::vector<uint64_t> share_ids(const pkc_set& s, int64_t k, int64_t nb, uint64_t& np, pkc_decoder& d, pkc_block& b) {
    int64_t first, count;
    need(pkc_set_range(&s, k, nb, &first, &count, &np, err) == 0, err);
    std::vector<uint64_t> ids;
    for (int64_t i = first; i < first + count; ++i) {
        const pkc_pair& p = s.pairs[i];
        need(pkc_read_block(&s.files[p.file], p.block, &d, &b, err) == 0, err);
        ids.insert(ids.end(), b.id, b.id + b.n);
    }
    need(ids.size() == np, "share particle count differs from its blocks");
    return ids;
}

static int cmd_shares(const std::string& prefix, const std::vector<int64_t>& nbs) {
    pkc_set s = open_set(prefix);
    pkc_decoder d;
    pkc_decoder_init(&d);
    pkc_block b = {};
    const uint64_t T = s.n_selected, B = s.files[0].h.block_size;
    for (int64_t nb : nbs) {
        std::vector<uint64_t> all;
        uint64_t mn = UINT64_MAX, mx = 0;
        for (int64_t k = 0; k < nb; ++k) {
            uint64_t np;
            auto ids = share_ids(s, k, nb, np, d, b);
            mn = std::min(mn, np), mx = std::max(mx, np);
            all.insert(all.end(), ids.begin(), ids.end());
        }
        std::sort(all.begin(), all.end());
        need(all.size() == T, "shares do not cover all particles");
        need(std::adjacent_find(all.begin(), all.end()) == all.end(), "an ID appears twice");
        if (s.files[0].h.lattice_n) need(all.front() == 0 && all.back() == T - 1, "lattice IDs not 0..T-1");
        const double ideal = double(T) / double(nb);
        // a share is short or long by at most one pkc block (when there are enough blocks)
        if (uint64_t(s.n_pairs) >= uint64_t(nb))
            need(double(mx) <= ideal + B && double(mn) + B >= ideal, "shares unbalanced by more than one block");
        std::printf("shares OK: NB=%lld, %llu particles once each, per share min %llu max %llu (ideal %.1f, block %llu)\n",
                    (long long)nb, (unsigned long long)T, (unsigned long long)mn, (unsigned long long)mx, ideal,
                    (unsigned long long)B);
    }
    pkc_block_free(&b);
    pkc_decoder_free(&d);
    pkc_set_close(&s);
    return 0;
}

static int cmd_subsel(const std::string& prefix, const std::string& list) {
    double want[16];
    int nw = pkc_parse_percents(list.c_str(), want, 16);
    need(nw > 0, "bad percent list");
    pkc_set s = open_set(prefix, std::vector<double>(want, want + nw));
    const pkc_header& h = s.files[0].h;
    snapcomp::SubsampleTable table(h.percents, int(h.n_subsamples), h.subsample_seed);
    std::vector<int> ok(h.n_subsamples, 0);
    for (int j = 0; j < nw; ++j)
        for (uint32_t k = 0; k < h.n_subsamples; ++k)
            if (std::fabs(h.percents[k] - want[j]) < 1e-9) ok[k] = 1;
    pkc_decoder d;
    pkc_decoder_init(&d);
    pkc_block b = {};
    uint64_t np, total = 0;
    auto ids = share_ids(s, 0, 1, np, d, b);
    for (uint64_t id : ids) need(ok[table(id)], "particle from a subsample that was not selected");
    for (int f = 0; f < s.n_files; ++f) total += s.files[f].h.n_particles;
    need(total == s.n_selected && ids.size() == total, "selected count mismatch");
    std::printf("subsel OK: %s -> %d files, %llu particles (%.3f%% of %llu), all in the selected subsamples\n", list.c_str(),
                s.n_files, (unsigned long long)total, 100.0 * double(total) / double(h.n_total), (unsigned long long)h.n_total);
    pkc_block_free(&b);
    pkc_decoder_free(&d);
    pkc_set_close(&s);
    return 0;
}

static int cmd_cosmo(const std::string& prefix, double Ol_x, double W0_x, double WA_x, double Ez_x) {
    pkc_set s = open_set(prefix);
    double Om, Ol, h0, W0, WA;
    need(pkc_cosmology(&s.files[0].h, &Om, &Ol, &h0, &W0, &WA, err) == 0, err);
    double Ez = pkc_info_E_z(&s.files[0].h);
    // E(z) from Rockstar's hubble.c formula (Aseem's curvature term, Saee's w_eff) with these values
    const double z = s.files[0].h.redshift, z1 = 1.0 + z, a = 1.0 / z1;
    const double weff = (a != 1.0) ? W0 + WA - WA * (a - 1.0) / std::log(a) : W0;
    const double Erock = std::sqrt(Om * z1 * z1 * z1 + (1 - Om - Ol) * z1 * z1 + Ol * std::pow(a, -3.0 * (1.0 + weff)));
    std::printf("cosmo: Om %.6g Ol %.6g h0 %.6g W0 %.6g WA %.6g; E_z header %.10g, Rockstar hubble_scaling %.10g (rel. diff %.2e)\n",
                Om, Ol, h0, W0, WA, Ez, Erock, std::fabs(Erock / Ez - 1));
    need(Ol == Ol_x && W0 == W0_x && WA == WA_x, "cosmology mapping differs from expected");
    need(std::fabs(Ez / Ez_x - 1) < 1e-9, "E_z from info differs from expected");
    need(std::fabs(Erock / Ez - 1) < 1e-9, "Rockstar E(z) differs from pkdgrav3 E(z)");
    std::printf("cosmo OK\n");
    pkc_set_close(&s);
    return 0;
}

// Gadget-2 (SnapFormat 1) files written by pkc_to_gadget: header, pos, vel, ids (uint64)
struct GHeader {
    uint32_t npart[6];
    double mass[6];
    double time, redshift;
    int32_t flag_sfr, flag_feedback;
    uint32_t npartTotal[6];
    int32_t flag_cooling, num_files;
    double BoxSize, Omega0, OmegaLambda, HubbleParam;
    char fill[256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8];
};
static_assert(sizeof(GHeader) == 256, "Gadget header size");

static std::vector<char> record(FILE* f) {
    uint32_t a, b;
    need(std::fread(&a, 4, 1, f) == 1, "short Gadget file");
    std::vector<char> v(a);
    need(std::fread(v.data(), 1, a, f) == a && std::fread(&b, 4, 1, f) == 1 && a == b, "bad Gadget record");
    return v;
}

static int cmd_gadget(const std::string& prefix, const std::string& gprefix) {
    // Rockstar's GADGET2 path (io_gadget.c): pos *= GADGET_LENGTH_CONVERSION (1), vel *= sqrt(SCALE_NOW),
    // SCALE_NOW = header time, PARTICLE_MASS = mass[1] * GADGET_MASS_CONVERSION (1e10)
    std::unordered_map<uint64_t, std::array<float, 6>> g;
    double scale = 0, gmass = 0;
    for (int k = 0;; ++k) {
        FILE* f = std::fopen((gprefix + "." + std::to_string(k)).c_str(), "rb");
        if (!f) break;
        GHeader gh;
        auto hr = record(f);
        std::memcpy(&gh, hr.data(), 256);
        auto pr = record(f), vr = record(f), ir = record(f);
        std::fclose(f);
        scale = gh.time, gmass = gh.mass[1];
        const double vel_rescale = std::sqrt(scale);
        const uint32_t n = gh.npart[1];
        const float* pp = reinterpret_cast<const float*>(pr.data());
        const float* vv = reinterpret_cast<const float*>(vr.data());
        const uint64_t* ii = reinterpret_cast<const uint64_t*>(ir.data());
        for (uint32_t i = 0; i < n; ++i) {
            std::array<float, 6> q;
            for (int d = 0; d < 3; ++d) {
                q[d] = pp[3 * i + d];
                q[d] *= 1.0;                 // GADGET_LENGTH_CONVERSION, float *= double as in io_gadget.c
                q[3 + d] = vv[3 * i + d];
                q[3 + d] *= vel_rescale;     // float *= double
            }
            g[ii[i]] = q;
        }
    }
    need(!g.empty(), "no Gadget files " + gprefix + ".0 ...");

    pkc_set s = open_set(prefix);
    const pkc_header& h = s.files[0].h;
    // native reader (io_pkc.c): same expressions
    const double SCALE_NOW = 1.0 / (1.0 + h.redshift);
    const double Q = double(uint64_t(h.ncell) * h.sub), L = h.box_size;
    const float vfac = float(h.vel_step / std::sqrt(SCALE_NOW));
    const double vel_rescale = std::sqrt(SCALE_NOW);
    const double mass = 27.7536627 * h.omega_m * L * L * L / double(s.n_selected) * 1e10;
    need(scale == SCALE_NOW, "scale factor differs");
    need(gmass * 1e10 == mass, "particle mass differs");
    pkc_decoder d;
    pkc_decoder_init(&d);
    pkc_block b = {};
    uint64_t n = 0, maxulp = 0;
    for (int64_t i = 0; i < s.n_pairs; ++i) {
        need(pkc_read_block(&s.files[s.pairs[i].file], s.pairs[i].block, &d, &b, err) == 0, err);
        for (uint32_t j = 0; j < b.n; ++j, ++n) {
            auto it = g.find(b.id[j]);
            need(it != g.end(), "ID missing from Gadget files");
            float q[6];
            for (int k = 0; k < 3; ++k) {
                q[k] = float((double(b.X[3 * j + k]) + 0.5) / Q * L);
                float vf = float(b.V[3 * j + k]) * vfac;
                q[3 + k] = float(double(vf) * vel_rescale);
            }
            for (int k = 0; k < 6; ++k) {
                uint32_t u1, u2;
                std::memcpy(&u1, &q[k], 4), std::memcpy(&u2, &it->second[k], 4);
                maxulp = std::max<uint64_t>(maxulp, u1 > u2 ? u1 - u2 : u2 - u1);
            }
        }
    }
    need(n == g.size(), "particle count differs from Gadget files");
    std::printf("gadget: %llu particles; max difference %llu ulp (positions and velocities); mass %.10e Msun/h\n",
                (unsigned long long)n, (unsigned long long)maxulp, mass);
    need(maxulp == 0, "native floats differ from the Gadget route");
    std::printf("gadget OK: bit-identical to Rockstar reading pkc_to_gadget output\n");
    pkc_block_free(&b);
    pkc_decoder_free(&d);
    pkc_set_close(&s);
    return 0;
}

static int cmd_blocks(const std::string& prefix, const std::string& gprefix, int64_t K) {
    pkc_set s = open_set(prefix);
    pkc_decoder d;
    pkc_decoder_init(&d);
    pkc_block b = {};
    for (int64_t k = 0; k < K; ++k) {
        uint64_t np;
        auto ids = share_ids(s, k, K, np, d, b);
        FILE* f = std::fopen((gprefix + "." + std::to_string(k)).c_str(), "rb");
        need(f != nullptr, "missing Gadget file " + std::to_string(k));
        record(f), record(f), record(f);
        auto ir = record(f);
        std::fclose(f);
        need(ir.size() == 8 * ids.size(), "share " + std::to_string(k) + ": particle count differs");
        need(std::memcmp(ir.data(), ids.data(), ir.size()) == 0, "share " + std::to_string(k) + ": IDs or order differ");
    }
    std::printf("blocks OK: %lld Gadget files hold exactly the native shares, same order\n", (long long)K);
    pkc_block_free(&b);
    pkc_decoder_free(&d);
    pkc_set_close(&s);
    return 0;
}

static int cmd_open(const std::string& prefix) {
    pkc_set s;
    if (pkc_set_open(prefix.c_str(), nullptr, 0, &s, err)) {
        std::printf("error: %s\n", err);
        return 1;
    }
    pkc_decoder d;
    pkc_decoder_init(&d);
    pkc_block b = {};
    int rc = 0;
    for (int64_t i = 0; i < s.n_pairs && !rc; ++i)
        if (pkc_read_block(&s.files[s.pairs[i].file], s.pairs[i].block, &d, &b, err)) {
            std::printf("error: %s\n", err);
            rc = 1;
        }
    if (!rc) std::printf("opened and decoded %llu particles\n", (unsigned long long)s.n_selected);
    pkc_block_free(&b);
    pkc_decoder_free(&d);
    pkc_set_close(&s);
    return rc;
}

int main(int argc, char** argv) {
    try {
        need(argc >= 3, "usage: pkc_compare decode|shares|subsel|cosmo|gadget|open <prefix> ...");
        std::string cmd = argv[1], prefix = argv[2];
        if (cmd == "decode") return cmd_decode(prefix);
        if (cmd == "shares") {
            std::vector<int64_t> nbs;
            for (int i = 3; i < argc; ++i) nbs.push_back(std::stoll(argv[i]));
            return cmd_shares(prefix, nbs);
        }
        if (cmd == "subsel" && argc == 4) return cmd_subsel(prefix, argv[3]);
        if (cmd == "cosmo" && argc == 7) return cmd_cosmo(prefix, std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]));
        if (cmd == "gadget" && argc == 4) return cmd_gadget(prefix, argv[3]);
        if (cmd == "blocks" && argc == 5) return cmd_blocks(prefix, argv[3], std::stoll(argv[4]));
        if (cmd == "open") return cmd_open(prefix);
        need(false, "unknown command or wrong arguments");
    } catch (std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
