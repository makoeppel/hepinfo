#include <ap_int.h>

// Define types and constants
constexpr int g_m = 7;
constexpr ap_uint<g_m> g_poly = 0b1100000;
constexpr int ROM_SIZE = 62;
constexpr ap_uint<10> rom[ROM_SIZE] = {
    0x030, 0x010, 0x000, 0x2f8, 0x170, 0x0b0, 0x050,
    0x020, 0x308, 0x278, 0x130, 0x090, 0x040, 0x318,
    0x180, 0x3b8, 0x1d0, 0x0e0, 0x368, 0x2a8, 0x248,
    0x218, 0x100, 0x378, 0x1b0, 0x0d0, 0x060, 0x328,
    0x288, 0x238, 0x110, 0x080, 0x338, 0x190, 0x0c0,
    0x358, 0x1a0, 0x3c8, 0x2d8, 0x160, 0x3a8, 0x2c8,
    0x258, 0x120, 0x388, 0x2b8, 0x150, 0x0a0, 0x348,
    0x298, 0x140, 0x398, 0x1c0, 0x3d8, 0x1e0, 0x3e8,
    0x2e8, 0x268, 0x228, 0x208, 0x1f8, 0x0f0
};

void bernoulli_lfsr(
    bool i_sync_reset,
    ap_uint<g_m> i_seed,
    bool i_en,
    ap_uint<10> i_activation,
    bool &o_bernoulli,
    ap_uint<g_m> &o_lfsr,
    bool i_reset_n,
    bool i_clk
) {
    static ap_uint<g_m> r_lfsr = -1; // Initialize to all 1s
    static int state_counter = 0;
    static ap_uint<10> dout = 0x030;

    ap_uint<g_m> w_mask = 0;

    // Generate the mask for feedback
    for (int k = 0; k < g_m; ++k) {
        w_mask[k] = g_poly[k] & r_lfsr[0];
    }

    // LFSR process
    if (!i_reset_n) {
        r_lfsr = -1; // Reset LFSR
    } else if (i_sync_reset) {
        r_lfsr = i_seed;
    } else if (i_en) {
        r_lfsr = (r_lfsr >> 1) ^ w_mask;
    }

    // Output LFSR state
    o_lfsr = r_lfsr;

    // Bernoulli output
    o_bernoulli = (i_activation > dout) ? 1 : 0;

    // ROM process
    if (!i_reset_n) {
        state_counter = 0;
        dout = rom[0];
    } else if (i_sync_reset) {
        state_counter = 0;
        dout = rom[0];
    } else if (i_en) {
        state_counter = (state_counter == ROM_SIZE - 1) ? 0 : state_counter + 1;
        dout = rom[state_counter];
    }
}
