#include <ap_int.h>
#include <hls_stream.h>

#define G_M 7

void bernoulli_layer(
    ap_uint<G_M> i_seed,
    ap_uint<10> i_activation,
    ap_uint<1> i_use_lfsr,
    ap_uint<1> i_sync_reset,
    ap_uint<1> i_en,
    ap_uint<1> i_reset_n,
    ap_uint<1> i_clk,
    ap_uint<1> &o_bernoulli
) {
    // LFSR state
    static ap_uint<G_M> r_lfsr = "0x1";
    static ap_uint<10> dout = "0x030";
    static int state_counter = 0;

    // Polynomial for feedback
    const ap_uint<G_M> g_poly = "0b1100000";

    // ROM data (pre-computed)
    const ap_uint<10> rom[62] = {
        0x30, 0x10, 0x00, 0x2F8, 0x170, 0x0B0, 0x050, 0x020, 0x308, 0x278, 0x130,
        0x090, 0x040, 0x318, 0x180, 0x3B8, 0x1D0, 0x0E0, 0x368, 0x2A8, 0x248,
        0x218, 0x100, 0x378, 0x1B0, 0x0D0, 0x060, 0x328, 0x288, 0x238, 0x110,
        0x080, 0x338, 0x190, 0x0C0, 0x358, 0x1A0, 0x3C8, 0x2D8, 0x160, 0x3A8,
        0x2C8, 0x258, 0x120, 0x388, 0x2B8, 0x150, 0x0A0, 0x348, 0x298, 0x140,
        0x398, 0x1C0, 0x3D8, 0x1E0, 0x3E8, 0x2E8, 0x268, 0x228, 0x208, 0x1F8, 0x0F0
    };

    // Reset behavior
    if (i_reset_n == 0) {
        r_lfsr = "0x1";
        dout = "0x030";
        state_counter = 0;
    } else if (i_clk) { // Clock edge
        if (i_sync_reset == 1) {
            r_lfsr = i_seed;
            dout = "0x030";
            state_counter = 0;
        } else if (i_en == 1) {
            // LFSR Update
            ap_uint<1> feedback = r_lfsr[G_M - 1];
            r_lfsr = (r_lfsr << 1) ^ (feedback ? g_poly : 0);

            // ROM Update
            dout = rom[state_counter];
            if (state_counter == 61) {
                state_counter = 0;
            } else {
                state_counter++;
            }
        }
    }

    // Bernoulli Output
    if (i_use_lfsr) {
        o_bernoulli = (i_activation > r_lfsr.range(G_M - 1, 1)) ? 1 : 0;
    } else {
        o_bernoulli = (i_activation > dout) ? 1 : 0;
    }
}
