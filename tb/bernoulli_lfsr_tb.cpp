#include <iostream>
#include "bernoulli_lfsr.h" // Include the HLS code file

int main() {
    bool o_bernoulli;
    ap_uint<g_m> o_lfsr;

    // Initialize inputs
    bool i_sync_reset = true;
    ap_uint<g_m> i_seed = 0b0010000;
    bool i_en = false;
    ap_uint<10> i_activation = 0x020;
    bool i_reset_n = true;
    bool i_clk = false;

    // Apply reset
    bernoulli_lfsr(i_sync_reset, i_seed, i_en, i_activation, o_bernoulli, o_lfsr, i_reset_n, i_clk);

    // Check outputs
    std::cout << "o_lfsr: " << o_lfsr.to_string(2) << std::endl;
    std::cout << "o_bernoulli: " << o_bernoulli << std::endl;

    // Simulate multiple clock cycles
    for (int cycle = 0; cycle < 10; ++cycle) {
        i_clk = !i_clk; // Toggle clock
        if (cycle == 1) i_sync_reset = false; // De-assert reset
        if (cycle == 2) i_en = true;         // Enable LFSR

        bernoulli_lfsr(i_sync_reset, i_seed, i_en, i_activation, o_bernoulli, o_lfsr, i_reset_n, i_clk);

        std::cout << "Cycle " << cycle << " - o_lfsr: " << o_lfsr.to_string(2)
                  << ", o_bernoulli: " << o_bernoulli << std::endl;
    }

    return 0;
}
