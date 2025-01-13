#include <iostream>
#include <ap_int.h>
#include <bernoulli_layer.h>

// Declare the prototype of the function
void bernoulli_layer(
    ap_uint<7> i_seed,
    ap_uint<10> i_activation,
    ap_uint<1> i_use_lfsr,
    ap_uint<1> i_sync_reset,
    ap_uint<1> i_en,
    ap_uint<1> i_reset_n,
    ap_uint<1> i_clk,
    ap_uint<1> &o_bernoulli
);

int main() {
    // Testbench signals
    ap_uint<7> i_seed = 0x20;        // Initial seed
    ap_uint<10> i_activation;        // Input activation threshold
    ap_uint<1> i_use_lfsr;           // Use LFSR or ROM
    ap_uint<1> i_sync_reset = 0;     // Sync reset signal
    ap_uint<1> i_en = 1;             // Enable signal
    ap_uint<1> i_reset_n = 1;        // Active-low reset
    ap_uint<1> i_clk = 0;            // Clock signal
    ap_uint<1> o_bernoulli;          // Output signal

    // Test cases
    std::cout << "Testing Bernoulli Layer\n";
    std::cout << "========================\n";

    // Test 1: Reset behavior
    i_reset_n = 0;
    bernoulli_layer(i_seed, 0, 0, i_sync_reset, i_en, i_reset_n, i_clk, o_bernoulli);
    std::cout << "[Reset Test] o_bernoulli = " << o_bernoulli << "\n";

    // Test 2: Use ROM with varying activation thresholds
    i_reset_n = 1;
    i_use_lfsr = 0;
    for (int act = 0; act < 1024; act += 128) {
        i_activation = act;
        bernoulli_layer(i_seed, i_activation, i_use_lfsr, i_sync_reset, i_en, i_reset_n, i_clk, o_bernoulli);
        std::cout << "[ROM Test] Activation = " << act << ", o_bernoulli = " << o_bernoulli << "\n";
    }

    // Test 3: Use LFSR with varying activation thresholds
    i_use_lfsr = 1;
    for (int act = 0; act < 1024; act += 128) {
        i_activation = act;
        bernoulli_layer(i_seed, i_activation, i_use_lfsr, i_sync_reset, i_en, i_reset_n, i_clk, o_bernoulli);
        std::cout << "[LFSR Test] Activation = " << act << ", o_bernoulli = " << o_bernoulli << "\n";
    }

    // Test 4: Verify LFSR progression
    std::cout << "[LFSR State Progression]\n";
    for (int i = 0; i < 10; ++i) {
        bernoulli_layer(i_seed, 512, 1, i_sync_reset, i_en, i_reset_n, i_clk, o_bernoulli);
        std::cout << "Cycle " << i << ": o_bernoulli = " << o_bernoulli << "\n";
        i_clk = ~i_clk; // Toggle clock
    }

    std::cout << "========================\n";
    std::cout << "Testbench Completed.\n";
    return 0;
}
