# Makefile for Bernoulli LFSR HLS Verification

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2

# Target executable
TARGET = bernoulli_layer_tb

# Source files
SRCS = bernoulli_layer.h bernoulli_layer_tb.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

# Run the testbench
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run