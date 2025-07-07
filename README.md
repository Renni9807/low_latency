# low_latency

Low Latency Trading Techniques Demo (ARM64 macOS)
=================================================

System Information:
  Hardware threads: 8
  Cache line size: 64 bytes (assumed)
  Architecture: ARM64 (Apple Silicon)
  Timer frequency: 24000000 Hz
  Timer resolution: 41.6667 ns

Testing Low Latency Techniques on ARM64...
==========================================

1. Testing Lock-Free SPSC Queue:
Latency Statistics:
  Samples: 10000
  Average: 0.0883 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 166 ns (0.166 μs)
  P50: 0 ns (0 μs)
  P95: 41 ns (0.041 μs)
  P99: 83 ns (0.083 μs)

2. Testing Lock-Free Memory Pool:
Latency Statistics:
  Samples: 1000
  Average: 0.522 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 41 ns (0.041 μs)
  P50: 41 ns (0.041 μs)
  P95: 41 ns (0.041 μs)
  P99: 41 ns (0.041 μs)

3. Testing Lock-Free Hash Map:
Insert performance:
Latency Statistics:
  Samples: 500
  Average: 0.052 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 41 ns (0.041 μs)
  P50: 0 ns (0 μs)
  P95: 41 ns (0.041 μs)
  P99: 41 ns (0.041 μs)

Lookup performance:
Latency Statistics:
  Samples: 500
  Average: 0.058 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 41 ns (0.041 μs)
  P50: 0 ns (0 μs)
  P95: 41 ns (0.041 μs)
  P99: 41 ns (0.041 μs)

4. Testing ARM64 SIMD Operations:
Vectorized sum result: 49500
SIMD processing time: 6250 ns
SIMD copy time: 0 ns

All ARM64 optimized tests completed!

woosangwon@Woos-MacBook-Air low_latency_system % sudo pmset -a powermode 2                                                                                                                                       
HighPowerMode not supported on Battery Power
woosangwon@Woos-MacBook-Air low_latency_system % sudo nice -n -20 ./low_latency_techniques                                                                                                                       
Low Latency Trading Techniques Demo (ARM64 macOS)

=================================================

System Information:
  Hardware threads: 8
  Cache line size: 64 bytes (assumed)
  Architecture: ARM64 (Apple Silicon)
  Timer frequency: 24000000 Hz
  Timer resolution: 41.6667 ns

Testing Low Latency Techniques on ARM64...

==========================================

1. Testing Lock-Free SPSC Queue:
Latency Statistics:
  Samples: 10000
  Average: 0.0527 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 2166 ns (2.166 μs)
  P50: 0 ns (0 μs)
  P95: 0 ns (0 μs)
  P99: 41 ns (0.041 μs)

2. Testing Lock-Free Memory Pool:
Latency Statistics:
  Samples: 1000
  Average: 0.341 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 208 ns (0.208 μs)
  P50: 0 ns (0 μs)
  P95: 41 ns (0.041 μs)
  P99: 41 ns (0.041 μs)

3. Testing Lock-Free Hash Map:
Insert performance:
Latency Statistics:
  Samples: 500
  Average: 0.03 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 41 ns (0.041 μs)
  P50: 0 ns (0 μs)
  P95: 0 ns (0 μs)
  P99: 41 ns (0.041 μs)

Lookup performance:
Latency Statistics:
  Samples: 500
  Average: 0.044 ticks (0 ns, 0 μs)
  Min: 0 ns (0 μs)
  Max: 41 ns (0.041 μs)
  P50: 0 ns (0 μs)
  P95: 0 ns (0 μs)
  P99: 41 ns (0.041 μs)

4. Testing ARM64 SIMD Operations:
Vectorized sum result: 49500
SIMD processing time: 3708 ns
SIMD copy time: 0 ns

All ARM64 optimized tests completed!


# Low Latency Trading Techniques for ARM64

High-performance, lock-free data structures optimized for Apple Silicon.

## Features
- Lock-free SPSC/MPSC queues
- Memory pool allocator
- Lock-free hash map
- ARM64 NEON SIMD optimizations
- Nanosecond precision benchmarking

## Performance Results
- SPSC Queue: < 50ns average latency
- Memory Pool: < 50ns allocation
- Hash Map: < 50ns lookup

## Build & Run
```bash
g++ -std=c++17 -O3 -march=native low_latency_techniques.cpp -o low_latency_techniques
./low_latency_techniques
