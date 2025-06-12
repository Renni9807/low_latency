// low_latency_techniques.cpp
#include <atomic>
#include <array>
#include <thread>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>

// ARM64 specific includes
#ifdef __aarch64__
#include <arm_neon.h>
#endif

// macOS specific includes
#ifdef __APPLE__
#include <pthread.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace low_latency {

//=============================================================================
// 1. Lock-Free SPSC Queue (Single Producer Single Consumer)
//=============================================================================
template<typename T, size_t Size = 1024>
class LockFreeSPSCQueue {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
    // Cache line padding to prevent false sharing
    alignas(64) std::array<T, Size> buffer;
    alignas(64) std::atomic<size_t> write_index{0};
    alignas(64) std::atomic<size_t> read_index{0};
    
public:
    bool push(const T& item) noexcept {
        const size_t current_write = write_index.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) & (Size - 1);
        
        if (next_write == read_index.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        buffer[current_write] = item;
        write_index.store(next_write, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) noexcept {
        const size_t current_read = read_index.load(std::memory_order_relaxed);
        
        if (current_read == write_index.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }
        
        item = buffer[current_read];
        read_index.store((current_read + 1) & (Size - 1), std::memory_order_release);
        return true;
    }
    
    size_t size() const noexcept {
        const size_t write = write_index.load(std::memory_order_acquire);
        const size_t read = read_index.load(std::memory_order_acquire);
        return (write - read) & (Size - 1);
    }
};

//=============================================================================
// 2. Lock-Free MPSC Queue (Multiple Producer Single Consumer)
//=============================================================================
template<typename T>
class LockFreeMPSCQueue {
private:
    struct Node {
        std::atomic<Node*> next{nullptr};
        T data;
    };
    
    alignas(64) std::atomic<Node*> head;
    alignas(64) std::atomic<Node*> tail;
    
public:
    LockFreeMPSCQueue() {
        Node* dummy = new Node;
        head.store(dummy);
        tail.store(dummy);
    }
    
    ~LockFreeMPSCQueue() {
        while (Node* old_head = head.load()) {
            head.store(old_head->next);
            delete old_head;
        }
    }
    
    void push(T item) {
        Node* new_node = new Node{nullptr, std::move(item)};
        Node* prev_tail = tail.exchange(new_node, std::memory_order_acq_rel);
        prev_tail->next.store(new_node, std::memory_order_release);
    }
    
    bool pop(T& result) {
        Node* head_node = head.load(std::memory_order_acquire);
        Node* next = head_node->next.load(std::memory_order_acquire);
        
        if (next == nullptr) {
            return false; // Queue empty
        }
        
        result = std::move(next->data);
        head.store(next, std::memory_order_release);
        delete head_node;
        return true;
    }
};

//=============================================================================
// 3. Memory Pool (Memory allocation optimization)
//=============================================================================
template<typename T, size_t PoolSize = 1024>
class LockFreeMemoryPool {
private:
    alignas(64) std::array<T, PoolSize> storage;
    alignas(64) std::atomic<size_t> free_list_head{0};
    alignas(64) std::array<std::atomic<size_t>, PoolSize> next_free;
    
public:
    LockFreeMemoryPool() {
        // Initialize free list
        for (size_t i = 0; i < PoolSize - 1; ++i) {
            next_free[i].store(i + 1, std::memory_order_relaxed);
        }
        next_free[PoolSize - 1].store(SIZE_MAX, std::memory_order_relaxed);
    }
    
    T* allocate() noexcept {
        size_t head = free_list_head.load(std::memory_order_acquire);
        
        while (head != SIZE_MAX) {
            size_t next = next_free[head].load(std::memory_order_acquire);
            if (free_list_head.compare_exchange_weak(head, next, 
                std::memory_order_release, std::memory_order_acquire)) {
                return &storage[head];
            }
        }
        return nullptr; // Pool exhausted
    }
    
    void deallocate(T* ptr) noexcept {
        if (!ptr) return;
        
        size_t index = ptr - storage.data();
        if (index >= PoolSize) return; // Invalid pointer
        
        size_t head = free_list_head.load(std::memory_order_acquire);
        do {
            next_free[index].store(head, std::memory_order_relaxed);
        } while (!free_list_head.compare_exchange_weak(head, index, 
            std::memory_order_release, std::memory_order_acquire));
    }
};

//=============================================================================
// 4. CPU Optimizations and Timing (ARM64 optimized)
//=============================================================================

// High-resolution timestamp (ARM64 optimized)
inline uint64_t get_timestamp() noexcept {
#ifdef __aarch64__
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

// Convert ARM64 timestamp to nanoseconds
inline uint64_t timestamp_to_ns(uint64_t timestamp) noexcept {
#ifdef __aarch64__
    // Get the counter frequency
    uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (freq));
    return (timestamp * 1000000000ULL) / freq;
#else
    return timestamp;
#endif
}

// CPU pause instruction (platform specific)
inline void cpu_pause() noexcept {
#ifdef __aarch64__
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Prefetch data into cache (ARM64)
template<int Locality = 3>
inline void prefetch(const void* addr) noexcept {
#ifdef __aarch64__
    if constexpr (Locality >= 3) {
        asm volatile("prfm pldl1keep, [%0]" : : "r" (addr) : "memory");
    } else if constexpr (Locality >= 2) {
        asm volatile("prfm pldl2keep, [%0]" : : "r" (addr) : "memory");
    } else {
        asm volatile("prfm pldl3keep, [%0]" : : "r" (addr) : "memory");
    }
#elif defined(__GNUC__)
    __builtin_prefetch(addr, 0, Locality);
#endif
}

// Memory barrier operations
inline void memory_barrier() noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline void compiler_barrier() noexcept {
    asm volatile("" ::: "memory");
}

// ARM64 specific cache operations
inline void cache_flush(const void* addr, size_t size) noexcept {
#ifdef __aarch64__
    const char* start = static_cast<const char*>(addr);
    const char* end = start + size;
    for (const char* p = start; p < end; p += 64) {
        asm volatile("dc cvac, %0" : : "r" (p) : "memory");
    }
    asm volatile("dsb sy" ::: "memory");
#endif
}

//=============================================================================
// 5. Lock-Free Hash Map (for fast lookups)
//=============================================================================
template<typename Key, typename Value, size_t Size = 1024>
class LockFreeHashMap {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
    struct Entry {
        std::atomic<Key> key{Key{}};
        std::atomic<Value> value{Value{}};
        std::atomic<bool> occupied{false};
    };
    
    alignas(64) std::array<Entry, Size> table;
    
    size_t hash(const Key& key) const noexcept {
        // Simple hash function - replace with better one for production
        return std::hash<Key>{}(key) & (Size - 1);
    }
    
public:
    bool insert(const Key& key, const Value& value) noexcept {
        size_t index = hash(key);
        
        for (size_t attempts = 0; attempts < Size; ++attempts) {
            Entry& entry = table[index];
            
            // Try to claim empty slot
            bool expected = false;
            if (entry.occupied.compare_exchange_weak(expected, true, 
                std::memory_order_acquire, std::memory_order_relaxed)) {
                
                entry.key.store(key, std::memory_order_relaxed);
                entry.value.store(value, std::memory_order_release);
                return true;
            }
            
            // Check if key already exists
            if (entry.key.load(std::memory_order_acquire) == key) {
                entry.value.store(value, std::memory_order_release);
                return true;
            }
            
            index = (index + 1) & (Size - 1);
        }
        return false; // Table full
    }
    
    bool find(const Key& key, Value& value) const noexcept {
        size_t index = hash(key);
        
        for (size_t attempts = 0; attempts < Size; ++attempts) {
            const Entry& entry = table[index];
            
            if (!entry.occupied.load(std::memory_order_acquire)) {
                return false; // Empty slot found, key doesn't exist
            }
            
            if (entry.key.load(std::memory_order_acquire) == key) {
                value = entry.value.load(std::memory_order_acquire);
                return true;
            }
            
            index = (index + 1) & (Size - 1);
        }
        return false;
    }
};

//=============================================================================
// 6. Ring Buffer with Batching (for high throughput)
//=============================================================================
template<typename T, size_t Size = 1024>
class BatchingRingBuffer {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
    alignas(64) std::array<T, Size> buffer;
    alignas(64) std::atomic<size_t> write_index{0};
    alignas(64) std::atomic<size_t> read_index{0};
    alignas(64) std::atomic<size_t> committed_write{0};
    
public:
    // Reserve space for batch writing
    bool reserve(size_t count, size_t& start_index) noexcept {
        const size_t current_write = write_index.load(std::memory_order_relaxed);
        const size_t new_write = current_write + count;
        const size_t read_pos = read_index.load(std::memory_order_acquire);
        
        // Check if we have enough space
        if (new_write - read_pos > Size) {
            return false;
        }
        
        // Try to reserve space
        size_t expected = current_write;
        if (write_index.compare_exchange_weak(expected, new_write, 
            std::memory_order_acq_rel, std::memory_order_relaxed)) {
            start_index = current_write & (Size - 1);
            return true;
        }
        return false;
    }
    
    // Commit batch write
    void commit_write(size_t count) noexcept {
        committed_write.fetch_add(count, std::memory_order_release);
    }
    
    // Batch read
    size_t read_batch(T* output, size_t max_count) noexcept {
        const size_t current_read = read_index.load(std::memory_order_relaxed);
        const size_t available = committed_write.load(std::memory_order_acquire) - current_read;
        const size_t to_read = std::min(max_count, available);
        
        if (to_read == 0) {
            return 0;
        }
        
        const size_t start_pos = current_read & (Size - 1);
        const size_t end_pos = (current_read + to_read) & (Size - 1);
        
        if (start_pos < end_pos) {
            // Contiguous read
            std::copy(buffer.begin() + start_pos, buffer.begin() + end_pos, output);
        } else {
            // Wrap-around read
            const size_t first_part = Size - start_pos;
            std::copy(buffer.begin() + start_pos, buffer.end(), output);
            std::copy(buffer.begin(), buffer.begin() + end_pos, output + first_part);
        }
        
        read_index.store(current_read + to_read, std::memory_order_release);
        return to_read;
    }
    
    // Write single item to reserved space
    void write_at(size_t index, const T& item) noexcept {
        buffer[index] = item;
    }
};

//=============================================================================
// 7. Spin Lock (when lock-free is not possible)
//=============================================================================
class SpinLock {
private:
    alignas(64) std::atomic_flag flag = ATOMIC_FLAG_INIT;
    
public:
    void lock() noexcept {
        int backoff = 1;
        while (flag.test_and_set(std::memory_order_acquire)) {
            // Exponential backoff
            for (int i = 0; i < backoff; ++i) {
                cpu_pause();
            }
            backoff = std::min(backoff * 2, 64);
        }
    }
    
    void unlock() noexcept {
        flag.clear(std::memory_order_release);
    }
    
    bool try_lock() noexcept {
        return !flag.test_and_set(std::memory_order_acquire);
    }
};

//=============================================================================
// 8. RAII Spin Lock Guard
//=============================================================================
class SpinLockGuard {
private:
    SpinLock& lock_;
    
public:
    explicit SpinLockGuard(SpinLock& lock) : lock_(lock) {
        lock_.lock();
    }
    
    ~SpinLockGuard() {
        lock_.unlock();
    }
    
    // Non-copyable, non-movable
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;
};

//=============================================================================
// 9. Thread Affinity and Priority (macOS/Darwin specific)
//=============================================================================
class ThreadOptimizer {
public:
    // Set thread to specific CPU core (macOS)
    static bool set_cpu_affinity(std::thread& thread, int cpu_id) noexcept {
#ifdef __APPLE__
        // macOS doesn't support direct CPU affinity, but we can set thread QoS
        thread_extended_policy_data_t policy;
        policy.timeshare = 0; // Set to 1 for timeshare, 0 for fixed priority
        
        kern_return_t result = thread_policy_set(
            pthread_mach_thread_np(thread.native_handle()),
            THREAD_EXTENDED_POLICY,
            (thread_policy_t)&policy,
            THREAD_EXTENDED_POLICY_COUNT);
        
        return result == KERN_SUCCESS;
#else
        return false;
#endif
    }
    
    // Set high priority (macOS)
    static bool set_high_priority(std::thread& thread) noexcept {
#ifdef __APPLE__
        struct sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        return pthread_setschedparam(thread.native_handle(), SCHED_FIFO, &param) == 0;
#else
        return false;
#endif
    }
    
    // Lock pages in memory (prevent swapping)
    static bool lock_memory() noexcept {
#ifdef __APPLE__
        return mlock(nullptr, 0) == 0; // Simplified version
#else
        return false;
#endif
    }
};

//=============================================================================
// 10. Performance Measurement Utilities
//=============================================================================
class LatencyMeasurer {
private:
    std::array<uint64_t, 10000> samples;
    size_t sample_count = 0;
    
public:
    void add_sample(uint64_t latency_ticks) noexcept {
        if (sample_count < samples.size()) {
            samples[sample_count++] = latency_ticks;
        }
    }
    
    void print_statistics() const {
        if (sample_count == 0) {
            std::cout << "No samples collected\n";
            return;
        }
        
        auto sorted_samples = samples;
        std::sort(sorted_samples.begin(), sorted_samples.begin() + sample_count);
        
        uint64_t sum = 0;
        for (size_t i = 0; i < sample_count; ++i) {
            sum += sorted_samples[i];
        }
        
        double avg_ticks = static_cast<double>(sum) / sample_count;
        double avg_ns = timestamp_to_ns(static_cast<uint64_t>(avg_ticks));
        
        std::cout << "Latency Statistics:\n";
        std::cout << "  Samples: " << sample_count << "\n";
        std::cout << "  Average: " << avg_ticks << " ticks (" << avg_ns << " ns, " << avg_ns/1000.0 << " μs)\n";
        
        uint64_t min_ns = timestamp_to_ns(sorted_samples[0]);
        uint64_t max_ns = timestamp_to_ns(sorted_samples[sample_count-1]);
        uint64_t p50_ns = timestamp_to_ns(sorted_samples[sample_count/2]);
        uint64_t p95_ns = timestamp_to_ns(sorted_samples[sample_count*95/100]);
        uint64_t p99_ns = timestamp_to_ns(sorted_samples[sample_count*99/100]);
        
        std::cout << "  Min: " << min_ns << " ns (" << min_ns/1000.0 << " μs)\n";
        std::cout << "  Max: " << max_ns << " ns (" << max_ns/1000.0 << " μs)\n";
        std::cout << "  P50: " << p50_ns << " ns (" << p50_ns/1000.0 << " μs)\n";
        std::cout << "  P95: " << p95_ns << " ns (" << p95_ns/1000.0 << " μs)\n";
        std::cout << "  P99: " << p99_ns << " ns (" << p99_ns/1000.0 << " μs)\n";
    }
};

//=============================================================================
// 11. Cache-Aligned Data Structure
//=============================================================================
template<typename T>
struct CacheAligned {
    alignas(64) T data;
    
    CacheAligned() = default;
    CacheAligned(const T& value) : data(value) {}
    
    T& operator*() { return data; }
    const T& operator*() const { return data; }
    
    T* operator->() { return &data; }
    const T* operator->() const { return &data; }
};

//=============================================================================
// 12. ARM64 SIMD Optimizations
//=============================================================================
#ifdef __aarch64__
class SIMDOperations {
public:
    // Vectorized sum using ARM NEON
    static double vectorized_sum(const float* data, size_t count) noexcept {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t vec_count = count & ~3; // Round down to multiple of 4
        
        for (size_t i = 0; i < vec_count; i += 4) {
            float32x4_t data_vec = vld1q_f32(&data[i]);
            sum_vec = vaddq_f32(sum_vec, data_vec);
        }
        
        // Sum the vector elements
        float sum = vaddvq_f32(sum_vec);
        
        // Handle remaining elements
        for (size_t i = vec_count; i < count; ++i) {
            sum += data[i];
        }
        
        return sum;
    }
    
    // Vectorized copy
    static void vectorized_copy(const float* src, float* dst, size_t count) noexcept {
        size_t vec_count = count & ~3;
        
        for (size_t i = 0; i < vec_count; i += 4) {
            float32x4_t data = vld1q_f32(&src[i]);
            vst1q_f32(&dst[i], data);
        }
        
        // Handle remaining elements
        for (size_t i = vec_count; i < count; ++i) {
            dst[i] = src[i];
        }
    }
};
#endif

//=============================================================================
// 13. Test Framework
//=============================================================================
void test_all_techniques() {
    std::cout << "Testing Low Latency Techniques on ARM64...\n";
    std::cout << "==========================================\n\n";
    
    // Test SPSC Queue
    {
        std::cout << "1. Testing Lock-Free SPSC Queue:\n";
        LockFreeSPSCQueue<int> queue;
        LatencyMeasurer measurer;
        
        const int iterations = 10000;
        std::atomic<bool> consumer_done{false};
        
        std::thread producer([&queue, &measurer, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                uint64_t start = get_timestamp();
                while (!queue.push(i)) {
                    cpu_pause();
                }
                uint64_t end = get_timestamp();
                measurer.add_sample(end - start);
            }
        });
        
        std::thread consumer([&queue, iterations, &consumer_done]() {
            int value;
            int received = 0;
            while (received < iterations) {
                if (queue.pop(value)) {
                    ++received;
                } else {
                    cpu_pause();
                }
            }
            consumer_done = true;
        });
        
        producer.join();
        consumer.join();
        
        measurer.print_statistics();
        std::cout << "\n";
    }
    
    // Test Memory Pool
    {
        std::cout << "2. Testing Lock-Free Memory Pool:\n";
        LockFreeMemoryPool<int, 1000> pool;
        LatencyMeasurer measurer;
        
        const int iterations = 1000;
        
        for (int i = 0; i < iterations; ++i) {
            uint64_t start = get_timestamp();
            int* ptr = pool.allocate();
            uint64_t end = get_timestamp();
            
            if (ptr) {
                *ptr = i;
                measurer.add_sample(end - start);
                pool.deallocate(ptr);
            }
        }
        
        measurer.print_statistics();
        std::cout << "\n";
    }
    
    // Test Hash Map
    {
        std::cout << "3. Testing Lock-Free Hash Map:\n";
        LockFreeHashMap<int, double> hashmap;
        LatencyMeasurer insert_measurer, lookup_measurer;
        
        const int iterations = 500;
        
        // Insert test
        for (int i = 0; i < iterations; ++i) {
            uint64_t start = get_timestamp();
            bool success = hashmap.insert(i, i * 3.14);
            uint64_t end = get_timestamp();
            
            if (success) {
                insert_measurer.add_sample(end - start);
            }
        }
        
        // Lookup test
        for (int i = 0; i < iterations; ++i) {
            double value;
            uint64_t start = get_timestamp();
            bool found = hashmap.find(i, value);
            uint64_t end = get_timestamp();
            
            if (found) {
                lookup_measurer.add_sample(end - start);
            }
        }
        
        std::cout << "Insert performance:\n";
        insert_measurer.print_statistics();
        std::cout << "\nLookup performance:\n";
        lookup_measurer.print_statistics();
        std::cout << "\n";
    }
    
#ifdef __aarch64__
    // Test SIMD operations
    {
        std::cout << "4. Testing ARM64 SIMD Operations:\n";
        
        const size_t data_size = 10000;
        std::vector<float> data(data_size);
        std::vector<float> result(data_size);
        
        // Initialize with random data
        for (size_t i = 0; i < data_size; ++i) {
            data[i] = static_cast<float>(i % 100) * 0.1f;
        }
        
        // Test vectorized sum
        uint64_t start = get_timestamp();
        double sum = SIMDOperations::vectorized_sum(data.data(), data_size);
        uint64_t end = get_timestamp();
        
        std::cout << "Vectorized sum result: " << sum << "\n";
        std::cout << "SIMD processing time: " << timestamp_to_ns(end - start) << " ns\n";
        
        // Test vectorized copy
        start = get_timestamp();
        SIMDOperations::vectorized_copy(data.data(), result.data(), data_size);
        end = get_timestamp();
        
        std::cout << "SIMD copy time: " << timestamp_to_ns(end - start) << " ns\n";
        std::cout << "\n";
    }
#endif
    
    std::cout << "All ARM64 optimized tests completed!\n";
}

} // namespace low_latency

int main() {
    std::cout << "Low Latency Trading Techniques Demo (ARM64 macOS)\n";
    std::cout << "=================================================\n\n";
    
    // Display system info
    std::cout << "System Information:\n";
    std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "  Cache line size: 64 bytes (assumed)\n";
    std::cout << "  Architecture: ARM64 (Apple Silicon)\n";
    
#ifdef __aarch64__
    // Get ARM64 timer frequency
    uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (freq));
    std::cout << "  Timer frequency: " << freq << " Hz\n";
    std::cout << "  Timer resolution: " << (1000000000.0 / freq) << " ns\n";
#endif
    
    std::cout << "\n";
    
    // Run tests
    low_latency::test_all_techniques();
    
    return 0;
}