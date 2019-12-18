/*!
 * \file pool.hpp
 *
 * \brief 基於buddy算法實現緩存池
 * \author cyy
 * \date 2017-11-27
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "allocator.hpp"

namespace cuda_buddy {
  class pool final {

  public:
    static void set_device_pool_size(uint8_t max_level);
    static void set_host_pool_size(uint8_t max_level);

  public:
    explicit pool(int gpu_no_);

    pool(const pool &) = delete;
    pool &operator=(const pool &) = delete;

    pool(pool &&rhs) = default;
    pool &operator=(pool &&rhs) = default;

    ~pool();

    void *alloc(size_t size);
    void *alloc(size_t size, size_t alignment);
    bool free(void *ptr);
    bool full() const;

    static void release_global_pool(int gpu_no);

  public:
    static constexpr uint8_t buddy_block_level{28};
    static constexpr int max_device_num{256};

  private:
    struct global_pool_type final {
      std::mutex pool_mutex;
      std::list<std::unique_ptr<allocator>> pool;
      size_t alloced_block_num;

      void add_block(std::unique_ptr<allocator> block) {
        std::lock_guard lk(pool_mutex);
        pool.push_back(std::move(block));
      }
      void clear() {
        std::lock_guard lk(pool_mutex);
        pool.clear();
      }
    };

  private:
    bool release();
    uint8_t get_max_level() const;
    std::unique_ptr<allocator> get_block();
    static global_pool_type &get_global_pool(int gpu_no);

  private:
    int gpu_no{-1};
    alloc_location data_location{alloc_location::host};
    std::vector<std::unique_ptr<allocator>> local_pool;
    mutable std::shared_timed_mutex local_pool_mutex;

  private:
    static inline std::atomic<uint8_t> device_max_level{0};
    static inline std::atomic<uint8_t> host_max_level{0};
    static inline std::array<global_pool_type, max_device_num>
        global_device_pool;
    static inline global_pool_type global_host_pool;
  };

} // namespace cuda_buddy
