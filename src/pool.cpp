/*!
 * \file pool.cpp
 *
 * \brief 基於buddy算法實現緩存池
 * \author cyy
 * \date 2017-11-27
 */
#include <algorithm>
#include <spdlog/spdlog.h>

#include "pool.hpp"

namespace cuda_buddy {

  void pool::set_device_pool_size(uint8_t max_level) {
    device_max_level.store((std::max)(buddy_block_level, max_level));
  }

  void pool::set_host_pool_size(uint8_t max_level) {
    host_max_level.store((std::max)(buddy_block_level, max_level));
  }

  pool::pool(int gpu_no_) : gpu_no(gpu_no_) {

    if (gpu_no < 0) {
      gpu_no = -1;
      data_location = alloc_location::host;
    } else {
      data_location = alloc_location::device;
    }
    if (data_location == alloc_location::host) {
      return;
    }

    if (gpu_no >= max_device_num) {
      throw std::runtime_error(std::string("unsupported gpu ") +
                               std::to_string(gpu_no));
    }
  }

  pool::~pool() { release(); }

  void *pool::alloc(size_t size) { return alloc(size, 1); }

  void *pool::alloc(size_t size, size_t alignment) {

    if (size > (1ULL << buddy_block_level)) {
      spdlog::warn("too large size {}", size);
      return nullptr;
    }

    if (get_max_level() == 0) {
      spdlog::warn("max level is 0");
      return nullptr;
    }

    //先在已有的空間中分配
    size_t prev_pool_size = 0;
    {
      std::shared_lock pool_lock(local_pool_mutex);
      prev_pool_size = local_pool.size();
      for (const auto &allocator : local_pool) {
        auto ptr = allocator->alloc(size, alignment);
        if (ptr) {
          return ptr;
        }
      }
    }

    auto block = get_block();
    if (!block.get()) {
      {
        std::shared_lock pool_lock(local_pool_mutex);
        if (prev_pool_size >= local_pool.size()) {
          return nullptr;
        }
      }
      return alloc(size, alignment);
    }

    {
      std::lock_guard pool_lock(local_pool_mutex);
      local_pool.emplace_back(std::move(block));
    }
    return alloc(size, alignment);
  }
  uint8_t pool::get_max_level() const {
    if (data_location == alloc_location::host) {
      return host_max_level.load();
    }
    return device_max_level.load();
  }

  pool::global_pool_type &pool::get_global_pool(int gpu_no) {
    if (gpu_no < 0) {
      return global_host_pool;
    }
    if (gpu_no >= max_device_num) {
      spdlog::error("invalid gpu {}", gpu_no);
      throw std::runtime_error(std::string("invalid gpu ") +
                               std::to_string(gpu_no));
    }
    return global_device_pool[gpu_no];
  }

  bool pool::free(void *ptr) {
    std::shared_lock pool_lock(local_pool_mutex);
    for (auto &allocator : local_pool) {
      if (allocator->free(ptr)) {
        return true;
      }
    }
    return false;
  }
  bool pool::full() const {
    std::shared_lock pool_lock(local_pool_mutex);
    return std::all_of(local_pool.begin(), local_pool.end(),
                       [](auto const &a) { return a->full(); });
  }

  bool pool::release() {
    std::lock_guard pool_lock(local_pool_mutex);
    if (local_pool.empty()) {
      return true;
    }
    local_pool[0]->sync_stream();
    auto &global_pool = get_global_pool(gpu_no);
    size_t i = 0;
    while (i < local_pool.size()) {
      if (!local_pool[i]->full()) {
        i++;
        continue;
      }
      if (i + 1 < local_pool.size()) {
        std::swap(local_pool[i], local_pool.back());
      }
      global_pool.add_block(std::move(local_pool.back()));
      local_pool.pop_back();
    }
    return local_pool.empty();
  }

  void pool::release_global_pool(int gpu_no) {
    auto &global_pool = get_global_pool(gpu_no);
    global_pool.clear();
  }

  std::unique_ptr<allocator> pool::get_block() {
    auto &global_pool = get_global_pool(gpu_no);
    std::lock_guard global_pool_lock(global_pool.pool_mutex);
    if (global_pool.pool.empty()) {
      auto max_block_num =
          static_cast<size_t>(1ULL << (get_max_level() - buddy_block_level));
      if (global_pool.alloced_block_num >= max_block_num) {
        auto location_str =
            (data_location == alloc_location::host) ? "host" : "device";
        spdlog::warn(
            "no {} block available,allocated_block_num {},max_block_num "
            "{},consider increasing {} pool size",
            location_str, global_pool.alloced_block_num, max_block_num,
            location_str);
        return {};
      }
      auto buddy_block =
          std::make_unique<allocator>(buddy_block_level, data_location);
      global_pool.alloced_block_num++;
      return buddy_block;
    }
    auto buddy_block = std::move(global_pool.pool.front());
    global_pool.pool.pop_front();
    return buddy_block;
  }
} // namespace cuda_buddy
