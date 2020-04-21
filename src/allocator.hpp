/*!
 * \file allocator.hpp
 *
 * \brief 實現buddy分配器
 * \author cyy
 * \date 2017-11-27
 */

#pragma once

#include <cstdint>
#include <shared_mutex>

namespace cuda_buddy {

  enum class alloc_location { device = 0, host };

  class allocator final {

  public:
    allocator() = delete;

    explicit allocator(uint8_t max_level_,
                       alloc_location data_location_ = alloc_location::device);
    allocator(const allocator &) = delete;
    allocator &operator=(const allocator &) = delete;

    allocator(allocator &&rhs) = delete;
    allocator &operator=(allocator &&rhs) = delete;

    ~allocator();

    void *alloc(size_t size);
    void *alloc(size_t size, size_t alignment);
    bool free(void *ptr);
    bool in_buddy(const void *ptr) const {
      return static_cast<const uint8_t *>(ptr) >=
                 static_cast<const uint8_t *>(data) &&
             static_cast<const uint8_t *>(ptr) <
                 static_cast<const uint8_t *>(data) + (1ULL << max_level);
    }
    bool full() const {
      std::shared_lock lk(alloc_mutex);
      return used_size == 0;
    }

    void sync_stream() const;

  private:
    enum class node_status : uint8_t {
      unused = 0,
      used = 1,
      used_with_alignment = 2,
      splited = 3,
    };
    void combine(size_t index) noexcept;
    node_status get_node_status(size_t index) const noexcept;
    void set_node_status(size_t index, node_status status) noexcept;
    static size_t left_child_index(size_t index) { return index * 2 + 1; }
    static size_t right_child_index(size_t index) { return index * 2 + 2; }
    static size_t parent_index(size_t index) { return (index + 1) / 2 - 1; }
    static size_t sibling_index(size_t index) { return index + (index & 1); }

  
    size_t used_size{};
    uint8_t max_level{28};
    uint8_t *tree{nullptr};
    void *data{nullptr};
    mutable std::shared_timed_mutex alloc_mutex;
    alloc_location data_location;
  };

} // namespace cuda_buddy
