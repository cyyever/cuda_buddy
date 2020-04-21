/*!
 * \file allocator.cpp
 *
 * \brief 實現buddy分配器
 * \author cyy
 * \date 2017-11-27
 */

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(__linux__)
#include <linux/mman.h>
#include <sys/mman.h>
#endif

#include "allocator.hpp"

namespace cuda_buddy {

  namespace {
    static inline bool is_pow_of_2(size_t x) { return !(x & (x - 1)); }

    static inline size_t next_pow_of_2(size_t x) {
      if (is_pow_of_2(x)) {
        return x;
      }
      x |= x >> 1u;
      x |= x >> 2u;
      x |= x >> 4u;
      x |= x >> 8u;
      x |= x >> 16u;
      return x + 1;
    }

    static inline size_t _index_offset(size_t index, uint8_t level,
                                       uint8_t max_level) {
      return ((index + 1) - (1ULL << level)) << (max_level - level);
    }
  } // namespace

  // CUDA: various checks for different function calls.
  static inline void cuda_check(cudaError_t error, const std::string &operation,
                                bool do_abort) {
    if (error != cudaSuccess && error != cudaErrorCudartUnloading) {
      std::string err_str(operation + " failed:");
      err_str += cudaGetErrorString(error);
      if (do_abort) {
        spdlog::error("{}", err_str);
        abort();
      } else {
        throw std::runtime_error(err_str);
      }
    }
  }

  allocator::allocator(uint8_t max_level_, alloc_location data_location_)
      : max_level(max_level_), tree(nullptr), data(nullptr),
        data_location(data_location_) {

    assert(max_level <= 32);
    size_t size = 1ULL << max_level;

#if defined(__linux__)
    // MAP_ANONYMOUS will do zero initialization
    tree = (uint8_t *)mmap(nullptr, size / 2, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (tree == MAP_FAILED) {
      spdlog::get("cuda_buddy")
          ->error(
              "mmap failed:{}",
              std::make_error_code(static_cast<std::errc>(errno)).message());
      throw std::bad_alloc();
    }
#else
    tree = new uint8_t[size / 2]{};
#endif

    if (data_location == alloc_location::device) {
      cuda_check(cudaMalloc(&data, size), "cudaMalloc", false);
    } else {
      cuda_check(cudaMallocHost(&data, size), "cudaMallocHost", false);
    }
  }

  allocator::~allocator() {

#if defined(__linux__)
    size_t size = 1ULL << max_level;
    if (munmap(tree, size / 2) != 0) {
      spdlog::get("cuda_buddy")
          ->error(
              "munmap failed:{}",
              std::make_error_code(static_cast<std::errc>(errno)).message());
      abort();
    }
#else
    delete[] tree;
#endif

    if (data) {
      if (data_location == alloc_location::device) {
        // according to nvidia documentation,cudaFree will perform
        // synchronization internally,so we don't need to call sync_stream()
        // here.
        cuda_check(cudaFree(data), "cudaFree", true);
      } else {
        cuda_check(cudaFreeHost(data), "cudaFreeHost", true);
      }
    }
  }

  void allocator::sync_stream() const {
    if (data_location == alloc_location::device) {
      auto error = cudaStreamSynchronize(cudaStreamPerThread);
      if (error != cudaErrorInitializationError) {
        cuda_check(error, "cudaStreamSynchronize", true);
      }
    }
  }

  void *allocator::alloc(size_t size, size_t alignment) {
    std::lock_guard lk(alloc_mutex);

    if (size == 0) {
      size = 1;
    }

    if (alignment > 1) {
      size += alignment - 1;
    }
    size = next_pow_of_2(size);
    //我们目前的参数类型决定了只能分配这么多
    if (size > static_cast<size_t>(UINT32_MAX)) {
      spdlog::warn("too large size {}", size);
      return nullptr;
    }

    size_t length = 1ULL << max_level;

    if (size > length) {
      spdlog::warn("too large size {}", size);
      return nullptr;
    }

    size_t index = 0;
    uint8_t level = 0;

    while (true) {
      if (size == length) {
        if (get_node_status(index) == node_status::unused) {
          used_size += size;
          auto ptr = static_cast<uint8_t *>(data) +
                     _index_offset(index, level, max_level);

          if (alignment > 1) {
            auto remainder = reinterpret_cast<uintptr_t>(ptr) % alignment;
            if (remainder != 0) {
              set_node_status(index, node_status::used_with_alignment);
              ptr += alignment - remainder;
              return ptr;
            }
          }
          set_node_status(index, node_status::used);
          return ptr;
        }
      } else {
        // size < length
        switch (get_node_status(index)) {
          case node_status::used:
            break;
          case node_status::used_with_alignment:
            break;
          case node_status::unused:
            // split first
            set_node_status(index, node_status::splited);
            set_node_status(left_child_index(index), node_status::unused);
            set_node_status(right_child_index(index), node_status::unused);
            [[fallthrough]];
          default:
            index = left_child_index(index);
            length /= 2;
            level++;
            continue;
        }
      }
      if (index & 1) { // try right child
        ++index;
        continue;
      }

      // 回退到上一层的右节点
      while (index != 0) {
        level--;
        length *= 2;
        index = parent_index(index);
        if (index & 1) { // try right child
          ++index;
          break;
        }
      }
      if (index == 0) {
        break;
      }
    }
    return nullptr;
  }

  void *allocator::alloc(size_t size) { return alloc(size, 1); }

  bool allocator::free(void *ptr) {
    if (!ptr) {
      return true;
    }

    if (!in_buddy(ptr)) {
      return false;
    }

    std::lock_guard lk(alloc_mutex);

    size_t left = 0;
    size_t length = 1ULL << max_level;
    size_t index = 0;
    uint8_t level = 0;
    size_t offset = static_cast<uint8_t *>(ptr) - static_cast<uint8_t *>(data);

    while (level <= max_level) {
      auto cur_node_status = get_node_status(index);
      switch (cur_node_status) {
        case node_status::used_with_alignment:
          [[fallthrough]];
        case node_status::used: {
          auto index_off_set = _index_offset(index, level, max_level);

          if (cur_node_status == node_status::used_with_alignment) {
            if (offset == index_off_set) {
              spdlog::get("cuda_buddy")
                  ->error("allocator can't free unaligned pointer");
              return false;
            }
          } else {
            if (offset != index_off_set) {
              spdlog::get("cuda_buddy")
                  ->error("allocator can't free pointer in allocated block");
              return false;
            }
          }
        }
          used_size -= (1ULL << (max_level - level));
          combine(index);
          return true;
        case node_status::unused:
          spdlog::get("cuda_buddy")
              ->debug("allocator can't free unallocated pointer");
          return false;
        default:
          length /= 2;
          level++;
          if (offset < left + length) {
            index = left_child_index(index);
          } else {
            left += length;
            index = right_child_index(index);
          }
          break;
      }
    }
    return false;
  }

  void allocator::combine(size_t index) noexcept {
    while (index != 0) {
      if (get_node_status(sibling_index(index)) != node_status::unused) {
        break;
      }
      index = parent_index(index);
    }

    set_node_status(index, node_status::unused);
    while (index > 0) {
      index = parent_index(index);
      set_node_status(index, node_status::splited);
    }
  }

  allocator::node_status inline allocator::get_node_status(size_t index) const
      noexcept {
    return static_cast<node_status>((tree[index / 4] >> (6 - (index % 4 * 2))) %
                                    4);
  }

  void inline allocator::set_node_status(size_t index,
                                         node_status status) noexcept {

    size_t cnt = 6 - (index % 4 * 2);
    uint8_t mask = 3;
    tree[index / 4] &= (~(mask << cnt));
    tree[index / 4] |= (static_cast<uint8_t>(status) << cnt);
  }
} // namespace cuda_buddy
