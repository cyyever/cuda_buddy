#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <doctest/doctest.h>

#include "../src/allocator.hpp"

namespace {
  void real_test(cuda_buddy::alloc_location location) {
    {

      cuda_buddy::allocator buddy_allocator(3, location);

      REQUIRE(buddy_allocator.full());

      SUBCASE("alloc and free") {
        for (auto size : {8u, 4u, 2u, 1u, 1u}) {
          std::vector<void *> ptrs;

          for (size_t i = 0; i < ((1ULL << 3) / size); i++) {
            auto ptr = buddy_allocator.alloc(size);
            REQUIRE(ptr);
            REQUIRE(buddy_allocator.in_buddy(ptr));
            ptrs.push_back(ptr);
          }

          auto ptr = buddy_allocator.alloc(1);
          REQUIRE(!ptr);

          for (auto &ptr : ptrs) {
            REQUIRE(buddy_allocator.free(ptr));
          }

          REQUIRE(buddy_allocator.full());
        }
      }

      SUBCASE("alloc and free with alignment") {
        constexpr size_t alignment = 3;
        for (auto size : {4u, 2u, 1u, 1u}) {
          std::vector<void *> ptrs;

          auto ptr = buddy_allocator.alloc(size, alignment);
          REQUIRE(ptr);
          REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
          REQUIRE(buddy_allocator.in_buddy(ptr));
          ptrs.push_back(ptr);

          for (auto &ptr : ptrs) {
            REQUIRE(buddy_allocator.free(ptr));
          }

          REQUIRE(buddy_allocator.full());
        }
      }

      SUBCASE("full alloc") {
        auto ptr = buddy_allocator.alloc(8);
        REQUIRE(ptr);
        REQUIRE(buddy_allocator.in_buddy(ptr));

        REQUIRE(buddy_allocator.free(ptr));
      }
    }
    cudaDeviceReset();
  }
} // namespace

TEST_CASE("host") { real_test(cuda_buddy::alloc_location::host); }
TEST_CASE("device") { real_test(cuda_buddy::alloc_location::device); }
