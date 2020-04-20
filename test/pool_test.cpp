#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <doctest/doctest.h>
#include <thread>

#include "../src/pool.hpp"

namespace {
  auto logger = spdlog::stdout_color_mt("cuda_buddy");
  void real_test(int gpu_no) {

    {
      SUBCASE("concurrent alloc and free") {
        std::vector<std::thread> thds;

        cuda_buddy::pool buddy_pool(gpu_no);
        for (int i = 0; i < 2; i++) {
          thds.emplace_back([&buddy_pool]() {
            std::vector<void *> ptrs;

            for (auto size : {4u, 2u, 1u, 1u}) {
              auto ptr = buddy_pool.alloc(size);
              REQUIRE(ptr);
              ptrs.push_back(ptr);
            }

            for (auto &ptr : ptrs) {
              bool res = buddy_pool.free(ptr);
              CHECK(res);
            }
          });
        }

        for (auto &thd : thds) {
          thd.join();
        }
        CHECK(buddy_pool.full());
      }

      SUBCASE("alloc and free with alignment") {
        constexpr size_t alignment = 3;
        cuda_buddy::pool buddy_pool(gpu_no);
        std::vector<void *> ptrs;
        for (auto size : {4u, 2u, 1u, 1u}) {

          auto ptr = buddy_pool.alloc(size, alignment);

          REQUIRE(ptr);
          REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
          ptrs.push_back(ptr);
        }
        for (auto &ptr : ptrs) {
          bool res = buddy_pool.free(ptr);
          CHECK(res);
        }
        CHECK(buddy_pool.full());
      }
    }

    cuda_buddy::pool::release_global_pool(gpu_no);
    cudaDeviceReset();
  }
} // namespace
TEST_CASE("device") {
  cuda_buddy::pool::set_device_pool_size(cuda_buddy::pool::buddy_block_level +
                                         2);
  real_test(0);
}

TEST_CASE("host") {
  cuda_buddy::pool::set_host_pool_size(cuda_buddy::pool::buddy_block_level + 2);
  real_test(-1);
}
