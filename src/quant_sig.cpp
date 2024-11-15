#include <nanobind/nanobind.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <parallel_hashmap/phmap.h>
#include <mutex>

namespace nb = nanobind;
using namespace std;

class HashesCounter {
private:
    phmap::parallel_flat_hash_map<
    uint64_t, uint32_t,
    std::hash<uint64_t>,
    std::equal_to<uint64_t>,
    std::allocator<std::pair<uint64_t, uint32_t>>,
    6,
    std::mutex> hash_to_count;

public:
    HashesCounter() {}

    void add_hashes(const vector<uint64_t>& hashes) {
        for (uint64_t hash_val : hashes) {
            hash_to_count[hash_val]++;
        }
    }

    uint64_t remove_singletons() {
        uint64_t singletons_counter = 0;
        for (auto it = hash_to_count.begin(); it != hash_to_count.end(); ) {
            if (it->second == 1) {
                it = hash_to_count.erase(it);
                ++singletons_counter;
            } else {
                ++it;
            }
        }
        return singletons_counter;
    }

    void keep_min_abundance(uint32_t min_abundance) {
        for (auto it = hash_to_count.begin(); it != hash_to_count.end(); ) {
            if (it->second < min_abundance) {
                it = hash_to_count.erase(it);
            } else {
                ++it;
            }
        }
    }

    uint64_t size() {
        return hash_to_count.size();
    }

    unordered_map<uint64_t, uint32_t> get_kmers() {
        unordered_map<uint64_t, uint32_t> result;
        for (auto it = hash_to_count.begin(); it != hash_to_count.end(); ++it) {
            result[it->first] = it->second;
        }
        return result;
    }
};

NB_MODULE(_hashes_counter_impl, m) {
    nb::class_<HashesCounter>(m, "HashesCounter")
        .def(nb::init<>())
        .def("add_hashes", &HashesCounter::add_hashes)
        .def("remove_singletons", &HashesCounter::remove_singletons)
        .def("keep_min_abundance", &HashesCounter::keep_min_abundance)
        .def("get_kmers", &HashesCounter::get_kmers)
        .def("size", &HashesCounter::size);
}
