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

class HashesCounter
{
private:
    phmap::parallel_flat_hash_map<
        uint64_t, uint32_t,
        std::hash<uint64_t>,
        std::equal_to<uint64_t>,
        std::allocator<std::pair<uint64_t, uint32_t>>,
        6,
        std::mutex>
        hash_to_count;

public:
    HashesCounter() {}

    void add_hashes(const vector<uint64_t> &hashes)
    {
        for (uint64_t hash_val : hashes)
        {
            hash_to_count[hash_val]++;
        }
    }

    uint64_t remove_singletons()
    {
        uint64_t singletons_counter = 0;
        for (auto it = hash_to_count.begin(); it != hash_to_count.end();)
        {
            if (it->second == 1)
            {
                it = hash_to_count.erase(it);
                ++singletons_counter;
            }
            else
            {
                ++it;
            }
        }
        return singletons_counter;
    }

    void keep_min_abundance(uint32_t min_abundance)
    {
        for (auto it = hash_to_count.begin(); it != hash_to_count.end();)
        {
            if (it->second < min_abundance)
            {
                it = hash_to_count.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    uint64_t size()
    {
        return hash_to_count.size();
    }

    unordered_map<uint64_t, uint32_t> get_kmers()
    {
        unordered_map<uint64_t, uint32_t> result;
        for (auto it = hash_to_count.begin(); it != hash_to_count.end(); ++it)
        {
            result[it->first] = it->second;
        }
        return result;
    }
};

class WeightedHashesCounter
{
protected:
    phmap::parallel_flat_hash_map<uint64_t, uint32_t, std::hash<uint64_t>,
                                  std::equal_to<uint64_t>, std::allocator<std::pair<uint64_t, uint32_t>>,
                                  6, std::mutex>
        hash_to_count;

    phmap::parallel_flat_hash_map<uint64_t, float, std::hash<uint64_t>, std::equal_to<uint64_t>, std::allocator<std::pair<uint64_t, float>>,
                                  6, std::mutex>
        hash_to_score;

public:
    WeightedHashesCounter() {}

    void add_hashes(const vector<uint64_t> &hashes, const vector<float> &abundances, float mean_abundance)
    {
        const float inv_mean_abundance = 1.0f / mean_abundance; // Precompute reciprocal for faster division
        const size_t n = hashes.size();                         // Cache the size for efficiency

        for (size_t i = 0; i < n; i++)
        {
            double score = abundances[i] * inv_mean_abundance;
            if (score >= 2)
                hash_to_score[hashes[i]] += 2;
            else
                hash_to_score[hashes[i]] += score;
        }
    }

    uint64_t round_scores()
    {
        uint64_t skipped_hashes_after_rounding = 0;
        for (auto it = hash_to_score.begin(); it != hash_to_score.end(); ++it)
        {
            uint32_t rounded_score = static_cast<uint32_t>(it->second);
            if (rounded_score > 1)
            {
                hash_to_count[it->first] = rounded_score;
            }
            else
            {
                skipped_hashes_after_rounding++;
            }
            hash_to_score.erase(it);
        }
        return skipped_hashes_after_rounding;
    }

    unordered_map<uint64_t, uint32_t> get_kmers()
    {
        unordered_map<uint64_t, uint32_t> result;
        for (auto it = hash_to_count.begin(); it != hash_to_count.end(); ++it)
        {
            result[it->first] = it->second;
        }
        return result;
    }

    uint64_t size()
    {
        return hash_to_count.size();
    }

    // keep_min_abundance
    void keep_min_abundance(uint32_t min_abundance)
    {
        for (auto it = hash_to_count.begin(); it != hash_to_count.end();)
        {
            if (it->second < min_abundance)
            {
                it = hash_to_count.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
};

class WeightedHashesCounterUncapped : public WeightedHashesCounter
{
public:
    WeightedHashesCounterUncapped() : WeightedHashesCounter() {}

    void add_hashes(const vector<uint64_t> &hashes, const vector<float> &abundances, float mean_abundance)
    {
        const float inv_mean_abundance = 1.0f / mean_abundance;
        const size_t n = hashes.size();

        for (size_t i = 0; i < n; i++)
        {
            hash_to_score[hashes[i]] += abundances[i] * inv_mean_abundance;
        }
    }
};

NB_MODULE(_hashes_counter_impl, m)
{
    nb::class_<HashesCounter>(m, "HashesCounter")
        .def(nb::init<>())
        .def("add_hashes", &HashesCounter::add_hashes)
        .def("remove_singletons", &HashesCounter::remove_singletons)
        .def("keep_min_abundance", &HashesCounter::keep_min_abundance)
        .def("get_kmers", &HashesCounter::get_kmers)
        .def("size", &HashesCounter::size);

    nb::class_<WeightedHashesCounter>(m, "WeightedHashesCounter")
        .def(nb::init<>())
        .def("add_hashes", &WeightedHashesCounter::add_hashes)
        .def("round_scores", &WeightedHashesCounter::round_scores)
        .def("get_kmers", &WeightedHashesCounter::get_kmers)
        .def("size", &WeightedHashesCounter::size);

    nb::class_<WeightedHashesCounterUncapped>(m, "WeightedHashesCounterUncapped")
        .def(nb::init<>())
        .def("add_hashes", &WeightedHashesCounterUncapped::add_hashes)
        .def("round_scores", &WeightedHashesCounterUncapped::round_scores)
        .def("get_kmers", &WeightedHashesCounterUncapped::get_kmers)
        .def("size", &WeightedHashesCounterUncapped::size)
        .def("keep_min_abundance", &WeightedHashesCounterUncapped::keep_min_abundance);
}