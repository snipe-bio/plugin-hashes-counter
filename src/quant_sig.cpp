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

class SamplesKmerDosageHybridCounter
{

public:
    // Updated to store kmer_dosage as float internally
    phmap::parallel_flat_hash_map<uint64_t, std::tuple<uint32_t, float>,
                                  std::hash<uint64_t>, std::equal_to<uint64_t>,
                                  std::allocator<std::pair<uint64_t, std::tuple<uint32_t, float>>>,
                                  6, std::mutex>
        hash_to_count;

    SamplesKmerDosageHybridCounter() {}

    void add_hashes(const vector<uint64_t> &hashes, const vector<float> &abundances, float mean_abundance)
    {
        if (hashes.size() != abundances.size())
        {
            throw std::invalid_argument("hashes and abundances vectors must be of the same size.");
        }

        const float inv_mean_abundance = 1.0f / mean_abundance;
        const size_t n = hashes.size();
        const uint32_t additional_sample_count = 1;

        for (size_t i = 0; i < n; i++)
        {
            float kmer_dosage = abundances[i] * inv_mean_abundance;

            // Optional: Handle cases where dosage might be negative or exceed expected ranges
            if (kmer_dosage < 0.0f)
            {
                throw std::invalid_argument("kmer_dosage cannot be negative.");
            }

            // Use try_emplace to optimize insertion
            auto [it, inserted] = hash_to_count.try_emplace(hashes[i], 1, kmer_dosage);
            if (!inserted)
            {
                std::get<0>(it->second)++;
                std::get<1>(it->second) += kmer_dosage;
            }
        }
    }

    uint64_t size() const
    {
        return hash_to_count.size();
    }

    // round scores and remove entries with count less than 2
    uint64_t round_scores()
    {
        uint64_t skipped_hashes_after_rounding = 0;
        for (auto it = hash_to_count.begin(); it != hash_to_count.end();)
        {
            uint32_t &count = std::get<0>(it->second);
            if (count < 2)
            {
                skipped_hashes_after_rounding++;
                it = hash_to_count.erase(it);
            }
            else
            {
                ++it;
            }
        }
        return skipped_hashes_after_rounding;
    }

    unordered_map<uint64_t, std::tuple<uint32_t, uint32_t>> get_kmers() const
    {
        unordered_map<uint64_t, std::tuple<uint32_t, uint32_t>> result;
        result.reserve(hash_to_count.size()); // Optimize by reserving space
        for (const auto &it : hash_to_count)
        {
            // Convert float dosage to uint32_t, using rounding to nearest integer
            uint32_t rounded_dosage = static_cast<uint32_t>(std::round(std::get<1>(it.second)));
            result[it.first] = std::make_tuple(std::get<0>(it.second), rounded_dosage);
        }
        return result;
    }

    vector<uint64_t> get_hashes() const
    {
        vector<uint64_t> result;
        result.reserve(hash_to_count.size());
        for (const auto &it : hash_to_count)
        {
            result.push_back(it.first);
        }
        return result;
    }

    vector<uint32_t> get_sample_counts() const
    {
        vector<uint32_t> result;
        result.reserve(hash_to_count.size());
        for (const auto &it : hash_to_count)
        {
            result.push_back(std::get<0>(it.second));
        }
        return result;
    }

    vector<uint32_t> get_kmer_dosages() const
    {
        vector<uint32_t> result;
        result.reserve(hash_to_count.size());
        for (const auto &it : hash_to_count)
        {
            uint32_t rounded_dosage = static_cast<uint32_t>(std::round(std::get<1>(it.second)));
            result.push_back(rounded_dosage);
        }
        return result;
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

    nb::class_<SamplesKmerDosageHybridCounter>(m, "SamplesKmerDosageHybridCounter")
        .def(nb::init<>())
        .def("add_hashes", &SamplesKmerDosageHybridCounter::add_hashes)
        .def("round_scores", &SamplesKmerDosageHybridCounter::round_scores)
        .def("size", &SamplesKmerDosageHybridCounter::size)
        .def("get_kmers", &SamplesKmerDosageHybridCounter::get_kmers)
        .def("get_hashes", &SamplesKmerDosageHybridCounter::get_hashes)
        .def("get_sample_counts", &SamplesKmerDosageHybridCounter::get_sample_counts)
        .def("get_kmer_dosages", &SamplesKmerDosageHybridCounter::get_kmer_dosages);
}