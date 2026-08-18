/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <streamseed/StreamSeedCore.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSWIncremental.h>

namespace faiss {
namespace streamseed {

uint64_t compute_semantic_signature(const float* query, idx_t dim) {
    if (!query || dim <= 0) {
        return 0;
    }

    constexpr int sample_dims = 8;
    constexpr float quant_scale = 16.0f;
    uint64_t signature = 0;
    const uint64_t u_dim = static_cast<uint64_t>(dim);

    for (int t = 0; t < sample_dims; ++t) {
        const uint64_t pick = (static_cast<uint64_t>(t) * 9973ULL + 13ULL) % u_dim;
        const float v = query[static_cast<idx_t>(pick)];
        const int q = static_cast<int>(std::lrint(v * quant_scale));
        const int clamped = std::max(-128, std::min(127, q));
        const uint8_t packed = static_cast<uint8_t>(clamped + 128);
        signature |= static_cast<uint64_t>(packed) << (t * 8);
    }

    return signature;
}

idx_t compute_semantic_slot_key(
        const float* query,
        idx_t dim,
        int hint_table_slots,
        idx_t fallback_query_id) {
    if (!query || dim <= 0 || hint_table_slots <= 0) {
        return fallback_query_id;
    }

    const uint64_t signature = compute_semantic_signature(query, dim);
    uint64_t h = 1469598103934665603ULL;

    for (int t = 0; t < 8; ++t) {
        const uint64_t token =
                ((signature >> (t * 8)) & 0xFFULL) ^
                (static_cast<uint64_t>(t + 1) * 11400714819323198485ULL);
        h ^= token;
        h *= 1099511628211ULL;
    }

    return static_cast<idx_t>(h % static_cast<uint64_t>(hint_table_slots));
}

bool OptimizationConfig::use_dictionary() const {
    return streamseed_mode == STREAMSEED_CORE && hint_table_slots > 0;
}

bool OptimizationConfig::use_two_storage() const {
    return streamseed_mode == STREAMSEED_TWO_STORAGE && hint_table_slots > 0;
}

bool OptimizationConfig::streamseed_enabled() const {
    return streamseed_mode == STREAMSEED_CORE ||
            streamseed_mode == STREAMSEED_TWO_STORAGE;
}

struct StoredSeedRecord {
    uint64_t record_id = 0;
    std::vector<idx_t> result_ids;
    idx_t query_key = -1;
    uint64_t signature = 0;
    uint64_t birth_tick = 0;
    uint64_t last_refresh_tick = 0;
    uint64_t last_success_tick = 0;
    uint32_t successful_reuses = 0;
};

struct TwoTierSeedStore {
    idx_t record_k = 0;
    size_t hot_capacity = 0;
    size_t semantic_slots = 0;
    size_t semantic_capacity = 0;
    std::vector<StoredSeedRecord> hot;
    std::unordered_map<idx_t, size_t> hot_exact;
    std::vector<std::vector<StoredSeedRecord>> semantic;
    uint64_t clock = 0;
    uint64_t next_record_id = 1;
    omp_lock_t lock;

    TwoTierSeedStore() {
        omp_init_lock(&lock);
    }

    ~TwoTierSeedStore() {
        omp_destroy_lock(&lock);
    }

    void rebuild_hot_exact() {
        hot_exact.clear();
        for (size_t i = 0; i < hot.size(); ++i) {
            if (hot[i].query_key >= 0) {
                hot_exact[hot[i].query_key] = i;
            }
        }
    }

    void reset(
            idx_t k,
            size_t new_hot_capacity,
            size_t new_semantic_slots,
            size_t new_semantic_capacity) {
        record_k = k;
        hot_capacity = new_hot_capacity;
        semantic_slots = new_semantic_slots;
        semantic_capacity = new_semantic_capacity;
        hot.clear();
        hot.reserve(hot_capacity);
        hot_exact.clear();
        semantic.assign(semantic_slots, {});
        for (auto& bucket : semantic) {
            bucket.reserve(semantic_capacity);
        }
        clock = 0;
        next_record_id = 1;
    }

    size_t semantic_size() const {
        size_t total = 0;
        for (const auto& bucket : semantic) {
            total += bucket.size();
        }
        return total;
    }
};

namespace {

using storage_idx_t = HNSWIncremental::storage_idx_t;

constexpr int ADAPTIVE_GATE_OFF = 0;
constexpr int ADAPTIVE_GATE_BATCH = 1;

float quantile_inplace(std::vector<float>& values, float q) {
    if (values.empty()) {
        return 0.0f;
    }
    const float q_clamped = std::max(0.0f, std::min(1.0f, q));
    const size_t n = values.size();
    const size_t kth = static_cast<size_t>(q_clamped * static_cast<float>(n - 1));
    std::nth_element(values.begin(), values.begin() + kth, values.end());
    return values[kth];
}

float signature_similarity(uint64_t a, uint64_t b) {
    int l1 = 0;
    for (int t = 0; t < 8; ++t) {
        const int av = static_cast<int>((a >> (t * 8)) & 0xFFULL);
        const int bv = static_cast<int>((b >> (t * 8)) & 0xFFULL);
        l1 += std::abs(av - bv);
    }
    constexpr float max_l1 = 8.0f * 255.0f;
    return 1.0f - static_cast<float>(l1) / max_l1;
}

size_t semantic_slot_from_signature(uint64_t signature, size_t slots) {
    if (slots == 0) {
        return 0;
    }
    uint64_t h = 1469598103934665603ULL;
    for (int t = 0; t < 8; ++t) {
        const uint64_t token =
                ((signature >> (t * 8)) & 0xFFULL) ^
                (static_cast<uint64_t>(t + 1) * 11400714819323198485ULL);
        h ^= token;
        h *= 1099511628211ULL;
    }
    return static_cast<size_t>(h % static_cast<uint64_t>(slots));
}

std::vector<size_t> semantic_probe_slots(
        uint64_t signature,
        size_t slots,
        size_t probe_count) {
    std::vector<size_t> probes;
    if (slots == 0 || probe_count == 0) {
        return probes;
    }
    probes.reserve(std::min(slots, probe_count));
    for (size_t p = 0; p < probe_count && probes.size() < slots; ++p) {
        const uint64_t probed_signature =
                p == 0 ? signature : signature ^ (1ULL << ((p - 1) % 64));
        const size_t slot = semantic_slot_from_signature(probed_signature, slots);
        if (std::find(probes.begin(), probes.end(), slot) == probes.end()) {
            probes.push_back(slot);
        }
    }
    return probes;
}

struct DictionarySeedSource : ISeedSource {
    std::vector<std::vector<idx_t>>& dictionary;
    std::vector<std::vector<idx_t>>& dictionary_owner_query;
    std::vector<std::vector<uint64_t>>& dictionary_owner_signature;
    std::vector<std::vector<float>>& dictionary_score;
    std::vector<std::vector<uint64_t>>& dictionary_age;
    std::vector<omp_lock_t>& dictionary_locks;
    uint64_t& dictionary_clock;
    uint64_t current_batch_round;
    float& adaptive_m_gate;
    float& adaptive_o_gate;
    bool level1_only;
    int adaptive_gate_mode;
    float gate_m_quantile;
    float gate_o_quantile;
    int gate_min_samples;
    int slot_capacity;
    mutable std::vector<float> batch_m_samples;
    mutable std::vector<float> batch_o_samples;
    mutable size_t batch_level2_attempt = 0;
    mutable size_t batch_level2_pass = 0;
    mutable size_t batch_level2_block = 0;
    mutable omp_lock_t adaptive_lock;

    DictionarySeedSource(
            std::vector<std::vector<idx_t>>& dictionary,
                        std::vector<std::vector<idx_t>>& dictionary_owner_query,
                        std::vector<std::vector<uint64_t>>& dictionary_owner_signature,
                        std::vector<std::vector<float>>& dictionary_score,
                        std::vector<std::vector<uint64_t>>& dictionary_age,
            std::vector<omp_lock_t>& dictionary_locks,
                        uint64_t& dictionary_clock,
                        uint64_t current_batch_round,
                        float& adaptive_m_gate,
                        float& adaptive_o_gate,
            bool level1_only,
                        int adaptive_gate_mode,
                        float gate_m_quantile,
                        float gate_o_quantile,
                        int gate_min_samples,
                        int slot_capacity)
            : dictionary(dictionary),
                            dictionary_owner_query(dictionary_owner_query),
                            dictionary_owner_signature(dictionary_owner_signature),
              dictionary_score(dictionary_score),
              dictionary_age(dictionary_age),
              dictionary_locks(dictionary_locks),
                            dictionary_clock(dictionary_clock),
              current_batch_round(current_batch_round),
              adaptive_m_gate(adaptive_m_gate),
              adaptive_o_gate(adaptive_o_gate),
              level1_only(level1_only),
              adaptive_gate_mode(adaptive_gate_mode),
              gate_m_quantile(gate_m_quantile),
              gate_o_quantile(gate_o_quantile),
              gate_min_samples(std::max(1, gate_min_samples)),
                            slot_capacity(std::max(1, slot_capacity)) {
        omp_init_lock(&adaptive_lock);
    }

    ~DictionarySeedSource() override {
        omp_destroy_lock(&adaptive_lock);
    }

    bool available(idx_t query_id, idx_t slot_key) const override {
        if (dictionary.empty() || query_id < 0 || slot_key < 0) {
            return false;
        }
        return true;
    }

    const std::vector<idx_t>& get(
            idx_t query_id,
            idx_t slot_key,
            const float* query,
            idx_t dim,
            SeedLookupMetadata* metadata) const override {
        static thread_local std::vector<idx_t> tls_matched_ids;
        tls_matched_ids.clear();
        if (metadata) {
            *metadata = SeedLookupMetadata{};
        }
        if (!available(query_id, slot_key)) {
            return tls_matched_ids;
        }

        const uint64_t query_signature = compute_semantic_signature(query, dim);

        const size_t slot = static_cast<size_t>(slot_key) % dictionary.size();
        omp_set_lock(&dictionary_locks[slot]);
        const std::vector<idx_t>& slot_ids = dictionary[slot];
        const std::vector<idx_t>& slot_owners = dictionary_owner_query[slot];
        const std::vector<uint64_t>& slot_signatures =
                dictionary_owner_signature[slot];
        const std::vector<float>& slot_counts = dictionary_score[slot];

        if (slot_owners.empty()) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        if (slot_ids.size() % slot_owners.size() != 0 ||
            slot_signatures.size() != slot_owners.size() ||
            slot_counts.size() != slot_owners.size()) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        const size_t record_len = slot_ids.size() / slot_owners.size();
        if (record_len == 0) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        for (size_t i = 0; i < slot_owners.size(); ++i) {
            if (slot_owners[i] == query_id) {
                const size_t begin = i * record_len;
                tls_matched_ids.assign(
                        slot_ids.begin() + begin,
                        slot_ids.begin() + begin + record_len);
                if (metadata) {
                    metadata->kind = SeedLookupKind::LEGACY_EXACT;
                    metadata->source_query_key = slot_owners[i];
                    metadata->source_signature = slot_signatures[i];
                    metadata->semantic_bucket = slot;
                }
                omp_unset_lock(&dictionary_locks[slot]);
                return tls_matched_ids;
            }
        }

        if (level1_only) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        // Cold-start protection: disable secondary strategy in early rounds.
        constexpr uint64_t warmup_rounds = 5;
        if (current_batch_round <= warmup_rounds) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        float max_count = 1.0f;
        for (float c : slot_counts) {
            if (c > max_count) {
                max_count = c;
            }
        }

        size_t best_i = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        constexpr float alpha = 0.7f;
        for (size_t i = 0; i < slot_owners.size(); ++i) {
            const float sim = signature_similarity(query_signature, slot_signatures[i]);
            const float cnt = slot_counts[i] / max_count;
            const float s = alpha * sim + (1.0f - alpha) * cnt;
            if (s > best_score) {
                best_score = s;
                best_i = i;
            }
        }

        const size_t begin = best_i * record_len;

        tls_matched_ids.assign(
                slot_ids.begin() + begin,
                slot_ids.begin() + begin + record_len);
        if (metadata) {
            metadata->kind = SeedLookupKind::LEGACY_SHARED;
            metadata->source_query_key = slot_owners[best_i];
            metadata->source_signature = slot_signatures[best_i];
            metadata->semantic_bucket = slot;
        }

        omp_unset_lock(&dictionary_locks[slot]);
        return tls_matched_ids;
    }

    void writeback(const SeedWritebackRecord& record) override {
        if (!available(record.query_id, record.slot_key) || record.k <= 0 || !record.idxi ||
            !record.simi || dictionary_locks.empty()) {
            return;
        }

        if (record.idxi[0] < 0 ||
            !std::isfinite(static_cast<double>(record.simi[0]))) {
            return;
        }

        const size_t slot =
            static_cast<size_t>(record.slot_key) % dictionary.size();
        const float new_count = record.hint_used ? 1.0f : 0.0f;
        constexpr uint64_t stale_window = 4096;
        constexpr float age_weight = 0.25f;

        uint64_t tick = 0;
#pragma omp atomic capture
        tick = ++dictionary_clock;

        omp_set_lock(&dictionary_locks[slot]);
        std::vector<idx_t>& slot_ids = dictionary[slot];
        std::vector<idx_t>& slot_owners = dictionary_owner_query[slot];
        std::vector<uint64_t>& slot_signatures = dictionary_owner_signature[slot];
        std::vector<float>& slot_scores = dictionary_score[slot];
        std::vector<uint64_t>& slot_ages = dictionary_age[slot];
        const size_t record_len = static_cast<size_t>(record.k);

        if (record_len == 0) {
            omp_unset_lock(&dictionary_locks[slot]);
            return;
        }

        if (slot_scores.size() != slot_ages.size() ||
            slot_scores.size() != slot_owners.size() ||
            slot_scores.size() != slot_signatures.size() ||
            slot_ids.size() != slot_scores.size() * record_len) {
            slot_ids.clear();
            slot_owners.clear();
            slot_signatures.clear();
            slot_scores.clear();
            slot_ages.clear();
        }

        for (size_t i = 0; i < slot_owners.size(); ++i) {
            if (slot_owners[i] == record.query_id) {
                const size_t begin = i * record_len;
                std::copy(record.idxi, record.idxi + record.k, slot_ids.begin() + begin);
                if (record.hint_used) {
                    slot_scores[i] += 1.0f;
                }
                slot_signatures[i] = record.owner_signature;
                slot_ages[i] = tick;
                omp_unset_lock(&dictionary_locks[slot]);
                return;
            }
        }

        const size_t capacity = static_cast<size_t>(slot_capacity);
        if (slot_scores.size() < capacity) {
            slot_ids.insert(slot_ids.end(), record.idxi, record.idxi + record.k);
            slot_owners.push_back(record.query_id);
            slot_signatures.push_back(record.owner_signature);
            slot_scores.push_back(new_count);
            slot_ages.push_back(tick);
            omp_unset_lock(&dictionary_locks[slot]);
            return;
        }

        size_t victim = 0;
        float victim_key = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < slot_scores.size(); ++i) {
            uint64_t age_delta = tick > slot_ages[i] ? tick - slot_ages[i] : 0;
            if (age_delta > stale_window) {
                age_delta = stale_window;
            }
            const float age_term =
                    static_cast<float>(age_delta) / static_cast<float>(stale_window);
            const float keep_key = slot_scores[i] - age_weight * age_term;
            if (keep_key < victim_key) {
                victim_key = keep_key;
                victim = i;
            }
        }

        const bool victim_stale = tick > slot_ages[victim] + stale_window;
        const bool should_replace = victim_stale ||
                (new_count > slot_scores[victim]) ||
                (slot_scores[victim] <= 0.0f);
        if (should_replace) {
            const size_t begin = victim * record_len;
            std::copy(record.idxi, record.idxi + record.k, slot_ids.begin() + begin);
            slot_owners[victim] = record.query_id;
            slot_signatures[victim] = record.owner_signature;
            slot_scores[victim] = new_count;
            slot_ages[victim] = tick;
        }
        omp_unset_lock(&dictionary_locks[slot]);
    }

    void on_batch_end(bool verbose, int64_t i0, int64_t i1) override {
        (void)verbose;
        (void)i0;
        (void)i1;
    }
};

struct TwoTierSeedSource : ISeedSource {
    std::shared_ptr<TwoTierSeedStore> store;
    int probe_count;
    float retrieval_threshold;
    float signature_weight;
    bool semantic_retrieval_enabled;
    int promotion_hits;
    uint64_t demotion_window;
    mutable size_t hot_exact_hits = 0;
    mutable size_t hot_signature_hits = 0;
    mutable size_t semantic_shared_hits = 0;
    mutable size_t semantic_same_query_hits = 0;
    mutable size_t semantic_cross_query_hits = 0;
    mutable size_t semantic_anonymous_hits = 0;
    mutable size_t semantic_threshold_rejects = 0;
    size_t promotions = 0;
    size_t demotions = 0;
    size_t semantic_evictions = 0;

    TwoTierSeedSource(
            std::shared_ptr<TwoTierSeedStore> store,
            int probe_count,
            float retrieval_threshold,
            float signature_weight,
            int hint_semantic_enabled,
            int promotion_hits,
            int demotion_window)
            : store(std::move(store)),
              probe_count(std::max(1, probe_count)),
              retrieval_threshold(
                      std::max(0.0f, std::min(1.0f, retrieval_threshold))),
              signature_weight(
                      std::max(0.0f, std::min(1.0f, signature_weight))),
              semantic_retrieval_enabled(hint_semantic_enabled != 0),
              promotion_hits(std::max(1, promotion_hits)),
              demotion_window(static_cast<uint64_t>(std::max(1, demotion_window))) {}

    bool available(idx_t query_id, idx_t slot_key) const override {
        (void)query_id;
        (void)slot_key;
        return store && store->semantic_slots > 0 && store->semantic_capacity > 0;
    }

    static float keep_score(
            const StoredSeedRecord& seed,
            uint64_t now,
            uint64_t age_window) {
        const float reuse = static_cast<float>(seed.successful_reuses);
        const uint64_t last_active_tick = std::max(
                seed.last_refresh_tick, seed.last_success_tick);
        const uint64_t age = now > last_active_tick
                ? now - last_active_tick
                : 0;
        const float normalized_age = std::min(
                1.0f,
                static_cast<float>(age) /
                        static_cast<float>(std::max<uint64_t>(1, age_window)));
        constexpr float xi = 0.75f;
        return xi * reuse - (1.0f - xi) * normalized_age;
    }

    const std::vector<idx_t>& get(
            idx_t query_id,
            idx_t slot_key,
            const float* query,
            idx_t dim,
            SeedLookupMetadata* metadata) const override {
        (void)slot_key;
        static thread_local std::vector<idx_t> tls_matched_ids;
        tls_matched_ids.clear();
        if (metadata) {
            *metadata = SeedLookupMetadata{};
        }
        if (!available(query_id, slot_key) || !query || dim <= 0) {
            return tls_matched_ids;
        }

        const uint64_t query_signature = semantic_retrieval_enabled
                ? compute_semantic_signature(query, dim)
                : 0;
        omp_set_lock(&store->lock);

        if (query_id >= 0) {
            const auto exact_it = store->hot_exact.find(query_id);
            if (exact_it != store->hot_exact.end() &&
                exact_it->second < store->hot.size()) {
                const StoredSeedRecord& seed = store->hot[exact_it->second];
                tls_matched_ids = seed.result_ids;
                if (metadata) {
                    metadata->kind = SeedLookupKind::HOT_EXACT;
                    metadata->record_id = seed.record_id;
                    metadata->source_query_key = seed.query_key;
                    metadata->source_signature = seed.signature;
                }
                hot_exact_hits += 1;
                omp_unset_lock(&store->lock);
                return tls_matched_ids;
            }
        }

        if (!semantic_retrieval_enabled) {
            omp_unset_lock(&store->lock);
            return tls_matched_ids;
        }

        if (query_id < 0 && !store->hot.empty()) {
            uint32_t max_reuse = 1;
            for (const auto& seed : store->hot) {
                max_reuse = std::max(max_reuse, seed.successful_reuses);
            }
            float best_score = -std::numeric_limits<float>::infinity();
            const StoredSeedRecord* best_seed = nullptr;
            for (const auto& seed : store->hot) {
                const float sim = signature_similarity(query_signature, seed.signature);
                const float reuse = static_cast<float>(seed.successful_reuses) /
                        static_cast<float>(max_reuse);
                const float score = signature_weight * sim +
                        (1.0f - signature_weight) * reuse;
                if (score > best_score) {
                    best_score = score;
                    best_seed = &seed;
                }
            }
            if (best_seed && best_score >= retrieval_threshold) {
                tls_matched_ids = best_seed->result_ids;
                if (metadata) {
                    metadata->kind = SeedLookupKind::HOT_SIGNATURE;
                    metadata->record_id = best_seed->record_id;
                    metadata->source_query_key = best_seed->query_key;
                    metadata->source_signature = best_seed->signature;
                }
                hot_signature_hits += 1;
                omp_unset_lock(&store->lock);
                return tls_matched_ids;
            }
        }

        const std::vector<size_t> probes = semantic_probe_slots(
                query_signature,
                store->semantic_slots,
                static_cast<size_t>(probe_count));
        uint32_t max_reuse = 1;
        for (size_t bucket_id : probes) {
            for (const auto& seed : store->semantic[bucket_id]) {
                max_reuse = std::max(max_reuse, seed.successful_reuses);
            }
        }
        float best_score = -std::numeric_limits<float>::infinity();
        const StoredSeedRecord* best_seed = nullptr;
        size_t best_bucket = 0;
        for (size_t bucket_id : probes) {
            for (const auto& seed : store->semantic[bucket_id]) {
                const float sim = signature_similarity(query_signature, seed.signature);
                const float reuse = static_cast<float>(seed.successful_reuses) /
                        static_cast<float>(max_reuse);
                const float score = signature_weight * sim +
                        (1.0f - signature_weight) * reuse;
                if (score > best_score) {
                    best_score = score;
                    best_seed = &seed;
                    best_bucket = bucket_id;
                }
            }
        }
        if (best_seed && best_score >= retrieval_threshold) {
            tls_matched_ids = best_seed->result_ids;
            if (metadata) {
                metadata->kind = SeedLookupKind::SEMANTIC_SHARED;
                metadata->record_id = best_seed->record_id;
                metadata->source_query_key = best_seed->query_key;
                metadata->source_signature = best_seed->signature;
                metadata->semantic_bucket = best_bucket;
                metadata->same_query_match =
                        query_id >= 0 && best_seed->query_key == query_id;
            }
            semantic_shared_hits += 1;
            if (query_id < 0) {
                semantic_anonymous_hits += 1;
            } else if (best_seed->query_key == query_id) {
                semantic_same_query_hits += 1;
            } else {
                semantic_cross_query_hits += 1;
            }
        } else if (best_seed) {
            semantic_threshold_rejects += 1;
        }

        omp_unset_lock(&store->lock);
        return tls_matched_ids;
    }

    void insert_semantic_locked(StoredSeedRecord seed) {
        if (store->semantic_slots == 0 || store->semantic_capacity == 0) {
            return;
        }
        const size_t bucket_id = semantic_slot_from_signature(
                seed.signature, store->semantic_slots);
        auto& bucket = store->semantic[bucket_id];
        if (seed.query_key >= 0) {
            for (auto& existing : bucket) {
                if (existing.query_key == seed.query_key) {
                    existing = std::move(seed);
                    return;
                }
            }
        }
        if (bucket.size() < store->semantic_capacity) {
            bucket.push_back(std::move(seed));
            return;
        }

        size_t victim = 0;
        float victim_score = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < bucket.size(); ++i) {
            const float score = keep_score(bucket[i], store->clock, demotion_window);
            if (score < victim_score) {
                victim_score = score;
                victim = i;
            }
        }
        const float incoming_score = keep_score(seed, store->clock, demotion_window);
        const bool victim_stale = store->clock >
                bucket[victim].last_refresh_tick + demotion_window;
        if (victim_stale || incoming_score >= victim_score) {
            bucket[victim] = std::move(seed);
            semantic_evictions += 1;
        }
    }

    void insert_hot_locked(StoredSeedRecord seed) {
        if (store->hot_capacity == 0) {
            insert_semantic_locked(std::move(seed));
            return;
        }

        if (seed.query_key >= 0) {
            const auto exact_it = store->hot_exact.find(seed.query_key);
            if (exact_it != store->hot_exact.end() &&
                exact_it->second < store->hot.size()) {
                store->hot[exact_it->second] = std::move(seed);
                return;
            }
        }

        if (store->hot.size() >= store->hot_capacity) {
            size_t victim = 0;
            float victim_score = std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < store->hot.size(); ++i) {
                const float score = keep_score(
                        store->hot[i], store->clock, demotion_window);
                if (score < victim_score) {
                    victim_score = score;
                    victim = i;
                }
            }
            StoredSeedRecord demoted = std::move(store->hot[victim]);
            store->hot.erase(store->hot.begin() + victim);
            insert_semantic_locked(std::move(demoted));
            demotions += 1;
            store->rebuild_hot_exact();
        }

        store->hot.push_back(std::move(seed));
        const size_t inserted = store->hot.size() - 1;
        if (store->hot[inserted].query_key >= 0) {
            store->hot_exact[store->hot[inserted].query_key] = inserted;
        }
    }

    bool promote_locked(size_t bucket_id, uint64_t record_id) {
        if (store->hot_capacity == 0 || bucket_id >= store->semantic.size()) {
            return false;
        }
        auto& bucket = store->semantic[bucket_id];
        auto it = std::find_if(
                bucket.begin(), bucket.end(),
                [record_id](const StoredSeedRecord& seed) {
                    return seed.record_id == record_id;
                });
        if (it == bucket.end() ||
            it->successful_reuses < static_cast<uint32_t>(promotion_hits)) {
            return false;
        }

        if (store->hot.size() >= store->hot_capacity) {
            float victim_score = std::numeric_limits<float>::infinity();
            for (const auto& hot_seed : store->hot) {
                victim_score = std::min(
                        victim_score,
                        keep_score(hot_seed, store->clock, demotion_window));
            }
            const float candidate_score =
                    keep_score(*it, store->clock, demotion_window);
            // Require one additional successful reuse worth of hysteresis.
            // Equal-frequency fixed queries should not continuously swap tiers.
            constexpr float promotion_margin = 0.75f;
            if (candidate_score <= victim_score + promotion_margin) {
                return false;
            }
        }

        StoredSeedRecord promoted = std::move(*it);
        bucket.erase(it);
        if (promoted.query_key >= 0 &&
            store->hot_exact.find(promoted.query_key) != store->hot_exact.end()) {
            const size_t hot_index = store->hot_exact[promoted.query_key];
            store->hot[hot_index] = std::move(promoted);
            store->rebuild_hot_exact();
            promotions += 1;
            return true;
        }

        insert_hot_locked(std::move(promoted));
        promotions += 1;
        return true;
    }

    void writeback(const SeedWritebackRecord& record) override {
        if (!store || record.k <= 0 || !record.idxi || !record.simi ||
            record.idxi[0] < 0 ||
            !std::isfinite(static_cast<double>(record.simi[0]))) {
            return;
        }

        omp_set_lock(&store->lock);
        const uint64_t tick = ++store->clock;

        if (record.hint_used && record.selected_seed.record_id != 0) {
            if (record.selected_seed.kind == SeedLookupKind::HOT_EXACT &&
                record.selected_seed.source_query_key >= 0) {
                const auto exact_it = store->hot_exact.find(
                        record.selected_seed.source_query_key);
                if (exact_it != store->hot_exact.end() &&
                    exact_it->second < store->hot.size()) {
                    auto& seed = store->hot[exact_it->second];
                    if (seed.record_id == record.selected_seed.record_id) {
                        seed.successful_reuses += 1;
                        seed.last_success_tick = tick;
                    }
                }
            } else if (record.selected_seed.kind ==
                       SeedLookupKind::HOT_SIGNATURE) {
                for (auto& seed : store->hot) {
                    if (seed.record_id == record.selected_seed.record_id) {
                        seed.successful_reuses += 1;
                        seed.last_success_tick = tick;
                        break;
                    }
                }
            } else if (record.selected_seed.kind ==
                               SeedLookupKind::SEMANTIC_SHARED &&
                       record.selected_seed.semantic_bucket < store->semantic.size()) {
                auto& bucket = store->semantic[record.selected_seed.semantic_bucket];
                for (auto& seed : bucket) {
                    if (seed.record_id == record.selected_seed.record_id) {
                        seed.successful_reuses += 1;
                        seed.last_success_tick = tick;
                        break;
                    }
                }
                promote_locked(
                        record.selected_seed.semantic_bucket,
                        record.selected_seed.record_id);
            }
        }

        if (record.query_id >= 0) {
            const auto hot_it = store->hot_exact.find(record.query_id);
            if (hot_it != store->hot_exact.end() &&
                hot_it->second < store->hot.size()) {
                auto& seed = store->hot[hot_it->second];
                seed.result_ids.assign(record.idxi, record.idxi + record.k);
                seed.signature = record.owner_signature;
                seed.last_refresh_tick = tick;
                omp_unset_lock(&store->lock);
                return;
            }
        }

        if (record.query_id >= 0 && !record.hint_used) {
            StoredSeedRecord seed;
            seed.record_id = store->next_record_id++;
            seed.result_ids.assign(record.idxi, record.idxi + record.k);
            seed.query_key = record.query_id;
            seed.signature = record.owner_signature;
            seed.birth_tick = tick;
            seed.last_refresh_tick = tick;
            if (store->hot.size() < store->hot_capacity) {
                insert_hot_locked(std::move(seed));
            } else {
                insert_semantic_locked(std::move(seed));
            }
            omp_unset_lock(&store->lock);
            return;
        }

        const size_t bucket_id = semantic_slot_from_signature(
                record.owner_signature, store->semantic_slots);
        auto& bucket = store->semantic[bucket_id];
        auto existing = std::find_if(
                bucket.begin(), bucket.end(),
                [&record](const StoredSeedRecord& seed) {
                    return record.query_id >= 0
                            ? seed.query_key == record.query_id
                            : seed.query_key < 0 &&
                                    seed.signature == record.owner_signature;
                });
        if (existing != bucket.end()) {
            existing->result_ids.assign(record.idxi, record.idxi + record.k);
            existing->signature = record.owner_signature;
            existing->last_refresh_tick = tick;
        } else {
            StoredSeedRecord seed;
            seed.record_id = store->next_record_id++;
            seed.result_ids.assign(record.idxi, record.idxi + record.k);
            seed.query_key = record.query_id;
            seed.signature = record.owner_signature;
            seed.birth_tick = tick;
            seed.last_refresh_tick = tick;
            insert_semantic_locked(std::move(seed));
        }
        omp_unset_lock(&store->lock);
    }

    void on_batch_end(bool verbose, int64_t i0, int64_t i1) override {
        omp_set_lock(&store->lock);
        for (size_t i = store->hot.size(); i > 0; --i) {
            const size_t index = i - 1;
            const auto& seed = store->hot[index];
            const uint64_t last_active_tick = std::max(
                    seed.last_refresh_tick, seed.last_success_tick);
            if (store->clock > last_active_tick + demotion_window) {
                StoredSeedRecord demoted = std::move(store->hot[index]);
                store->hot.erase(store->hot.begin() + index);
                insert_semantic_locked(std::move(demoted));
                demotions += 1;
            }
        }
        store->rebuild_hot_exact();
        const size_t hot_size = store->hot.size();
        const size_t semantic_size = store->semantic_size();
        omp_unset_lock(&store->lock);

        if (verbose) {
            printf("streamseed_two_storage_config signature_weight=%.3f retrieval_threshold=%.3f probe_count=%d semantic_enabled=%d\n",
                   signature_weight, retrieval_threshold, probe_count,
                   semantic_retrieval_enabled ? 1 : 0);
            printf("streamseed_two_storage_hits hot_exact=%zu hot_signature=%zu semantic=%zu in chunk [%" PRId64 ", %" PRId64 ")\n",
                   hot_exact_hits, hot_signature_hits, semantic_shared_hits, i0, i1);
            printf("streamseed_two_storage_semantic signature_reuse=%zu same_query=%zu cross_query=%zu anonymous=%zu threshold_reject=%zu in chunk [%" PRId64 ", %" PRId64 ")\n",
                   semantic_shared_hits,
                   semantic_same_query_hits,
                   semantic_cross_query_hits,
                   semantic_anonymous_hits,
                   semantic_threshold_rejects,
                   i0,
                   i1);
            printf("streamseed_two_storage_maintenance promotions=%zu demotions=%zu semantic_evictions=%zu in chunk [%" PRId64 ", %" PRId64 ")\n",
                   promotions, demotions, semantic_evictions, i0, i1);
            printf("streamseed_two_storage_size hot=%zu/%zu semantic=%zu/%zu in chunk [%" PRId64 ", %" PRId64 ")\n",
                   hot_size,
                   store->hot_capacity,
                   semantic_size,
                   store->semantic_slots * store->semantic_capacity,
                   i0,
                   i1);
        }
        hot_exact_hits = 0;
        hot_signature_hits = 0;
        semantic_shared_hits = 0;
        semantic_same_query_hits = 0;
        semantic_cross_query_hits = 0;
        semantic_anonymous_hits = 0;
        semantic_threshold_rejects = 0;
        promotions = 0;
        demotions = 0;
        semantic_evictions = 0;
    }
};

struct NoopSeedSource : ISeedSource {
    mutable std::vector<idx_t> empty;

    bool available(idx_t query_id, idx_t slot_key) const override {
        return false;
    }

    const std::vector<idx_t>& get(
            idx_t query_id,
            idx_t slot_key,
            const float* query,
            idx_t dim,
            SeedLookupMetadata* metadata) const override {
        if (metadata) {
            *metadata = SeedLookupMetadata{};
        }
        return empty;
    }

    void writeback(const SeedWritebackRecord& record) override {}

    void on_batch_end(bool verbose, int64_t i0, int64_t i1) override {}
};

struct StreamSeedCoreStrategy : IHintStrategy {
        StreamSeedCoreStrategy(
                        int hint_hops,
                        int hint_max_candidates,
                        float hint_gate,
                        float hint_qual_gate,
                        float hint_cons_gate,
                        int hint_boundary_gap_profile)
            : hint_hops(hint_hops),
              hint_max_candidates(hint_max_candidates),
                            hint_gate(hint_gate),
                            hint_qual_gate(hint_qual_gate),
                            hint_cons_gate(hint_cons_gate),
                            hint_boundary_gap_profile(hint_boundary_gap_profile != 0) {}

    HintSearchResult apply(const HintSearchContext& ctx) const override {
        HintSearchResult result;
        if (ctx.cache_ids.empty()) {
            return result;
        }

        std::vector<storage_idx_t> frontier;
        std::vector<storage_idx_t> candidates;
        frontier.reserve(ctx.cache_ids.size());
        candidates.reserve(ctx.cache_ids.size() * 8);

        for (idx_t cached_id : ctx.cache_ids) {
            if (cached_id < 0) {
                continue;
            }
            storage_idx_t sid = static_cast<storage_idx_t>(cached_id);
            if (!ctx.vt.get(sid)) {
                ctx.vt.set(sid);
                result.touched = true;
                frontier.push_back(sid);
                candidates.push_back(sid);
                if (hint_max_candidates > 0 &&
                    candidates.size() >=
                            static_cast<size_t>(hint_max_candidates)) {
                    break;
                }
            }
        }

        if (!frontier.empty()) {
            std::vector<storage_idx_t> next_frontier;
            int effective_hops = std::max(0, hint_hops);
            if (!ctx.high_confidence_seed && !ctx.same_query_seed) {
                // Cross-query secondary seeds use one extra hop for broader
                // candidate recovery. A signature-selected seed from the same
                // stable query uses the configured hop count directly.
                effective_hops += 1;
            }
            for (int level = 0; level < effective_hops; ++level) {
                next_frontier.clear();
                for (storage_idx_t u : frontier) {
                    size_t begin = 0, end = 0;
                    ctx.hnsw.neighbor_range(u, 0, &begin, &end);
                    for (size_t j = begin; j < end; j++) {
                        storage_idx_t v = ctx.hnsw.neighbors[j];
                        if (v < 0) {
                            break;
                        }
                        if (!ctx.vt.get(v)) {
                            ctx.vt.set(v);
                            result.touched = true;
                            next_frontier.push_back(v);
                            candidates.push_back(v);
                            if (hint_max_candidates > 0 &&
                                candidates.size() >= static_cast<size_t>(
                                                         hint_max_candidates)) {
                                break;
                            }
                        }
                    }
                    if (hint_max_candidates > 0 &&
                        candidates.size() >=
                                static_cast<size_t>(hint_max_candidates)) {
                        break;
                    }
                }
                if (next_frontier.empty()) {
                    break;
                }
                frontier.swap(next_frontier);
                if (hint_max_candidates > 0 &&
                    candidates.size() >=
                            static_cast<size_t>(hint_max_candidates)) {
                    break;
                }
            }
        }

        if (candidates.empty()) {
            return result;
        }

        std::vector<std::pair<float, storage_idx_t>> scored;
        scored.reserve(candidates.size());

        size_t j = 0;
        for (; j + 4 <= candidates.size(); j += 4) {
            float dis0 = 0.0f, dis1 = 0.0f, dis2 = 0.0f, dis3 = 0.0f;
            ctx.dis.distances_batch_4(
                    candidates[j],
                    candidates[j + 1],
                    candidates[j + 2],
                    candidates[j + 3],
                    dis0,
                    dis1,
                    dis2,
                    dis3);
            scored.emplace_back(dis0, candidates[j]);
            scored.emplace_back(dis1, candidates[j + 1]);
            scored.emplace_back(dis2, candidates[j + 2]);
            scored.emplace_back(dis3, candidates[j + 3]);
        }
        for (; j < candidates.size(); ++j) {
            scored.emplace_back(ctx.dis(candidates[j]), candidates[j]);
        }

        const size_t requested_topn =
                ctx.k > 0 ? static_cast<size_t>(ctx.k) : 0;
        const bool observes_boundary_gap =
                !ctx.high_confidence_seed &&
                (hint_qual_gate >= 0.0f || hint_boundary_gap_profile);
        const size_t boundary_extra =
                observes_boundary_gap && requested_topn < scored.size() ? 1 : 0;
        const size_t ranked_count = std::min(
                scored.size(), requested_topn + boundary_extra);
        const size_t topn = std::min(requested_topn, scored.size());
        std::partial_sort(
                scored.begin(),
                scored.begin() + ranked_count,
                scored.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        if (observes_boundary_gap && requested_topn > 0 &&
            scored.size() > requested_topn) {
            const float eps = 1e-6f;
            const float d_k = scored[requested_topn - 1].first;
            const float d_k_plus_1 = scored[requested_topn].first;
            const float raw_gap = d_k_plus_1 - d_k;
            const float normalized_gap = raw_gap / (std::fabs(d_k) + eps);
            if (std::isfinite(static_cast<double>(raw_gap)) &&
                std::isfinite(static_cast<double>(normalized_gap))) {
                result.boundary_gap_available = true;
                result.boundary_raw_gap = raw_gap;
                result.boundary_normalized_gap = normalized_gap;
            }
        }

        if (topn > 0 && hint_gate >= 0.0f && scored[0].first > hint_gate) {
            return result;
        }

        if (!ctx.high_confidence_seed && hint_qual_gate >= 0.0f) {
            // Boundary-gap validation is only for secondary-selected seeds.
            // A valid gap requires both the k-th and (k+1)-th candidates.
            if (!result.boundary_gap_available) {
                result.rejected_by_qual = true;
                return result;
            }
            if (result.boundary_normalized_gap < hint_qual_gate) {
                result.rejected_by_qual = true;
                return result;
            }
        }

        if (hint_cons_gate >= 0.0f) {
            float cons = 0.0f;
            if (ctx.k > 0) {
                std::unordered_set<idx_t> seed_set;
                seed_set.reserve(ctx.cache_ids.size());
                for (idx_t sid : ctx.cache_ids) {
                    if (sid >= 0) {
                        seed_set.insert(sid);
                    }
                }
                size_t overlap = 0;
                for (size_t t = 0; t < topn; ++t) {
                    if (seed_set.find(scored[t].second) != seed_set.end()) {
                        overlap += 1;
                    }
                }
                cons = static_cast<float>(overlap) / static_cast<float>(ctx.k);
            }
            if (cons < hint_cons_gate) {
                result.rejected_by_cons = true;
                return result;
            }
        }

        for (idx_t j2 = 0; j2 < ctx.k; j2++) {
            if (static_cast<size_t>(j2) < topn) {
                ctx.simi[j2] = scored[j2].first;
                ctx.idxi[j2] = scored[j2].second;
            } else {
                ctx.simi[j2] = std::numeric_limits<float>::infinity();
                ctx.idxi[j2] = -1;
            }
        }

        result.used = true;
        return result;
    }

   private:
    int hint_hops;
    int hint_max_candidates;
    float hint_gate;
    float hint_qual_gate;
    float hint_cons_gate;
    bool hint_boundary_gap_profile;
};

} // namespace

OptimizationConfig resolve_optimization_config(
        const HNSWIncremental& hnsw,
        const SearchParametersHNSWIncremental* params) {
    OptimizationConfig config;
    config.ef_search = hnsw.efSearch;
    if (!params) {
        return config;
    }

    config.ef_search = params->efSearch;
    config.streamseed_mode = params->streamseed_mode;
    config.hint_level1_only = params->hint_level1_only;
    config.hint_adaptive_gate_mode = params->hint_adaptive_gate_mode;
    config.hint_hops = params->hint_hops;
    config.hint_max_candidates = params->hint_max_candidates;
    config.hint_gate = params->hint_gate;
    config.hint_qual_gate = params->hint_qual_gate;
    config.hint_cons_gate = params->hint_cons_gate;
    config.hint_gate_m_quantile = params->hint_gate_m_quantile;
    config.hint_gate_o_quantile = params->hint_gate_o_quantile;
    config.hint_gate_min_samples = params->hint_gate_min_samples;
    config.hint_table_slots = params->hint_table_slots;
    config.hint_slot_capacity = params->hint_slot_capacity;
    config.hint_hot_capacity = params->hint_hot_capacity;
    config.hint_probe_count = params->hint_probe_count;
    config.hint_retrieval_threshold = params->hint_retrieval_threshold;
    config.hint_signature_weight = params->hint_signature_weight;
    config.hint_boundary_gap_profile = params->hint_boundary_gap_profile != 0;
    config.hint_promotion_hits = params->hint_promotion_hits;
    config.hint_semantic_enabled = params->hint_semantic_enabled != 0;
    config.hint_demotion_window = params->hint_demotion_window;

    if (config.streamseed_enabled()) {
        if (config.hint_hops < 0) {
            config.hint_hops = 0;
        }
        if (config.hint_max_candidates <= 0) {
            config.hint_max_candidates = 256;
        }
        if (config.hint_table_slots <= 0) {
            config.hint_table_slots = 1024;
        }
        if (config.hint_slot_capacity <= 0) {
            config.hint_slot_capacity = 2;
        }
        if (config.hint_hot_capacity <= 0) {
            config.hint_hot_capacity = 512;
        }
        if (config.hint_probe_count <= 0) {
            config.hint_probe_count = 1;
        }
        config.hint_retrieval_threshold = std::max(
                0.0f, std::min(1.0f, config.hint_retrieval_threshold));
        config.hint_signature_weight = std::max(
                0.0f, std::min(1.0f, config.hint_signature_weight));
        if (config.hint_promotion_hits <= 0) {
            config.hint_promotion_hits = 3;
        }
        if (config.hint_demotion_window <= 0) {
            config.hint_demotion_window = 20000;
        }
        if (config.hint_level1_only < 0) {
            config.hint_level1_only = 0;
        }
        if (config.hint_adaptive_gate_mode < ADAPTIVE_GATE_OFF ||
            config.hint_adaptive_gate_mode > ADAPTIVE_GATE_BATCH) {
            config.hint_adaptive_gate_mode = ADAPTIVE_GATE_OFF;
        }
        config.hint_gate_m_quantile =
                std::max(0.0f, std::min(1.0f, config.hint_gate_m_quantile));
        config.hint_gate_o_quantile =
                std::max(0.0f, std::min(1.0f, config.hint_gate_o_quantile));
        if (config.hint_cons_gate > 1.0f) {
            config.hint_cons_gate = 1.0f;
        }
        if (config.hint_gate_min_samples <= 0) {
            config.hint_gate_min_samples = 128;
        }
    }

    return config;
}

std::unique_ptr<IHintStrategy> create_streamseed_strategy(
        const OptimizationConfig& config) {
    if (!config.streamseed_enabled()) {
        return nullptr;
    }
        return std::unique_ptr<IHintStrategy>(new StreamSeedCoreStrategy(
            std::max(0, config.hint_hops),
            config.hint_max_candidates,
            config.hint_gate,
            config.hint_qual_gate,
            config.hint_cons_gate,
            config.hint_boundary_gap_profile));
}

void clear_dictionary_locks(std::vector<omp_lock_t>& locks) {
    for (size_t i = 0; i < locks.size(); ++i) {
        omp_destroy_lock(&locks[i]);
    }
    locks.clear();
}

void prepare_dictionary_if_needed(
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
        std::vector<std::vector<idx_t>>& warm_seed_dictionary_owner_query,
        std::vector<std::vector<uint64_t>>& warm_seed_dictionary_owner_signature,
        idx_t& warm_seed_dictionary_k,
        std::vector<std::vector<float>>& warm_seed_dictionary_score,
        std::vector<std::vector<uint64_t>>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
        const OptimizationConfig& config,
        idx_t k) {
    if (!config.use_dictionary()) {
        return;
    }
    if (warm_seed_dictionary_k != k) {
        warm_seed_dictionary_k = k;
        warm_seed_dictionary.clear();
        warm_seed_dictionary_owner_query.clear();
        warm_seed_dictionary_owner_signature.clear();
        warm_seed_dictionary_score.clear();
        warm_seed_dictionary_age.clear();
    }

    const size_t slots = static_cast<size_t>(config.hint_table_slots);
    if (warm_seed_dictionary.size() != slots) {
        clear_dictionary_locks(warm_seed_dictionary_locks);
        warm_seed_dictionary.assign(slots, {});
        warm_seed_dictionary_owner_query.assign(slots, {});
        warm_seed_dictionary_owner_signature.assign(slots, {});
        warm_seed_dictionary_score.assign(slots, {});
        warm_seed_dictionary_age.assign(slots, {});
        warm_seed_dictionary_locks.resize(slots);
        for (size_t i = 0; i < slots; ++i) {
            omp_init_lock(&warm_seed_dictionary_locks[i]);
        }
    }
    printf("Prepared seed dictionary with %zu slots (cap=%d) for ef_search=%d\n",
           warm_seed_dictionary.size(),
           config.hint_slot_capacity,
           config.ef_search);
}

void prepare_two_tier_store_if_needed(
        std::shared_ptr<TwoTierSeedStore>& store,
        const OptimizationConfig& config,
        idx_t k) {
    if (!config.use_two_storage()) {
        return;
    }
    if (!store) {
        store = std::make_shared<TwoTierSeedStore>();
    }
    const size_t hot_capacity =
            static_cast<size_t>(std::max(1, config.hint_hot_capacity));
    const size_t semantic_slots =
            static_cast<size_t>(std::max(1, config.hint_table_slots));
    const size_t semantic_capacity =
            static_cast<size_t>(std::max(1, config.hint_slot_capacity));
    omp_set_lock(&store->lock);
    if (store->record_k != k || store->hot_capacity != hot_capacity ||
        store->semantic_slots != semantic_slots ||
        store->semantic_capacity != semantic_capacity) {
        store->reset(k, hot_capacity, semantic_slots, semantic_capacity);
        printf("Prepared two-tier seed store hot=%zu semantic=%zux%zu for ef_search=%d\n",
               hot_capacity, semantic_slots, semantic_capacity, config.ef_search);
    }
    omp_unset_lock(&store->lock);
}

std::unique_ptr<ISeedSource> create_seed_source(
        const OptimizationConfig& config,
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
    std::vector<std::vector<idx_t>>& warm_seed_dictionary_owner_query,
    std::vector<std::vector<uint64_t>>& warm_seed_dictionary_owner_signature,
    std::vector<std::vector<float>>& warm_seed_dictionary_score,
    std::vector<std::vector<uint64_t>>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
    uint64_t& warm_seed_dictionary_clock,
    uint64_t current_batch_round,
    float& warm_seed_adaptive_m_gate,
    float& warm_seed_adaptive_o_gate,
    std::shared_ptr<TwoTierSeedStore>& two_tier_store) {
    if (config.use_two_storage() && two_tier_store) {
        return std::unique_ptr<ISeedSource>(new TwoTierSeedSource(
                two_tier_store,
                config.hint_probe_count,
                config.hint_retrieval_threshold,
                config.hint_signature_weight,
                config.hint_semantic_enabled,
                config.hint_promotion_hits,
                config.hint_demotion_window));
    }
    if (config.use_dictionary()) {
        return std::unique_ptr<ISeedSource>(
                new DictionarySeedSource(
                        warm_seed_dictionary,
            warm_seed_dictionary_owner_query,
            warm_seed_dictionary_owner_signature,
                        warm_seed_dictionary_score,
                        warm_seed_dictionary_age,
                        warm_seed_dictionary_locks,
            warm_seed_dictionary_clock,
            current_batch_round,
                        warm_seed_adaptive_m_gate,
                        warm_seed_adaptive_o_gate,
                        config.hint_level1_only != 0,
                    config.hint_adaptive_gate_mode,
                    config.hint_gate_m_quantile,
                    config.hint_gate_o_quantile,
                    config.hint_gate_min_samples,
            config.hint_slot_capacity));
    }
    return std::unique_ptr<ISeedSource>(new NoopSeedSource());
}

} // namespace streamseed
} // namespace faiss
