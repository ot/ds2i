#pragma once

#include <chrono>
#include <iostream>
#include <unordered_map>

namespace ds2i {

class progress {
   public:
    progress(const std::string &name, size_t goal) : m_name(name) {
        if (goal == 0) {
            throw std::runtime_error("goal must be positive");
        }
        m_goal = goal;
    }
    ~progress() { std::cerr << "\n"; }

    void update(size_t inc) {
        std::unique_lock<std::mutex> lock(m_mut);
        m_count += inc;
    }

    void update_and_print(size_t inc) {
        std::unique_lock<std::mutex> lock(m_mut);
        m_count += inc;
        size_t progress = (100 * m_count) / m_goal;
        std::chrono::seconds elapsed  = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - m_start);
        std::cerr << '\r' << m_name << ": " << progress << "% [" << elapsed.count() << " s]"
                  << std::flush;
    }

   private:
    std::string m_name;
    size_t m_count = 0;
    size_t m_goal  = 0;

    std::chrono::time_point<std::chrono::steady_clock> m_start = std::chrono::steady_clock::now();

    std::mutex m_mut;
};

}
