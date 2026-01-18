#ifndef __ORDERED_MAP_HPP__
#define __ORDERED_MAP_HPP__

#include <iostream>
#include <list>
#include <string>
#include <unordered_map>

#include <initializer_list>
#include <iterator>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

template <typename Key, typename T>
class OrderedMap {
public:
    using key_type        = Key;
    using mapped_type     = T;
    using value_type      = std::pair<const Key, T>;
    using list_type       = std::list<value_type>;
    using size_type       = typename list_type::size_type;
    using difference_type = typename list_type::difference_type;
    using iterator        = typename list_type::iterator;
    using const_iterator  = typename list_type::const_iterator;

private:
    list_type data_;
    std::unordered_map<Key, iterator> index_;

public:
    // --- constructors ---
    OrderedMap() = default;

    OrderedMap(std::initializer_list<value_type> init) {
        for (const auto& kv : init)
            insert(kv);
    }

    OrderedMap(const OrderedMap&)                = default;
    OrderedMap(OrderedMap&&) noexcept            = default;
    OrderedMap& operator=(const OrderedMap&)     = default;
    OrderedMap& operator=(OrderedMap&&) noexcept = default;

    // --- element access ---
    T& at(const Key& key) {
        auto it = index_.find(key);
        if (it == index_.end())
            throw std::out_of_range("OrderedMap::at: key not found");
        return it->second->second;
    }

    const T& at(const Key& key) const {
        auto it = index_.find(key);
        if (it == index_.end())
            throw std::out_of_range("OrderedMap::at: key not found");
        return it->second->second;
    }

    T& operator[](const Key& key) {
        auto it = index_.find(key);
        if (it == index_.end()) {
            data_.emplace_back(key, T{});
            auto iter   = std::prev(data_.end());
            index_[key] = iter;
            return iter->second;
        }
        return it->second->second;
    }

    // --- iterators ---
    iterator begin() noexcept { return data_.begin(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }

    iterator end() noexcept { return data_.end(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    // --- capacity ---
    bool empty() const noexcept { return data_.empty(); }
    size_type size() const noexcept { return data_.size(); }

    // --- modifiers ---
    void clear() noexcept {
        data_.clear();
        index_.clear();
    }

    std::pair<iterator, bool> insert(const value_type& value) {
        auto it = index_.find(value.first);
        if (it != index_.end()) {
            return {it->second, false};
        }
        data_.push_back(value);
        auto iter           = std::prev(data_.end());
        index_[value.first] = iter;
        return {iter, true};
    }

    std::pair<iterator, bool> insert(value_type&& value) {
        auto it = index_.find(value.first);
        if (it != index_.end()) {
            return {it->second, false};
        }
        data_.push_back(std::move(value));
        auto iter           = std::prev(data_.end());
        index_[iter->first] = iter;
        return {iter, true};
    }

    void erase(const Key& key) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            data_.erase(it->second);
            index_.erase(it);
        }
    }

    iterator erase(iterator pos) {
        index_.erase(pos->first);
        return data_.erase(pos);
    }

    // --- lookup ---
    size_type count(const Key& key) const {
        return index_.count(key);
    }

    iterator find(const Key& key) {
        auto it = index_.find(key);
        if (it == index_.end())
            return data_.end();
        return it->second;
    }

    const_iterator find(const Key& key) const {
        auto it = index_.find(key);
        if (it == index_.end())
            return data_.end();
        return it->second;
    }

    bool contains(const Key& key) const {
        return index_.find(key) != index_.end();
    }

    // --- comparison ---
    bool operator==(const OrderedMap& other) const {
        return data_ == other.data_;
    }

    bool operator!=(const OrderedMap& other) const {
        return !(*this == other);
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        value_type value(std::forward<Args>(args)...);
        auto it = index_.find(value.first);
        if (it != index_.end()) {
            return {it->second, false};
        }
        data_.push_back(std::move(value));
        auto iter           = std::prev(data_.end());
        index_[iter->first] = iter;
        return {iter, true};
    }

    void swap(OrderedMap& other) noexcept {
        data_.swap(other.data_);
        index_.swap(other.index_);
    }
};

#endif  // __ORDERED_MAP_HPP__