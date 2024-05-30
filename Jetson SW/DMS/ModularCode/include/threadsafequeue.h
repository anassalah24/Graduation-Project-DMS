#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    // Add an item to the queue
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        dataQueue.push(item);
        lock.unlock();
        condVar.notify_one();
    }

    // Try to pop an item from the queue. Returns false if the queue is empty
    bool tryPop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (dataQueue.empty()) {
            return false;
        }
        item = dataQueue.front();
        dataQueue.pop();
        return true;
    }

    // Wait and pop an item from the queue
    void waitAndPop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        condVar.wait(lock, [this]{ return !dataQueue.empty(); });
        item = dataQueue.front();
        dataQueue.pop();
    }

    // Check if the queue is empty
    bool empty() const {
        std::unique_lock<std::mutex> lock(mtx);
        return dataQueue.empty();
    }

    // Clear all items from the queue
    void clear() {
        std::unique_lock<std::mutex> lock(mtx);
        while (!dataQueue.empty()) {
            dataQueue.pop();
        }
        // Optionally, notify all waiting threads that the state has changed
        condVar.notify_all();
    }


private:
    mutable std::mutex mtx;
    std::queue<T> dataQueue;
    std::condition_variable condVar;
};

