#ifndef _TIMER_H
#define _TIMER_H


class Timer {

private:
    std::chrono::time_point<std::chrono::steady_clock> first;

public:

    Timer() {}
    ~Timer() {}

    void start() {
        first = std::chrono::steady_clock::now();
    }

    double stop() {
        double res = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - first).count();
        std::cerr << "Time measured: " << res << "ms." << std::endl;
        return res;
    }

};


#endif //_TIMER_H
