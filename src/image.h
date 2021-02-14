#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <vector>
#include <string>

enum ImageType {
    PGM, PPM
};

class Image {

public:

    Image(int num_rows, int num_cols, int num_depts, ImageType type);

    Image(const std::string &fname, int num_images, ImageType type);

    Image(const Image &other);

    ~Image() {};

public:

    const float &operator()(const int &i_row, const int &i_col, const int &i_dept) const;

    float &operator()(const int &i_row, const int &i_col, const int &i_dept);

public:

    void read(const std::string &fname, const int num_images, const ImageType type);

    void write(const std::string &fname) const;

public:

    int n_rows;
    int n_cols;
    int n_depts;

public:

    std::vector<float> data;

    ImageType it;

private:

    Image();

    Image &operator=(const Image &);

};

#endif
