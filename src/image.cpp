#include "image.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <cuda_fp16.h>

Image::Image(int num_rows, int num_cols, int num_depts, ImageType type) :
        n_rows(num_rows),
        n_cols(num_cols),
        n_depts(num_depts),
        data(0),
        it(type) {
    if(type == PPM) {
        data.resize(num_cols * num_rows *num_depts * 3);
        memset(data.data(), 0, sizeof(float) * data.size());
    } else {
        data.resize(num_cols * num_rows * num_depts);
        memset(data.data(), 0, sizeof(float) * data.size());
    }
}

Image::Image(const std::string &fname, int num_images, ImageType type) :
        n_rows(0),
        n_cols(0),
        n_depts(0),
        data(0) {
    read(fname, num_images, type);
}

Image::Image(const Image &other) :
        n_rows(other.n_rows),
        n_cols(other.n_cols),
        n_depts(other.n_depts),
        data(other.data) {}

const float &
Image::operator()(const int &i_row, const int &i_col, const int &i_dept) const {
    assert(i_row < n_rows);
    assert(i_col < n_cols);
    assert(i_dept < n_depts);
    return data[i_row * n_cols * n_depts + i_col * n_depts + i_dept];
}

float &
Image::operator()(const int &i_row, const int &i_col, const int &i_dept) {
    assert(i_row < n_rows);
    assert(i_col < n_cols);
    assert(i_dept < n_depts);
    return data[i_row * n_cols * n_depts + i_col * n_depts + i_dept];
}

void
Image::write(const std::string &fname) const {

    // run through all images
    for(int i = 0; i < n_depts; i++) {

        // open file and check
        std::fstream file;
        if(it == PPM) {
            file.open(fname + std::to_string(i) + ".ppm", std::ios::out);
        } else {
            file.open(fname + std::to_string(i) + ".pgm", std::ios::out);
        }
        if (!file.good()) {
            std::cerr << "Image::write() : Failed to open \"" << fname << "\"" << std::endl;
            return;
        }

        // write header
        if(it == PPM) {
            file << "P3\n";
        } else {
            file << "P2\n";
        }

        file << n_rows << " " << n_cols << '\n';
        file << 255 << '\n';

        // write image data
        for (int i_row = 0; i_row < n_rows; i_row++) {
            if(it == PPM) {
                for (int i_col = 0; i_col < n_cols; ++i_col) {
                    // clamp if necessary
                    for(int k = 0; k < 3; k++) {
                        file << std::max(std::min(255, static_cast<int>( std::round(data[(i_row * n_cols * n_depts + i_col * n_depts + i) * 3 + k]))),0) << " ";
                    }
                }
            } else {
                for (int i_col = 0; i_col < n_cols; ++i_col) {
                    // clamp if necessary
                    file << std::max(std::min(255, static_cast<int>(std::round(data[i_row * n_cols * n_depts + i_col * n_depts + i]))), 0) << " ";
                }
            }
            file << '\n';
        }

        // check if the file is still good after writing
        if (!file.good()) {
            std::cerr << "Image::write() : Failed to write '" << fname << "'" << std::endl;
            return;
        }
        file.close();
    }
}

void
Image::read(const std::string &fname, const int num_images, const ImageType type) {
    n_depts = num_images;
    for(int i = 0; i < num_images; i++) {
        // open file
        std::fstream file;
        if(type == PGM) {
            file.open(fname + std::to_string(i) + ".pgm", std::ios::in);
        } else {
            file.open(fname + std::to_string(i) + ".ppm", std::ios::in);
        }
        if(!file.good()) {
            std::cerr << "Image::read() : Failed to open \"" << fname << i << ".pgm" << "\"" << std::endl;
            return;
        }

        try {
            std::string line;
            std::getline(file, line);

            if("P2" == line) {
                it = PGM;
            } else if("P3" == line) {
                it = PPM;
            } else {
                std::cerr << "Image::read() : Cannot read file. Incorrect format." << std::endl;
                return;
            }

            std::string line_size;
            std::getline(file, line_size);
            std::stringstream sstr_line_size(line_size);
            sstr_line_size >> n_rows;
            sstr_line_size >> n_cols;

            std::string line_max_val;
            std::getline(file, line_max_val);
            std::stringstream sstr_max_val(line_max_val);
            int max_val = 0;
            sstr_max_val >> max_val;
            if (255 != max_val) {
                std::cerr << "Image::read() : incorrect image format :: max_val = " << max_val << std::endl;
                return;
            }

            // resize data depending on type
            if(i == 0) {
                if(it == PPM) {
                    data.resize(n_rows * n_cols * num_images * 3);
                } else {
                    data.resize(n_rows * n_cols * num_images);
                }
            }

            // read image data
            int k = 0;
            for (int i_row = 0; i_row < n_rows; i_row++) {

                std::string str_row;
                if (!std::getline(file, str_row)) {
                    if (file.eof()) {
                        std::cerr << "Image::read() : Failed to read '" << fname << "' :: "
                                  << i_row << " at k = " << k << std::endl;
                        return;
                    }
                }
                std::stringstream sstr_row(str_row);

                float val = 0.0;
                if(it == PPM) {
                    for (int i_col = 0; i_col < n_cols; ++i_col) {
                        float rgb[3];
                        sstr_row >> rgb[0];
                        sstr_row >> rgb[1];
                        sstr_row >> rgb[2];
                        data[(i_row * n_cols * n_depts + i_col * n_depts + i) * 3] = rgb[0];
                        data[(i_row * n_cols * n_depts + i_col * n_depts + i) * 3 + 1] = rgb[1];
                        data[(i_row * n_cols * n_depts + i_col * n_depts + i) * 3 + 2] = rgb[2];
                    }
                } else {
                    for (int i_col = 0; i_col < n_cols; ++i_col) {
                        sstr_row >> val;
                        data[i_row * n_cols * n_depts + i_col * n_depts + i] = val;
                    }
                }
            }

        }
        catch (const std::exception &excep) {
            std::cerr << "Image::read() : exeception occurred : " << excep.what() << std::endl;
        }
        file.close();
    }
}