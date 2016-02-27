#define ASC_LINE_DETECTOR_IMPLEMENTATION
#define ASC_LINE_DETECTOR_SSE
#include "../asc_line_detector.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#define GDB_NO_STB_IMAGE_WRITE
#include <gdb.cpp>

unsigned char *rgb_to_gray(unsigned char *in, int w, int h)
{
    unsigned char *out = (unsigned char*)calloc(w*h, 1);
    unsigned char *pixel = in;
    for (int i = 0; i < w*h; i++)
    {
        float r = (float)pixel[0];
        float g = (float)pixel[1];
        float b = (float)pixel[2];
        float result_real = (r + r + b + g + g + g) / 6.0f;
        int result_rounded = (int)result_real;
        if (result_rounded < 0) result_rounded = 0;
        if (result_rounded > 255) result_rounded = 255;
        unsigned char result = (unsigned char)result_rounded;

        out[i] = result;
        pixel += 3;
    }
    return out;
}

int main(int argc, char **argv)
{
    int width, height, channels;
    unsigned char *input_rgb = stbi_load("data/video0040.jpg", &width, &height, &channels, 3);
    unsigned char *input_gray = rgb_to_gray(input_rgb, width, height);

    asc_Line *lines = 0;
    int lines_found = 0;
    asc_find_lines(
        input_rgb,
        input_gray,
        width,
        height,
        &lines,
        &lines_found);

    printf("Found %d lines\n", lines_found);
    for (int i = 0; i < lines_found; i++)
    {
        asc_Line line = lines[i];
        printf("%d: x0=%.2f y0=%.2f x1=%.2f y1=%.2f\n",
               i, line.x_min, line.y_min, line.x_max, line.y_max);
    }

    stbi_image_free(input_rgb);
    free(input_gray);
    return 0;
}
