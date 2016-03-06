#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#ifdef USE_GDB
#define GDB_NO_STB_IMAGE_WRITE
#include <gdb.cpp>
#endif

#define ASC_LINE_DETECTOR_IMPLEMENTATION
#define ASC_LINE_DETECTOR_SSE
#include "../asc_line_detector.h"

#include <stdio.h>

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
    struct Test
    {
        const char *filename;
        int expected_count;
    };
    Test tests[] = {
        { "data/test1.png", 0 },
        { "data/test2.png", 1 },
        { "data/test3.png", 2 },
        { "data/test4.png", 3 },
        { "data/test5.png", 2 },
        { "data/test6.png", 2 },
        { "data/test7.png", 3 }
    };

    for (int test_id = 0; test_id < sizeof(tests)/sizeof(Test); test_id++)
    {
        int width, height, channels;
        unsigned char *input_rgb = stbi_load(tests[test_id].filename, &width, &height, &channels, 3);
        unsigned char *input_gray = rgb_to_gray(input_rgb, width, height);

        asc_LineDetectorOptions options;
        options.sobel_threshold            = 10;
        options.hough_sample_count         = 4096;
        options.suppression_window_t       = 0.349f;
        options.suppression_window_r       = 300.0f;
        options.peak_exit_threshold        = 0.1f;
        options.normal_error_threshold     = 20.0f;
        options.normal_error_std_threshold = 20.0f;

        const int max_lines = 16;
        asc_Line lines[max_lines] = {0};
        int lines_found = 0;
        asc_find_lines(
            input_rgb,
            input_gray,
            width,
            height,
            lines,
            &lines_found,
            max_lines,
            options);

        if (lines_found != tests[test_id].expected_count)
        {
            printf("FAILED ");
        }
        printf("Test %d: Found %d (expected %d)\n", test_id+1, lines_found, tests[test_id].expected_count);
        for (int i = 0; i < lines_found; i++)
        {
            asc_Line line = lines[i];
            printf("\t%d: x0=%.2f y0=%.2f x1=%.2f y1=%.2f\n", i, line.x_min, line.y_min, line.x_max, line.y_max);
        }
        printf("\n");

        {
            unsigned char *out = (unsigned char*)calloc(width*height, 3);
            for (int i = 0; i < width*height; i++)
            {
                out[i] = input_rgb[i];
            }
            for (int i = 0; i < lines_found; i++)
            {
                asc_Line line = lines[i];
                float x0 = line.x_min;
                float x1 = line.x_max;
                float y0 = line.y_min;
                float y1 = line.y_max;
                float dx = x1-x0;
                float dy = y1-y0;
                float len = sqrt(dx*dx+dy*dy);
                int samples = (int)(len);
                for (int j = 0; j < samples; j++)
                {
                    float t = (float)j/samples;
                    int x = (int)(x0 + t*dx+0.5f);
                    int y = (int)(y0 + t*dy+0.5f);
                    if (x >= 0 && x < width &&
                        y >= 0 && y < height)
                    {
                        out[(y*width+x)*3+0] = 255;
                        out[(y*width+x)*3+1] = 40;
                        out[(y*width+x)*3+2] = 40;
                    }
                }
            }
            char filename[256];
            sprintf(filename, "data/test%d.out.png", test_id+1);
            stbi_write_png(filename, width, height, 3, out, width*3);
            free(out);
        }

        free(input_gray);
        stbi_image_free(input_rgb);
    }

    return 0;
}
