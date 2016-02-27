#define ASC_LINE_DETECTOR_IMPLEMENTATION
#define ASC_LINE_DETECTOR_SSE
#include "asc_line_detector.h"

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

#include <stdint.h>
#include <stdio.h>
typedef float    r32;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u08;
typedef int32_t  s32;
typedef int16_t  s16;
typedef int8_t   s08;

u08 *rgb_to_gray(u08 *in, s32 w, s32 h)
{
    u08 *out = (u08*)calloc(w*h, 1);
    u08 *pixel = in;
    for (s32 i = 0; i < w*h; i++)
    {
        r32 r = (r32)pixel[0];
        r32 g = (r32)pixel[1];
        r32 b = (r32)pixel[2];
        r32 result_real = (r + r + b + g + g + g) / 6.0f;
        s32 result_rounded = (s32)result_real;
        if (result_rounded < 0) result_rounded = 0;
        if (result_rounded > 255) result_rounded = 255;
        u08 result = (u08)result_rounded;

        out[i] = result;
        pixel += 3;
    }
    return out;
}

int main()
{
    s32 width, height, channels;
    u08 *input_rgb = stbi_load("C:/Downloads/temp/video2/video0036.jpg", &width, &height, &channels, 3);
    u08 *input_gray = rgb_to_gray(input_rgb, width, height);

    asc_Line *lines = 0;
    s32 lines_found = 0;
    asc_find_lines(
        input_rgb,
        input_gray,
        width,
        height,
        &lines,
        &lines_found);

    printf("Found %d lines\n", lines_found);
    for (s32 i = 0; i < lines_found; i++)
    {
        asc_Line line = lines[i];
        printf("%d: x0=%.2f y0=%.2f x1=%.2f y1=%.2f\n",
               i, line.x_min, line.y_min, line.x_max, line.y_max);
    }
}
