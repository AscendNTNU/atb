// SIMD-optimized line detection algorithm for image and video processing.
//
// Features:
//   x AVX/AVX2 optimizations
//   x SSE/SSE2 optimizations
//   x Optional no-optimizations
//   x Improved detection accuracy by processing image sequences over time
//   x Line-cluster detection and merging
//
// Relevant papers and links:
//   Title:        Randomized Hough Transform (RHT)
//   Authors:      Pekka Kultanen, Lei Xut, Erkki Oja
//   Published in: Pattern Recognition, 1990. Proceedings.,
//                 10th International Conference on (Volume:i)
//
//   Title:        Sobel Operator
//   https://en.wikipedia.org/wiki/Sobel_operator
//
// Written by: Simen Haugo (simen.haugo@ascendntnu.no)
// For:        Ascend NTNU (ascendntnu.no)
//
// How to compile:
// ------------------------------------------------------------------------
// This file contains both the header file and the implementation file.
// To compile, insert the following in A SINGLE source file in your project
//
//     #define ASC_LINE_DETECTOR_IMPLEMENTATION
//     #include "asc_line_detector.h"
//
// You may otherwise include this file as you would include a traditional
// header file. You may choose between SSE, AVX or no optimizations by
// inserting the following in the same source file as above
//
//     #define ASC_LINE_DETECTOR_AVX // Enable AVX/AVX2 level optimization
//     #define ASC_LINE_DETECTOR_SSE // Enable SSE/SSE2 level optimization
//
// Default is no optimization.
//
// Changelog
// ------------------------------------------------------------------------
//   1.00 (10. feb 2016) Ported code to single-file header library
//
// Licence
// ------------------------------------------------------------------------
// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy,
// distribute, and modify this file as you see fit.
//
// No warranty for any purpose is expressed or implied by the author (nor
// by Ascend NTNU). Report bugs and send enhancements to the author.
//

#ifndef ASC_LINE_DETECTOR_HEADER_INCLUDE
#define ASC_LINE_DETECTOR_HEADER_INCLUDE
#include <stdint.h>
#include <stdlib.h> // for calloc

struct asc_Line
{
    // Lines are parametrized in the "normal form",
    // by t (angle of the normal against x-axis) and
    // r (distance from origin to the line), such that
    // a point (x, y) is on the line if it satisfies:
    //     x cos(t) + y sin(t) = r
    float t;
    float r;

    // The line boundary points are given in image-space,
    // with the following directions:
    // x = 0:      left
    // x = width:  right
    // y = 0:      top
    // y = height: bottom
    float x_min;
    float y_min;
    float x_max;
    float y_max;

    // Estimated dominant color, in normalized values [0, 1].
    float color_r;
    float color_g;
    float color_b;
};

// TODO: Flags enabling or disabling estimation of
// color, boundaries, etc... Fisheye correction?

void asc_find_lines(
    uint8_t   *in_rgb,
    uint8_t   *in_gray,
    int32_t    in_width,
    int32_t    in_height,
    asc_Line **out_lines,
    int32_t   *out_count,
    int16_t    param_feature_threshold = 10,
    int32_t    param_max_ms_iterations = 32,
    float      param_cluster_size_t = 10.0f*3.1415f/180.0f,
    float      param_cluster_size_r = 100.0f);

// void asc_fit_grid(
//     asc_Line *in_lines,
//     int32_t   in_count,
//     rotation matrix, x, y, z);

// TODO: I would like to implement a function exploiting
// temporal correspondence in line detections in a video
// stream. It would look something like this:
//
//   asc_find_lines_video(uint08_t **frames, int32_t frame_count, ...)
//
// You would call the function with a pointer to the
// first frame in a window with 'frame_count' frames
// leading up to the most recent,
//
//   frames->[frame8, frame9, frame10, frame11]
//                                        ^ most recent
// I.e. the resulting line detection would be the most
// likely set of lines that exist in frame11, given the
// last 3 frames that came before it.

// asc_lines_push_frame(...)?

#endif

#ifdef ASC_LINE_DETECTOR_IMPLEMENTATION
#include <math.h>
#define ASCI_PI 3.1415926f
#define ASCI_FLT_MAX 3.402823466e+38F
#define s32 int32_t
#define s16 int16_t
#define s08 int8_t
#define u32 uint32_t
#define u16 uint16_t
#define u08 uint8_t
#define r32 float

struct asci_Feature
{
    s32 x;
    s32 y;
    s16 gx;
    s16 gy;
    s16 gg;
};

void asci_sobel(
    u08 *in_rgb,
    u08 *in_gray,
    s32  in_width,
    s32  in_height,
    s16  threshold,
    asci_Feature *out_features,
    s32 *out_feature_count)
{
    s32 feature_count = 0;
    for (s32 y = 1; y < in_height-1; y++)
    {
        for (s32 x = 1; x < in_width-1; x++)
        {
            s16 i00 = (s16)in_gray[(y-1)*in_width+x-1] >> 2;
            s16 i01 = (s16)in_gray[(y-1)*in_width+x] >> 2;
            s16 i02 = (s16)in_gray[(y-1)*in_width+x+1] >> 2;
            s16 i20 = (s16)in_gray[(y+1)*in_width+x-1] >> 2;
            s16 i21 = (s16)in_gray[(y+1)*in_width+x] >> 2;
            s16 i22 = (s16)in_gray[(y+1)*in_width+x+1] >> 2;
            s16 i10 = (s16)in_gray[y*in_width+x-1] >> 2;
            s16 i12 = (s16)in_gray[y*in_width+x+1] >> 2;

            s16 gx = i02-i00+i12+i12-i10-i10+i22-i20;
            s16 gy = i20-i00+i21+i21-i01-i01+i22-i02;
            s16 gg = abs(gx) + abs(gy);
            if (gg > threshold)
            {
                asci_Feature feature = {0};
                feature.x = x;
                feature.y = y;
                feature.gx = gx;
                feature.gy = gy;
                feature.gg = gg;
                out_features[feature_count++] = feature;
            }
        }
    }
    *out_feature_count = feature_count;
}

// This algorithm has a maximal period of 2^128 − 1.
// https://en.wikipedia.org/wiki/Xorshift
u32 asci_xor128()
{
    static u32 x = 123456789;
    static u32 y = 362436069;
    static u32 z = 521288629;
    static u32 w = 88675123;
    u32 t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w = w ^ (w >> 19) ^ t ^ (t >> 8);
    return w;
}

s32 asci_round_positive(r32 x)
{
    return (s32)(x + 0.5f);
}

s32 asci_clamp_s32(s32 x, s32 low, s32 high)
{
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

struct asci_Vote
{
    s32 x1;
    s32 y1;
    s32 x2;
    s32 y2;
    s16 gx1;
    s16 gy1;
    s16 gx2;
    s16 gy2;
    r32 r;
    r32 t;
};

void asci_hough(
    asci_Feature *in_features,
    s32 in_feature_count,
    s32 sample_count,
    asci_Vote *out_votes,
    s32 *out_count,
    r32 *out_t_min,
    r32 *out_t_max,
    r32 *out_r_min,
    r32 *out_r_max)
{
    r32 t_min = ASCI_FLT_MAX;
    r32 t_max = -ASCI_FLT_MAX;
    r32 r_min = ASCI_FLT_MAX;
    r32 r_max = -ASCI_FLT_MAX;
    s32 count = 0;

    for (s32 sample = 0; sample < sample_count; sample++)
    {
        s32 sample_i1 = asci_xor128() % (in_feature_count);
        s32 sample_i2 = asci_xor128() % (in_feature_count);
        asci_Feature f1 = in_features[sample_i1];
        asci_Feature f2 = in_features[sample_i2];

        // Note(Simen): I reject a vote if the gradients of
        // the drawn samples differ more than a threshold.
        if (f1.gg > 0 && f2.gg > 0)
        {
            r32 dot = f1.gx*f2.gx + f1.gy*f2.gy;
            if (abs(dot) < 0.5f*f1.gg*f2.gg)
            {
                continue;
            }
        }
        r32 dx = (r32)(f2.x - f1.x);
        r32 dy = (r32)(f2.y - f1.y);
        r32 t = atan((r32)dy / (r32)dx) + ASCI_PI / 2.0f;
        r32 c = cos(t);
        r32 s = sin(t);

        // Note(Simen): I also reject a vote if the normal of
        // the line drawn between the samples differs from the
        // gradients of the samples by more than a threshold.
        if (f1.gg > 0 && f2.gg > 0)
        {
            r32 dot1 = (f1.gx*c+f1.gy*s) / f1.gg;
            r32 dot2 = (f2.gx*c+f2.gy*s) / f2.gg;
            r32 adot = 0.5f*(abs(dot1) + abs(dot2));
            if (adot < 0.5f)
            {
                continue;
            }

        }

        r32 r = f1.x*c + f1.y*s;
        if (r < r_min) r_min = r;
        if (r > r_max) r_max = r;
        if (t < t_min) t_min = t;
        if (t > t_max) t_max = t;

        asci_Vote vote = {0};
        vote.t = t;
        vote.r = r;
        vote.x1 = f1.x;
        vote.y1 = f1.y;
        vote.x2 = f2.x;
        vote.y2 = f2.y;
        vote.gx1 = f1.gx;
        vote.gy1 = f1.gy;
        vote.gx2 = f2.gx;
        vote.gy2 = f2.gy;
        out_votes[count++] = vote;
    }

    *out_t_min = t_min;
    *out_t_max = t_max;
    *out_r_min = r_min;
    *out_r_max = r_max;
    *out_count = count;
}

void asc_find_lines(
    u08 *in_rgb,
    u08 *in_gray,
    s32 in_width,
    s32 in_height,
    asc_Line **out_lines,
    s32 *out_count,
    s16 sobel_threshold,
    s32 cluster_iterations,
    r32 cluster_size_t,
    r32 cluster_size_r)
{
    // TODO: This can be a static array. Let the user define max
    // dimensions for the image, and provide default sizes.
    asci_Feature *features = (asci_Feature*)calloc(in_width*in_height, sizeof(asci_Feature));
    s32 feature_count = 0;

    asci_sobel(
        in_rgb,
        in_gray,
        in_width,
        in_height,
        sobel_threshold,
        features,
        &feature_count);

    GDB("sobel features",
    {
        Ortho(0.0f, in_width, in_height, 0.0f);
        glPointSize(2.0f);
        Clear(0.0f, 0.0f, 0.0f, 1.0f);
        glBegin(GL_POINTS);
        {
            for (int i = 0; i < feature_count; i++)
            {
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                glVertex2f(features[i].x, features[i].y);
            }
        }
        glEnd();
    });

    const s32 sample_count = 4096*4;
    static asci_Vote votes[sample_count];
    s32 vote_count = 0;
    r32 t_min = 0.0f;
    r32 t_max = 0.0f;
    r32 r_min = 0.0f;
    r32 r_max = 0.0f;
    asci_hough(
        features,
        feature_count,
        sample_count,
        votes,
        &vote_count,
        &t_min, &t_max,
        &r_min, &r_max);

    struct HoughCell
    {
        r32 avg_r;
        r32 avg_t;
        r32 weight;
    };

    const s32 bins_t = 32;
    const s32 bins_r = 32;
    static HoughCell histogram[bins_r*bins_t];
    for (s32 i = 0; i < bins_r*bins_t; i++)
    {
        histogram[i].avg_r = 0.0f;
        histogram[i].avg_t = 0.0f;
        histogram[i].weight = 0.0f;
    }

    r32 histogram_max_weight = 0.0f;
    for (s32 vote_index = 0; vote_index < vote_count; vote_index++)
    {
        asci_Vote vote = votes[vote_index];
        r32 t = vote.t;
        r32 r = vote.r;
        s32 ti = clamp_s32(bins_t*(t-t_min)/(t_max-t_min), 0, bins_t-1);
        s32 ri = clamp_s32(bins_r*(r-r_min)/(r_max-r_min), 0, bins_r-1);
        HoughCell *cell = &histogram[ti + ri*bins_t];
        cell->avg_r += r;
        cell->avg_t += t;
        cell->weight += 1.0f;
        if (cell->weight > histogram_max_weight)
            histogram_max_weight = cell->weight;
    }

    // TODO: Perform running average instead
    for (s32 i = 0; i < bins_r*bins_t; i++)
    {
        if (histogram[i].weight > 0.0f)
        {
            histogram[i].avg_r /= histogram[i].weight;
            histogram[i].avg_t /= histogram[i].weight;
        }
    }

    // Peak extraction
    const s32 lines_to_find = 16;
    r32 suppression_window_t = 20.0f * ASCI_PI / 180.0f;
    r32 suppression_window_r = 300.0f;
    r32 selection_window_t = 20.0f * ASCI_PI / 180.0f;
    r32 selection_window_r = 300.0f;
    r32 bin_size_t = (t_max-t_min) / bins_t;
    r32 bin_size_r = (r_max-r_min) / bins_r;
    s32 suppression_window_ti = round_r32_plus(suppression_window_t / bin_size_t);
    s32 suppression_window_ri = round_r32_plus(suppression_window_r / bin_size_r);
    s32 selection_window_ti = round_r32_plus(selection_window_t / bin_size_t);
    s32 selection_window_ri = round_r32_plus(selection_window_r / bin_size_r);
    if (suppression_window_ti % 2 != 0) suppression_window_ti++;
    if (suppression_window_ri % 2 != 0) suppression_window_ri++;
    if (selection_window_ti % 2 != 0) selection_window_ti++;
    if (selection_window_ti % 2 != 0) selection_window_ti++;
    for (s32 iteration = 0; iteration < lines_to_find; iteration++)
    {
        // Extract max
        s32 peak_index = 0;
        r32 peak_weight = 0;
        for (s32 i = 0; i < bins_t*bins_r; i++)
        {
            if (histogram[i].weight > peak_weight)
            {
                peak_weight = histogram[i].weight;
                peak_index = i;
            }
        }

        if (peak_weight < 0.025f*histogram_max_weight)
        {
            // TODO: Early exit
            printf("Early exit at iteration %d; peak weight (%.2f) was less than threshold (%.2f)\n",
                   iteration, peak_weight, 0.025f*histogram_max_weight);
            break;
        }

        s32 peak_ti = peak_index % bins_t;
        s32 peak_ri = peak_index / bins_t;
        r32 peak_t = histogram[peak_index].avg_t;
        r32 peak_r = histogram[peak_index].avg_r;

        GDB("hough histogram",
        {
            s32 mouse_ti = round_r32_plus((0.5f+0.5f*input.mouse.x)*bins_t);
            s32 mouse_ri = round_r32_plus((0.5f-0.5f*input.mouse.y)*bins_r);

            Ortho(t_min, t_max, r_min, r_max);
            BlendMode();
            Clear(0.0f, 0.0f, 0.0f, 1.0f);
            glPointSize(6.0f);
            glBegin(GL_POINTS);
            {
                for (s32 ri = 0; ri < bins_r; ri++)
                for (s32 ti = 0; ti < bins_t; ti++)
                {
                    r32 r = histogram[ti + ri*bins_t].avg_r;
                    r32 t = histogram[ti + ri*bins_t].avg_t;
                    r32 weight = histogram[ti + ri*bins_t].weight;

                    if (mouse_ti == ti && mouse_ri == ri)
                    {
                        glColor4f(0.4f, 1.0f, 0.4f, 1.0f);
                        SetTooltip("%.2f %.2f %.2f", t, r, weight);
                    }
                    else
                    {
                        ColorRamp(weight / (0.2f*histogram_max_weight));
                    }
                    glVertex2f(t, r);
                }
            }
            glEnd();

            glPointSize(14.0f);
            glBegin(GL_POINTS);
            glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
            glVertex2f(peak_t, peak_r);
            glEnd();
        });

        // Compute two base points from which
        // line distance will be measured relative to.
        float x0, y0, x1, y1;
        float normal_x = cos(peak_t);
        float normal_y = sin(peak_t);
        float tangent_x = normal_y;
        float tangent_y = -normal_x;
        {
            if (abs(normal_y) > abs(normal_x))
            {
                x0 = 0.0f;
                x1 = in_width;
                y0 = (peak_r-x0*normal_x)/normal_y;
                y1 = (peak_r-x1*normal_x)/normal_y;
            }
            else
            {
                y0 = 0.0f;
                y1 = in_height;
                x0 = (peak_r-y0*normal_y)/normal_x;
                x1 = (peak_r-y1*normal_y)/normal_x;
            }
        }

        GDB("line estimate", {
            static GLuint texture = 0;
            if (!texture)
                texture = MakeTexture2D(in_rgb, in_width, in_height, GL_RGB);
            BlendMode(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            Clear(0.0f, 0.0f, 0.0f, 1.0f);
            Ortho(-1.0f, +1.0f, +1.0f, -1.0f);
            DrawTexture(texture, 0.5f, 0.5f, 0.5f);
            Ortho(0.0f, in_width, in_height, 0.0f);
            glLineWidth(5.0f);
            glBegin(GL_LINES);
            glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
            glEnd();

            s32 ti0 = clamp_s32(peak_ti - suppression_window_ti/2, 0, bins_t-1);
            s32 ti1 = clamp_s32(peak_ti + suppression_window_ti/2, 0, bins_t-1);
            s32 ri0 = clamp_s32(peak_ri - suppression_window_ri/2, 0, bins_r-1);
            s32 ri1 = clamp_s32(peak_ri + suppression_window_ri/2, 0, bins_r-1);
            r32 t0 = t_min + (t_max-t_min)*ti0/bins_t;
            r32 r0 = r_min + (r_max-r_min)*ri0/bins_r;
            r32 t1 = t_min + (t_max-t_min)*ti1/bins_t;
            r32 r1 = r_min + (r_max-r_min)*ri1/bins_r;
            Ortho(0.0f, in_width, in_height, 0.0f);
            BlendMode(GL_ONE, GL_ONE);
            glPointSize(4.0f);
            glBegin(GL_POINTS);
            glColor4f(0.2f*0.3f, 0.2f*0.5f, 0.2f*0.8f, 1.0f);
            for (s32 i = 0; i < vote_count; i++)
            {
                asci_Vote vote = votes[i];
                if (vote.t >= t0 && vote.t <= t1 &&
                    vote.r >= r0 && vote.r <= r1)
                {
                    glVertex2f(vote.x1, vote.y1);
                    glVertex2f(vote.x2, vote.y2);
                }
            }
            glEnd();
        });

        // suppress neighborhood
        #if 1
        // TODO: "Unroll" underflowing or oveflowing segments
        s32 ti0 = peak_ti - suppression_window_ti/2;
        s32 ti1 = peak_ti + suppression_window_ti/2;
        s32 ri0 = peak_ri - suppression_window_ri/2;
        s32 ri1 = peak_ri + suppression_window_ri/2;
        for (s32 ti = ti0; ti <= ti1; ti++)
        for (s32 ri = ri0; ri <= ri1; ri++)
        {
            s32 write_t = 0;
            s32 write_r = 0;
            if (ti < 0)
            {
                // Topographically, t=0 and t=pi are identified
                // by gluing (0, r_min)-(0, r_max) with
                // (pi, r_max)-(pi, r_min)
                write_t = clamp_s32(ti+bins_t, 0, bins_t-1);
                write_r = clamp_s32(bins_r-ri, 0, bins_r-1);
            }
            else if (ti >= bins_t)
            {
                write_t = clamp_s32(ti-bins_t, 0, bins_t-1);
                write_r = clamp_s32(bins_r-1-ri, 0, bins_r-1);
            }
            else
            {
                write_t = clamp_s32(ti, 0, bins_t-1);
                write_r = clamp_s32(ri, 0, bins_r-1);
            }
            histogram[write_t + write_r*bins_t].weight = 0.0f;
        }
        #else
        s32 ti0 = clamp_s32(peak_ti - suppression_window_ti/2, 0, bins_t-1);
        s32 ti1 = clamp_s32(peak_ti + suppression_window_ti/2, 0, bins_t-1);
        s32 ri0 = clamp_s32(peak_ri - suppression_window_ri/2, 0, bins_r-1);
        s32 ri1 = clamp_s32(peak_ri + suppression_window_ri/2, 0, bins_r-1);
        for (s32 ti = ti0; ti <= ti1; ti++)
        for (s32 ri = ri0; ri <= ri1; ri++)
        {
            histogram[ti + ri*bins_t].weight = 0.0f;
        }
        #endif
    }

    free(features);
}

#undef s32
#undef s16
#undef s08
#undef u32
#undef u16
#undef u08
#undef r32
#endif
