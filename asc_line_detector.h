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
//     int32_t   in_count);

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

#endif

#ifdef ASC_LINE_DETECTOR_IMPLEMENTATION
#include <math.h>
#define ASCI_PI 3.1415926f
#define ASCI_FLT_MAX 3.402823466e+38F

struct asci_Feature
{
    int32_t x;
    int32_t y;
    int16_t gx;
    int16_t gy;
};

void asci_sobel(
    uint8_t *in_rgb,
    uint8_t *in_gray,
    int32_t  in_width,
    int32_t  in_height,
    int16_t  threshold,
    asci_Feature *out_features,
    int32_t *out_feature_count)
{
    int32_t feature_count = 0;
    for (int32_t y = 1; y < in_height-1; y++)
    {
        for (int32_t x = 1; x < in_width-1; x++)
        {
            int16_t i00 = (int16_t)in_gray[(y-1)*in_width+x-1] >> 2;
            int16_t i01 = (int16_t)in_gray[(y-1)*in_width+x] >> 2;
            int16_t i02 = (int16_t)in_gray[(y-1)*in_width+x+1] >> 2;
            int16_t i20 = (int16_t)in_gray[(y+1)*in_width+x-1] >> 2;
            int16_t i21 = (int16_t)in_gray[(y+1)*in_width+x] >> 2;
            int16_t i22 = (int16_t)in_gray[(y+1)*in_width+x+1] >> 2;
            int16_t i10 = (int16_t)in_gray[y*in_width+x-1] >> 2;
            int16_t i12 = (int16_t)in_gray[y*in_width+x+1] >> 2;

            int16_t gx = i02-i00+i12+i12-i10-i10+i22-i20;
            int16_t gy = i20-i00+i21+i21-i01-i01+i22-i02;
            int16_t gg = abs(gx) + abs(gy);
            if (gg > threshold)
            {
                asci_Feature feature = {0};
                feature.x = x;
                feature.y = y;
                feature.gx = gx;
                feature.gy = gy;
                out_features[feature_count++] = feature;
            }
        }
    }
    *out_feature_count = feature_count;
}

// This algorithm has a maximal period of 2^128 − 1.
// https://en.wikipedia.org/wiki/Xorshift
uint32_t asci_xor128()
{
    static uint32_t x = 123456789;
    static uint32_t y = 362436069;
    static uint32_t z = 521288629;
    static uint32_t w = 88675123;
    uint32_t t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w = w ^ (w >> 19) ^ t ^ (t >> 8);
    return w;
}

int32_t asci_round_positive(float x)
{
    return (int32_t)(x + 0.5f);
}

int32_t asci_clamp_s32(int32_t x, int32_t low, int32_t high)
{
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

struct asci_Vote
{
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    float r;
    float t;
};

void asci_hough(
    asci_Feature *in_features,
    int32_t       in_feature_count,
    int32_t       sample_count,
    asci_Vote    *out_votes,
    int32_t      *out_count,
    float        *out_t_min,
    float        *out_t_max,
    float        *out_r_min,
    float        *out_r_max)
{
    int32_t count = sample_count;
    if (in_feature_count < sample_count)
        count = in_feature_count;

    float t_min = ASCI_FLT_MAX;
    float t_max = -ASCI_FLT_MAX;
    float r_min = ASCI_FLT_MAX;
    float r_max = -ASCI_FLT_MAX;

    for (int32_t sample = 0; sample < count; sample++)
    {
        int32_t sample_i1 = asci_xor128() % (in_feature_count);
        int32_t sample_i2 = asci_xor128() % (in_feature_count);
        asci_Feature f1 = in_features[sample_i1];
        asci_Feature f2 = in_features[sample_i2];

        // TODO: Use gradient info?
        float dx = (float)(f2.x - f1.x);
        float dy = (float)(f2.y - f1.y);
        float t = atan((float)dy / (float)dx) + ASCI_PI / 2.0f;
        float c = cos(t);
        float s = sin(t);
        float r = f1.x*c + f1.y*s;

        if (r < r_min) r_min = r;
        if (r > r_max) r_max = r;
        if (t < t_min) t_min = t;
        if (t > t_max) t_max = t;

        out_votes[sample].t = t;
        out_votes[sample].r = r;
        out_votes[sample].x1 = f1.x;
        out_votes[sample].y1 = f1.y;
        out_votes[sample].x2 = f2.x;
        out_votes[sample].y2 = f2.y;
    }

    *out_t_min = t_min;
    *out_t_max = t_max;
    *out_r_min = r_min;
    *out_r_max = r_max;
    *out_count = count;
}

struct asci_Quad
{
    float t0;
    float t1;
    float r0;
    float r1;
    float mt;
    float mr;
    int32_t count;
    int32_t overlaps;
    bool merged;
};

void asc_find_lines(
    uint8_t  *in_rgb,
    uint8_t  *in_gray,
    int32_t   in_width,
    int32_t   in_height,
    asc_Line **out_lines,
    int32_t  *out_count,
    int16_t   param_feature_threshold,
    int32_t   param_max_ms_iterations,
    float     param_cluster_size_t,
    float     param_cluster_size_r)
{
    // May want to let these be user-parameters
    int16_t threshold = param_feature_threshold;
    int32_t max_iterations = param_max_ms_iterations;
    float cluster_size_t = param_cluster_size_t;
    float cluster_size_r = param_cluster_size_r;

    static asci_Feature features[1920*1080];
    int32_t feature_count = 0;

    asci_sobel(
        in_rgb,
        in_gray,
        in_width,
        in_height,
        threshold,
        features,
        &feature_count);

    const int32_t sample_count = 4096;
    static asci_Vote votes[sample_count];
    int32_t vote_count = 0;
    float t_min = 0.0f;
    float t_max = 0.0f;
    float r_min = 0.0f;
    float r_max = 0.0f;
    asci_hough(
        features,
        feature_count,
        sample_count,
        votes,
        &vote_count,
        &t_min, &t_max,
        &r_min, &r_max);

    // TIMING("cluster detection");
    int32_t quad_count_t = 1 + (int32_t)((t_max - t_min) / cluster_size_t);
    int32_t quad_count_r = 1 + (int32_t)((r_max - r_min) / cluster_size_r);

    // Distribute quads on a grid
    int32_t quad_count = quad_count_t*quad_count_r;
    asci_Quad *quads = (asci_Quad*)calloc(quad_count, sizeof(asci_Quad));
    asci_Quad *quads_swap = (asci_Quad*)calloc(quad_count, sizeof(asci_Quad));
    {
        asci_Quad *quad = quads;
        for (int32_t ri = 0; ri < quad_count_r; ri++)
        for (int32_t ti = 0; ti < quad_count_t; ti++)
        {
            quad->r0 = r_min + ri*cluster_size_r;
            quad->t0 = t_min + ti*cluster_size_t;
            quad->r1 = r_min + (ri+1)*cluster_size_r;
            quad->t1 = t_min + (ti+1)*cluster_size_t;
            quad->mt = 0.0f;
            quad->mr = 0.0f;
            quad->count = 0;
            quad->overlaps = 0;
            quad->merged = false;
            quad++;
        }
    }

    // Perform iterative mean-shift cluster detection
    // TODO(Simen): Terminate loop when the quads have converged?
    for (int32_t iteration = 0; iteration < max_iterations; iteration++)
    {
        // compute center of mass for each quad
        // and prune quads with low count
        int32_t new_quad_count = 0;
        for (int32_t qi = 0; qi < quad_count; qi++)
        {
            asci_Quad quad = quads[qi];
            quad.count = 0;
            quad.mr = 0.0f;
            quad.mt = 0.0f;
            float ct = (quad.t0 + quad.t1) / 2.0f;
            float cr = (quad.r0 + quad.r1) / 2.0f;

            for (int32_t i = 0; i < sample_count; i++)
            {
                asci_Vote vote = votes[i];
                float r = vote.r;
                float t = vote.t;
                if (r >= quad.r0 && r <= quad.r1 &&
                    t >= quad.t0 && t <= quad.t1)
                {
                    quad.mt += t - ct;
                    quad.mr += r - cr;
                    quad.count++;
                }
            }

            // Ignore quads with nothing in them, since they
            // will not move at all nor contribute to any lines.
            if (quad.count == 0)
            {
                continue;
            }
            else
            {
                quad.mt /= (float)quad.count;
                quad.mr /= (float)quad.count;
                quads_swap[new_quad_count++] = quad;
            }
        }

        // swap buffers
        {
            quad_count = new_quad_count;
            new_quad_count = 0;
            asci_Quad *temp = quads;
            quads = quads_swap;
            quads_swap = temp;
        }

        // Merge quads that are "close" together into one quad
        for (int32_t ia = 0; ia < quad_count; ia++)
        {
            asci_Quad a = quads[ia];
            if (a.merged)
                continue;
            int32_t overlaps = 0;
            float a_center_t = (a.t0 + a.t1) / 2.0f;
            float a_center_r = (a.r0 + a.r1) / 2.0f;
            for (int32_t ib = 0; ib < quad_count; ib++)
            {
                asci_Quad b = quads[ib];
                if (ib == ia || b.merged)
                    continue;
                float b_center_t = (b.t0 + b.t1) / 2.0f;
                float b_center_r = (b.r0 + b.r1) / 2.0f;
                float dt = (b_center_t+b.mt)-(a_center_t+a.mt);
                float dr = (b_center_r+b.mr)-(a_center_r+a.mr);
                if (abs(dt) < 0.1f*cluster_size_t &&
                    abs(dr) < 0.1f*cluster_size_r) // TODO: formalize threshold
                {
                    quads[ib].merged = true;
                    overlaps += 1+quads[ib].overlaps;
                }
            }

            // TODO: Do we want to sort quads_swap by decreasing
            // overlap count? Sometimes the quad with the lower
            // overlap takes the position of the one with higher
            // overlap... which I guess is ok maybe?
            a.overlaps += overlaps;
            quads_swap[new_quad_count++] = a;
        }

        // swap buffers
        {
            quad_count = new_quad_count;
            new_quad_count = 0;
            asci_Quad *temp = quads;
            quads = quads_swap;
            quads_swap = temp;
        }

        GDB("hough output",
        {
            Clear(0.0f, 0.0f, 0.0f, 1.0f);
            Ortho(t_min, t_max, r_min, r_max);
            BlendMode(GL_ONE, GL_ONE);

            glPointSize(4.0f);
            glBegin(GL_POINTS);
            {
                glColor4f(0.2f*0.3f, 0.2f*0.5f, 0.2f*0.8f, 1.0f);
                for (s32 i = 0; i < sample_count; i++)
                {
                    asci_Vote vote = votes[i];
                    glVertex2f(vote.t, vote.r);
                }
            }
            glEnd();

            r32 mouse_t = t_min + (t_max-t_min)*(0.5f+0.5f*input.mouse.x);
            r32 mouse_r = r_min + (r_max-r_min)*(0.5f-0.5f*input.mouse.y);

            BlendMode(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glLineWidth(2.0f);
            glBegin(GL_LINES);
            {
                for (s32 i = 0; i < quad_count; i++)
                {
                    asci_Quad quad = quads[i];

                    bool hover = false;
                    r32 ct = (quad.t0 + quad.t1)/2.0f;
                    r32 cr = (quad.r0 + quad.r1)/2.0f;
                    if (mouse_t >= quad.t0 && mouse_t <= quad.t1 &&
                        mouse_r >= quad.r0 && mouse_r <= quad.r1)
                    {
                        hover = true;
                        SetTooltip("(%.2f %.2f)\ncount: %d\noverlaps: %d\niteration: %d\nactive: %d",
                                   ct, cr, quad.count, quad.overlaps, iteration, quad_count);
                    }

                    if (hover)
                    {
                        glColor4f(1.0f, 0.3f, 0.3f, 1.0f);
                        glVertex2f(quad.t0, cr + quad.mr);
                        glVertex2f(quad.t1, cr + quad.mr);
                        glVertex2f(ct + quad.mt, quad.r0);
                        glVertex2f(ct + quad.mt, quad.r1);
                    }
                    else if (quad.overlaps >= 2 && quad.count > 25)
                        glColor4f(1.0f, 0.3f, 0.3f, 0.5f);
                    else
                        glColor4f(1.0f, 1.0f, 1.0f, 0.1f);

                    DrawQuad(quad.t0, quad.r0,
                             quad.t1 - quad.t0,
                             quad.r1 - quad.r0);

                }
            }
            glEnd();
        });

        // mean shift (TODO: merge with above for)
        for (int32_t qi = 0; qi < quad_count; qi++)
        {
            asci_Quad *quad = &quads[qi];
            float ct = (quad->t0 + quad->t1) / 2.0f;
            float cr = (quad->r0 + quad->r1) / 2.0f;
            quad->t0 += quad->mt;
            quad->r0 += quad->mr;
            quad->t1 = quad->t0 + cluster_size_t;
            quad->r1 = quad->r0 + cluster_size_r;
        }
    }

    int32_t final_lines_count = 0;
    asc_Line *final_lines = (asc_Line*)calloc(sample_count, sizeof(asc_Line));
    {
        for (int32_t qi = 0; qi < quad_count; qi++)
        {
            asci_Quad quad = quads[qi];
            // TODO: Formalize this check
            if (quad.overlaps > 4 && quad.count > 100)
            {
                // TODO: Use CoM?
                float t = (quad.t0 + quad.t1)/2.0f;
                float r = (quad.r0 + quad.r1)/2.0f;

                // Note (Simen): If we DO normalize the
                // point coordinates, make sure that we
                // use a UNIFORM scale. Otherwise, we
                // will be morphing the slope of the normal.
                float normal_x = cos(t);
                float normal_y = sin(t);
                float tangent_x = normal_y;
                float tangent_y = -normal_x;

                float normalization_factor = 1.0f / in_width;

                // Compute two base points from which
                // line distance will be measured relative to.
                float x0, y0, x1, y1;
                {
                    if (abs(normal_y) > abs(normal_x))
                    {
                        x0 = 0.0f;
                        x1 = in_width;
                        y0 = (r-x0*normal_x)/normal_y;
                        y1 = (r-x1*normal_x)/normal_y;
                    }
                    else
                    {
                        y0 = 0.0f;
                        y1 = in_height;
                        x0 = (r-y0*normal_y)/normal_x;
                        x1 = (r-y1*normal_y)/normal_x;
                    }
                }

                // Note(Simen): We normalize the coordinates before
                // computing statistical properties. This is to avoid
                // the numerical issues related to large numbers.
                x0 *= normalization_factor;
                x1 *= normalization_factor;
                y0 *= normalization_factor;
                y1 *= normalization_factor;

                // Note(Simen): Estimate normalized line bounds
                float l0, l1;
                {
                    int32_t N = 0;
                    float mean_sum = 0.0f;
                    float var_sum1 = 0.0f;
                    float var_sum2 = 0.0f;

                    // TODO: Study numerical issues here
                    for (int32_t sample = 0; sample < sample_count; sample++)
                    {
                        // Skip samples that do not lie inside the quad
                        if (votes[sample].r < quad.r0 ||
                            votes[sample].r > quad.r1 ||
                            votes[sample].t < quad.t0 ||
                            votes[sample].t > quad.t1)
                            continue;

                        // Each vote came from two feature point samples
                        // So I compute the distance along the line for
                        // both of the source features.
                        float sample_xs[] = {
                            votes[sample].x1 * normalization_factor,
                            votes[sample].x2 * normalization_factor
                        };
                        float sample_ys[] = {
                            votes[sample].y1 * normalization_factor,
                            votes[sample].y2 * normalization_factor
                        };
                        for (int32_t i = 0; i < 2; i++)
                        {
                            float sample_x = sample_xs[i];
                            float sample_y = sample_ys[i];
                            float dx = sample_x - x0;
                            float dy = sample_y - y0;
                            float p = dx*normal_x + dy*normal_y;

                            // Note(Simen): This is kind of a hack fix
                            // for the outlier problem. Ideally, we want
                            // to do some sort of least-squares regression
                            // to a) find the actually best line (instead
                            // of using CoM parameters) and b) fit the
                            // uniform distribution.
                            if (abs(p) > 15.0f * normalization_factor)
                                continue;

                            float l = dx*tangent_x + dy*tangent_y;

                            // See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                            float cond_shift = 0.5f;
                            mean_sum += l;
                            var_sum1 += l-cond_shift;
                            var_sum2 += (l-cond_shift)*(l-cond_shift);

                            N++;
                        }
                    }

                    float mean = mean_sum / N;
                    float var = (var_sum2 - var_sum1*var_sum1/N)/(N-1);
                    float var_std = sqrt(var);
                    float sqrt3 = 1.7321f;
                    l0 = mean - sqrt3*var_std;
                    l1 = mean + sqrt3*var_std;
                }

                // Note(Simen): Estimate dominant color of the line
                // by sampling input points along the best fit line.
                float color_r, color_g, color_b;
                {
                    int32_t color_samples = 32;
                    float sum_color_r = 0.0f;
                    float sum_color_g = 0.0f;
                    float sum_color_b = 0.0f;
                    float dl = 1.0f / color_samples;
                    for (int32_t i = 0; i < color_samples; i++)
                    {
                        float l = l0 + (l1-l0)*i*dl;
                        float x_ndc = x0 + tangent_x*l;
                        float y_ndc = y0 + tangent_y*l;
                        int32_t x_rgb = asci_round_positive(x_ndc / normalization_factor);
                        int32_t y_rgb = asci_round_positive(y_ndc / normalization_factor);
                        x_rgb = asci_clamp_s32(x_rgb, 0, in_width-1);
                        y_rgb = asci_clamp_s32(y_rgb, 0, in_height-1);
                        uint8_t *rgb_pixel = &in_rgb[(x_rgb+y_rgb*in_width)*3];
                        sum_color_r += rgb_pixel[0] / 255.0f;
                        sum_color_g += rgb_pixel[1] / 255.0f;
                        sum_color_b += rgb_pixel[2] / 255.0f;
                    }
                    color_r = sum_color_r / color_samples;
                    color_g = sum_color_g / color_samples;
                    color_b = sum_color_b / color_samples;
                }

                asc_Line line = {0};
                line.t = t;
                line.r = r;
                line.x_min = (x0 + tangent_x * l0) / normalization_factor;
                line.x_max = (x0 + tangent_x * l1) / normalization_factor;
                line.y_min = (y0 + tangent_y * l0) / normalization_factor;
                line.y_max = (y0 + tangent_y * l1) / normalization_factor;
                line.color_r = color_r;
                line.color_g = color_g;
                line.color_b = color_b;
                final_lines[final_lines_count++] = line;
            }
        }
    }

    free(quads);
    free(quads_swap);
    *out_lines = final_lines;
    *out_count = final_lines_count;
}

#endif
