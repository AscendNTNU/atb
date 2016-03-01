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
//   1.2 (01. mar 2016) Bugfixes and robustification.
//                      Empty or almost empty images are now handled in a
//                      safer manner.
//   1.1 (27. feb 2016) First public release
//   1.0 (10. feb 2016) Ported code to single-file header library
//
// Parameter descriptions
// ------------------------------------------------------------------------
// The algorithms used have a number of cogs and levers that you
// can modify depending on your situation. Below I have written
// a summary of the parameters that you can specify, and how they
// relate to the output.
//
// EDGE DETECTOR
// The edge detector is a straightforward Sobel filter run on
// every input pixel. It computes a smoothed intensity gradient.
// The magnitude of this gradient is used to determine whether
// a given pixel as an edge (aka a feature) or not.
//
// sobel_threshold
//   A feature is kept if the magnitude of its intensity gradient
//   is greater than or equal to this threshold. If the edges of
//   the lines you are trying to detect are faint, you will want
//   this to be a small value.
//
// HOUGH TRANSFORM
// The particular Hough transform used is called the randomized
// Hough transform. It works by sampling from the feature space
// (the output from the edge detector) two features at a time,
// and voting for the line that goes through them. You can control
// the number of samples performed.
//
// hough_sample_count
//   The randomized Hough transforms performs hough_sample_count
//   random samples of the feature space, during the line voting
//   process.
//
// PEAK EXTRACTOR
// The peak extractor extracts plausible lines from the Hough
// transform space. When it extracts a peak, it sets neighboring
// votes to zero in a window. This is done to eliminate multiple
// detections of the same line. The size of this window may be
// specified in terms of extent in theta (radians) and extent
// in rho (pixels). Since the vote count of the extracted peaks
// decrease for each line found, you can early-exit the search by
// specifying the threshold vote count. If a peak is found with
// a count lower than this threshold, the search exits and the peak
// is rejected.
//
// suppression_window_t
//   Controls the extent of suppression of neighboring votes in
//   theta (radians) during the peak extraction.
// suppression_window_r
//   Controls the extent of suppression of neighboring votes in
//   rho (pixels) during the peak extraction.
// peak_exit_threshold
//   Controls the minimum number of votes a peak must have to
//   pass as a detected line. Specified as a fraction of the max
//   count in the Hough space.
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
    int32_t    param_max_out_count = 16,
    int16_t    param_sobel_threshold = 10,
    int32_t    param_hough_sample_count = 4096,
    float      param_suppression_window_t = 0.349f,
    float      param_suppression_window_r = 300.0f,
    float      param_peak_exit_threshold = 0.1f);

#endif

#ifdef ASC_LINE_DETECTOR_IMPLEMENTATION
#include <stdlib.h>
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

#ifdef ASC_LINE_DETECTOR_SSE
#include "xmmintrin.h"
#include "emmintrin.h"
#include "smmintrin.h"
void asci_sobel(
    u08 *in_gray,
    s32  in_width,
    s32  in_height,
    s16  threshold,
    asci_Feature *out_features,
    s32 *out_feature_count)
{
    s32 count = 0;
    for (s32 y = 1; y < in_height; y++)
    {
        for (s32 x = 1; x <= in_width-16; x += 16)
        {
            u08 *addr00 = in_gray + (y-1)*in_width + x-1;
            u08 *addr01 = in_gray + (y-1)*in_width + x;
            u08 *addr02 = in_gray + (y-1)*in_width + x+1;

            u08 *addr10 = in_gray + (y)*in_width + x-1;
            u08 *addr12 = in_gray + (y)*in_width + x+1;

            u08 *addr20 = in_gray + (y+1)*in_width + x-1;
            u08 *addr21 = in_gray + (y+1)*in_width + x;
            u08 *addr22 = in_gray + (y+1)*in_width + x+1;

            __m128i source00 = _mm_loadu_si128((__m128i*)addr00);
            __m128i source01 = _mm_loadu_si128((__m128i*)addr01);
            __m128i source02 = _mm_loadu_si128((__m128i*)addr02);

            __m128i source10 = _mm_loadu_si128((__m128i*)addr10);
            __m128i source12 = _mm_loadu_si128((__m128i*)addr12);

            __m128i source20 = _mm_loadu_si128((__m128i*)addr20);
            __m128i source21 = _mm_loadu_si128((__m128i*)addr21);
            __m128i source22 = _mm_loadu_si128((__m128i*)addr22);

            // divide pixels by 4

            __m128i shift_mask = _mm_set1_epi8(0x3F);
            source00 = _mm_and_si128(shift_mask, _mm_srli_epi16(source00, 2));
            source01 = _mm_and_si128(shift_mask, _mm_srli_epi16(source01, 2));
            source02 = _mm_and_si128(shift_mask, _mm_srli_epi16(source02, 2));

            source10 = _mm_and_si128(shift_mask, _mm_srli_epi16(source10, 2));
            source12 = _mm_and_si128(shift_mask, _mm_srli_epi16(source12, 2));

            source20 = _mm_and_si128(shift_mask, _mm_srli_epi16(source20, 2));
            source21 = _mm_and_si128(shift_mask, _mm_srli_epi16(source21, 2));
            source22 = _mm_and_si128(shift_mask, _mm_srli_epi16(source22, 2));

            // I compute the x and y gradients in their positive
            // and negative components, to fit everything in u08
            // values.

            // TODO: Div only by two for source12, source10,
            // source21 and source01.

            // px
            __m128i positive_x = _mm_set1_epi8(0);
            positive_x = _mm_adds_epu8(positive_x, source02);
            positive_x = _mm_adds_epu8(positive_x, source12);
            positive_x = _mm_adds_epu8(positive_x, source12);
            positive_x = _mm_adds_epu8(positive_x, source22);

            // nx
            __m128i negative_x = _mm_set1_epi8(0);
            negative_x = _mm_adds_epu8(negative_x, source00);
            negative_x = _mm_adds_epu8(negative_x, source10);
            negative_x = _mm_adds_epu8(negative_x, source10);
            negative_x = _mm_adds_epu8(negative_x, source20);

            // py
            __m128i positive_y = _mm_set1_epi8(0);
            positive_y = _mm_adds_epu8(positive_y, source20);
            positive_y = _mm_adds_epu8(positive_y, source21);
            positive_y = _mm_adds_epu8(positive_y, source21);
            positive_y = _mm_adds_epu8(positive_y, source22);

            // ny
            __m128i negative_y = _mm_set1_epi8(0);
            negative_y = _mm_adds_epu8(negative_y, source00);
            negative_y = _mm_adds_epu8(negative_y, source01);
            negative_y = _mm_adds_epu8(negative_y, source01);
            negative_y = _mm_adds_epu8(negative_y, source02);

            // Approximate magnitude of gradient by absolute value

            // x
            __m128i abs_gx = _mm_subs_epu8(
                _mm_max_epu8(positive_x, negative_x),
                _mm_min_epu8(positive_x, negative_x));

            // y
            __m128i abs_gy = _mm_subs_epu16(
                _mm_max_epu8(positive_y, negative_y),
                _mm_min_epu8(positive_y, negative_y));

            __m128i magnitude = _mm_adds_epu8(abs_gx, abs_gy);

            __m128i skip_value = _mm_set1_epi8(20);
            __m128i skip_cmp = _mm_cmplt_epi8(magnitude, skip_value);
            int move_mask = _mm_movemask_epi8(skip_cmp);
            if (move_mask == 0xffff)
            {
                continue;
            }

            u08 dst_magnitude[16];
            u08 dst_positive_x[16];
            u08 dst_negative_x[16];
            u08 dst_positive_y[16];
            u08 dst_negative_y[16];
            _mm_storeu_si128((__m128i*)dst_magnitude,  magnitude);
            _mm_storeu_si128((__m128i*)dst_positive_x, positive_x);
            _mm_storeu_si128((__m128i*)dst_negative_x, negative_x);
            _mm_storeu_si128((__m128i*)dst_positive_y, positive_y);
            _mm_storeu_si128((__m128i*)dst_negative_y, negative_y);

            // TODO: Compute average? Compute sum?
            // Skip pushing entire block if almost all lt 10

            for (s32 dx = 0; dx < 16; dx++)
            {
                if (dst_magnitude[dx] > threshold)
                {
                    asci_Feature feature = {0};
                    feature.x = x+dx;
                    feature.y = y;
                    feature.gx = (s16)dst_positive_x[dx]-(s16)dst_negative_x[dx];
                    feature.gy = (s16)dst_positive_y[dx]-(s16)dst_negative_y[dx];
                    feature.gg = (s16)dst_magnitude[dx];
                    out_features[count++] = feature;
                }
            }
        }
    }
    *out_feature_count = count;
}
#else
void asci_sobel(
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
#endif

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
    if (in_feature_count == 0)
    {
        *out_count = 0;
        *out_t_min = 0;
        *out_t_max = 0;
        *out_r_min = 0;
        *out_r_max = 0;
        return;
    }
    r32 t_min = ASCI_FLT_MAX;
    r32 t_max = -ASCI_FLT_MAX;
    r32 r_min = ASCI_FLT_MAX;
    r32 r_max = -ASCI_FLT_MAX;
    s32 count = 0;

    if (in_feature_count == 0)
    {
        *out_count = 0;
        *out_t_min = 0.0f;
        *out_t_max = 0.0f;
        *out_r_min = 0.0f;
        *out_r_max = 0.0f;
        return;
    }

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

        r32 t;
        s32 dx = f2.x - f1.x;
        s32 dy = f2.y - f1.y;
        if (dx == 0)
        {
            t = 0.0f;
        }
        else
        {
            t = atan((r32)dy / (r32)dx) + ASCI_PI / 2.0f;
        }
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
    s32 max_out_count,
    s16 sobel_threshold,
    s32 sample_count,
    r32 suppression_window_t,
    r32 suppression_window_r,
    r32 peak_exit_threshold)
{
    // TODO: This can be a static array. Let the user define max
    // dimensions for the image, and provide default sizes.
    asci_Feature *features = (asci_Feature*)calloc(in_width*in_height, sizeof(asci_Feature));
    s32 feature_count = 0;

    asci_sobel(
        in_gray,
        in_width,
        in_height,
        sobel_threshold,
        features,
        &feature_count);

    if (feature_count == 0)
    {
        *out_lines = 0;
        *out_count = 0;
        return;
    }

    asci_Vote *votes = (asci_Vote*)calloc(sample_count, sizeof(asci_Vote));
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

    if (vote_count == 0)
    {
        *out_count = 0;
        *out_lines = 0;
        return;
    }

    // I need to ensure that the ranges t_max-t_min and r_max-r_min
    // atleast as large as the suppression windows, since I iterate
    // over the neighborhoods later. If they are zero, it means that
    // we either found nothing, or that we only found one type of line.
    if (abs(t_max-t_min) < suppression_window_t*1.25f)
    {
        t_max = suppression_window_t*1.25f;
        t_min = 0.0f;
    }
    if (abs(r_max-r_min) < suppression_window_r*2.5f)
    {
        r_max = (suppression_window_r/2.0f)*1.25f;
        r_min = -r_max;
    }

    struct HoughCell
    {
        r32 avg_r;
        r32 avg_t;
        s32 count;
    };

    const s32 bins_t = 32;
    const s32 bins_r = 32;
    static HoughCell histogram[bins_r*bins_t];
    for (s32 i = 0; i < bins_r*bins_t; i++)
    {
        histogram[i].avg_r = 0.0f;
        histogram[i].avg_t = 0.0f;
        histogram[i].count = 0;
    }

    s32 histogram_max_count = 0.0f;
    for (s32 vote_index = 0; vote_index < vote_count; vote_index++)
    {
        asci_Vote vote = votes[vote_index];
        r32 t = vote.t;
        r32 r = vote.r;
        s32 ti = asci_clamp_s32(bins_t*(t-t_min)/(t_max-t_min), 0, bins_t-1);
        s32 ri = asci_clamp_s32(bins_r*(r-r_min)/(r_max-r_min), 0, bins_r-1);
        HoughCell *cell = &histogram[ti + ri*bins_t];
        cell->avg_r += r;
        cell->avg_t += t;
        cell->count++;
        if (cell->count > histogram_max_count)
            histogram_max_count = cell->count;
    }

    // TODO: Perform running average instead?
    for (s32 i = 0; i < bins_r*bins_t; i++)
    {
        if (histogram[i].count > 0)
        {
            histogram[i].avg_r /= (r32)histogram[i].count;
            histogram[i].avg_t /= (r32)histogram[i].count;
        }
    }

    // Peak extraction
    asc_Line *lines = (asc_Line*)calloc(max_out_count, sizeof(asc_Line));
    s32 lines_found = 0;
    r32 bin_size_t = (t_max-t_min) / bins_t;
    r32 bin_size_r = (r_max-r_min) / bins_r;
    s32 suppression_window_ti = asci_round_positive(suppression_window_t / bin_size_t);
    s32 suppression_window_ri = asci_round_positive(suppression_window_r / bin_size_r);
    if (suppression_window_ti % 2 != 0) suppression_window_ti++;
    if (suppression_window_ri % 2 != 0) suppression_window_ri++;
    for (s32 iteration = 0; iteration < max_out_count; iteration++)
    {
        // Extract max
        s32 peak_index = 0;
        s32 peak_count = 0;
        for (s32 i = 0; i < bins_t*bins_r; i++)
        {
            if (histogram[i].count > peak_count)
            {
                peak_count = histogram[i].count;
                peak_index = i;
            }
        }

        if (peak_count <= peak_exit_threshold*histogram_max_count)
        {
            break;
        }

        s32 peak_ti = peak_index % bins_t;
        s32 peak_ri = peak_index / bins_t;
        r32 peak_t = histogram[peak_index].avg_t;
        r32 peak_r = histogram[peak_index].avg_r;

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

        lines[lines_found].t = peak_t;
        lines[lines_found].r = peak_r;
        lines[lines_found].x_min = x0;
        lines[lines_found].y_min = y0;
        lines[lines_found].x_max = x1;
        lines[lines_found].y_max = y1;
        lines_found++;

        // Neighborhood suppression
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
                write_t = asci_clamp_s32(ti+bins_t, 0, bins_t-1);
                write_r = asci_clamp_s32(bins_r-ri, 0, bins_r-1);
            }
            else if (ti >= bins_t)
            {
                write_t = asci_clamp_s32(ti-bins_t, 0, bins_t-1);
                write_r = asci_clamp_s32(bins_r-1-ri, 0, bins_r-1);
            }
            else
            {
                write_t = asci_clamp_s32(ti, 0, bins_t-1);
                write_r = asci_clamp_s32(ri, 0, bins_r-1);
            }
            histogram[write_t + write_r*bins_t].count = 0;
        }
    }

    *out_lines = lines;
    *out_count = lines_found;

    free(votes);
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
