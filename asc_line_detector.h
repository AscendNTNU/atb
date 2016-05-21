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
// You can avoid including <assert.h> by defining your own version
// of ASCI_ASSERT.
//
// Changelog
// ------------------------------------------------------------------------
//   1.3 (06. mar 2016) Better error metrics for line rejection.
//                      Fisheye correction.
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
// normal_error_threshold, normal_error_std_threshold
//   I use two error metrics to decide which lines to keep, based
//   on the votes that supported any given line:
//     1) Mean normal distance to the line taken over the votes
//     2) Standard deviation of the normal distance
//   normal_error_threshold denotes the maximum value of the
//   mean distance to be accepted. normal_error_var_threshold
//   denotes the maximum value of the standard deviation for
//   the line to be accepted. Both of these are in unit pixels.
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

struct asc_LineDetectorOptions
{
    int16_t sobel_threshold;
    int32_t hough_sample_count;
    float suppression_window_t;
    float suppression_window_r;
    float peak_exit_threshold;
    float normal_error_threshold;
    float normal_error_std_threshold;

    bool correct_fisheye;
    float fisheye_radius;   // Distance in pixels of the fisheye lens radius
                            // (The lens perimeter is usually visible near the edges as a black region)
    float fisheye_fov;      // Field of view of the fisheye lens (e.g. 180 degrees)
    float fisheye_center_x; // Center of distortion in x (e.g. width/2)
    float fisheye_center_y; // Center of distortion in y (e.g. height/2)
    float pinhole_fov_x;    // Desired horizontal field of view of the pinhole projection
};

// in_rgb
//   A densily packed array of 8bit RGB values
// in_gray
//   A densily packed array of 8bit grayscale values
// in_width, in_height
//   Width and height of in_rgb and in_gray.
// out_lines, max_out_count
//   The resulting list of lines found. You allocate this.
//   The list will be filled with at most max_out_count elements.
// out_count
//   The number of lines that were written to the output buffer.
void asc_find_lines(
    uint8_t   *in_rgb,
    uint8_t   *in_gray,
    int32_t    in_width,
    int32_t    in_height,
    asc_Line  *out_lines,
    int32_t   *out_count,
    int32_t    max_out_count,
    asc_LineDetectorOptions options);

#endif

#ifdef ASC_LINE_DETECTOR_IMPLEMENTATION
#ifndef ASCI_ASSERT
#include <assert.h>
#define ASCI_ASSERT assert
#endif

#include <stdlib.h>
#include <math.h>
#define ASCI_PI 3.1415926f
#define ASCI_MAX_WIDTH (1920)
#define ASCI_MAX_HEIGHT (1080)
#define ASCI_MAX_VOTE_COUNT (4096*8)
#define s32 int32_t
#define s16 int16_t
#define s08 int8_t
#define u32 uint32_t
#define u16 uint16_t
#define u08 uint8_t
#define r32 float

s32 asci_round_positive(r32 x)
{
    return (s32)(x + 0.5f);
}

s32 asci_floor_positive(r32 x)
{
    return (s32)(x);
}

s32 asci_clamp_s32(s32 x, s32 low, s32 high)
{
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

// @ intrinsics
r32 asci_max(r32 x, r32 y)
{
    if (x > y)
        return x;
    else
        return y;
}

// @ intrinsics
s32 asci_min_s32(s32 x, s32 y)
{
    if (x < y)
        return x;
    else
        return y;
}

s32 asci_max_s32(s32 x, s32 y)
{
    if (x > y)
        return x;
    else
        return y;
}

s16 asci_abs_s16(s16 x)
{
    return abs(x);
}

r32 asci_abs_r32(r32 x)
{
    return fabs(x);
}

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
    for (s32 y = 1; y < in_height-1; y++)
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

            __m128i skip_value = _mm_set1_epi8(threshold);
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

            // @ SIMD Sobel

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
            s16 gg = asci_abs_s16(gx) + asci_abs_s16(gy);
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

void asci_fisheye_undistort(
    asci_Feature *in_features,
    s32 in_count,
    s32 in_width,
    s32 in_height,
    asci_Feature *out_features,
    s32 *out_count,
    r32 fisheye_radius,
    r32 fisheye_fov,
    r32 fisheye_center_x,
    r32 fisheye_center_y,
    r32 pinhole_fov_x)
{
    r32 pinhole_f = (in_width/2.0f)/tan(pinhole_fov_x/2.0f);
    s32 count = 0;
    for (s32 i = 0; i < in_count; i++)
    {
        asci_Feature feature = in_features[i];
        r32 xd = feature.x - fisheye_center_x;
        r32 yd = feature.y - fisheye_center_y;
        r32 rd = sqrt(xd*xd+yd*yd);
        r32 theta = (fisheye_fov/2.0f)*rd/fisheye_radius;
        if (theta > pinhole_fov_x/2.0f)
            continue;
        r32 ru = pinhole_f*tan(theta);

        r32 xu;
        r32 yu;
        if (rd > 1.0f)
        {
            xu = (xd/rd)*ru;
            yu = (yd/rd)*ru;
        }
        else // Handle limit case in center
        {
            xu = xd*ru;
            yu = yd*ru;
        }

        r32 gxu;
        r32 gyu;
        {
            r32 ff = fisheye_radius/(fisheye_fov/2.0f);
            r32 fp = pinhole_f;
            r32 gxd = (r32)feature.gx;
            r32 gyd = (r32)feature.gy;
            r32 DIDphid = -gxd*yd + gyd*xd;
            r32 DIDrd = (gxd*xd+gyd*yd)/rd;

            r32 DrdDru = (ff/fp)/((ru/fp)*(ru/fp)+1.0f);
            r32 DIDphiu = DIDphid;
            r32 DIDru = DIDrd*DrdDru;

            r32 DruDxu = xu/ru;
            r32 DruDyu = yu/ru;

            r32 DphiuDxu = 0.0f;
            r32 DphiuDyu = 0.0f;
            if (asci_abs_r32(yu) > 1.0f)
                DphiuDxu = ((xu*xu)/(ru*ru)-1.0f)/yu;
            if (asci_abs_r32(xu) > 1.0f)
                DphiuDyu = (1.0f-(yu*yu)/(ru*ru))/xu;

            gxu = DIDru*DruDxu + DIDphiu*DphiuDxu;
            gyu = DIDru*DruDyu + DIDphiu*DphiuDyu;

            // Renormalize
            r32 ggu = sqrt(gxu*gxu+gyu*gyu);
            gxu *= feature.gg/ggu;
            gyu *= feature.gg/ggu;
        }

        s32 ix = asci_round_positive(fisheye_center_x+xu);
        s32 iy = asci_round_positive(fisheye_center_y+yu);

        feature.x = ix;
        feature.y = iy;
        feature.gx = (s16)gxu;
        feature.gy = (s16)gyu;
        out_features[count++] = feature;
    }

    *out_count = count;
}

struct asci_HoughCell
{
    r32 avg_r;
    r32 avg_t;
    s32 count;
};

void asc_find_lines(
    u08 *in_rgb,
    u08 *in_gray,
    s32 in_width,
    s32 in_height,
    asc_Line *out_lines,
    s32 *out_count,
    s32 max_out_count,
    asc_LineDetectorOptions options)
{
    ASCI_ASSERT(in_width <= ASCI_MAX_WIDTH);
    ASCI_ASSERT(in_height <= ASCI_MAX_HEIGHT);
    static asci_Feature features[ASCI_MAX_WIDTH*ASCI_MAX_HEIGHT];
    s32 feature_count = 0;
    asci_sobel(
        in_gray,
        in_width,
        in_height,
        options.sobel_threshold,
        features,
        &feature_count);

    #ifdef ASCDEBUG
    VDBBS("sobel features");
    {
        vdbOrtho(0.0f, in_width, 0.0f, in_height);
        glPointSize(4.0f);
        vdbClear(0.0f, 0.0f, 0.0f, 1.0f);
        glBegin(GL_POINTS);
        {
            for (int i = 0; i < feature_count; i++)
            {
                r32 gg = (r32)features[i].gg;
                r32 gx = (r32)features[i].gx/gg;
                r32 gy = (r32)features[i].gy/gg;
                r32 x = (r32)features[i].x;
                r32 y = (r32)features[i].y;

                glColor4f(0.5f+0.5f*gx, 0.5f+0.5f*gy, 0.5f, 1.0f);
                glVertex2f(x, y);
            }
        }
        glEnd();
    }
    VDBE();
    #endif

    if (feature_count == 0)
    {
        *out_count = 0;
        return;
    }

    if (options.correct_fisheye)
    {
        // Here I'm modifying the features array in place
        asci_fisheye_undistort(features, feature_count,
                               in_width, in_height,
                               features, &feature_count,
                               options.fisheye_radius,
                               options.fisheye_fov,
                               options.fisheye_center_x,
                               options.fisheye_center_y,
                               options.pinhole_fov_x);
    }

    #ifdef ASCDEBUG
    VDBB("features lines");
    {
        vdbClear(0.0f, 0.0f, 0.0f, 1.0f);
        vdbOrtho(0.0f, in_width, 0.0f, in_height);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        {
            for (int j = 0; j < feature_count; j++)
            {
                r32 gg = (r32)features[j].gg;
                r32 gx = (r32)features[j].gx/gg;
                r32 gy = (r32)features[j].gy/gg;
                r32 x = (r32)features[j].x;
                r32 y = (r32)features[j].y;

                glColor4f(0.2f, 0.2f, 0.3f, 1.0f);
                glVertex2f(x, y);
            }
        }
        glEnd();

        r32 mouse_x = (0.5f+0.5f*MOUSEX)*in_width;
        r32 mouse_y = (0.5f-0.5f*MOUSEY)*in_height;

        r32 avg_t = 0.0f;
        r32 avg_x = 0.0f;
        r32 avg_y = 0.0f;
        s32 avg_n = 0;
        Begin("Thetas");
        for (int j = 0; j < feature_count; j++)
        {
            asci_Feature f = features[j];
            r32 x = f.x;
            r32 y = f.y;

            r32 dx = x-mouse_x;
            r32 dy = y-mouse_y;
            if (dx*dx+dy*dy > 5.0f)
            {
                continue;
            }

            r32 t = atan2(f.gy, f.gx);
            if (t < 0.0f)
                t += ASCI_PI;

            if (abs(t-ASCI_PI - avg_t / (r32)avg_n) <
                abs(t - avg_t / (r32)avg_n))
                t -= ASCI_PI;

            avg_t += t;
            avg_x += x;
            avg_y += y;
            avg_n++;

            Text("%.2f\n", t);
        }
        End();

        if (avg_n > 0)
        {
            avg_t /= (r32)avg_n;
            avg_x /= (r32)avg_n;
            avg_y /= (r32)avg_n;
            r32 avg_r = avg_x*cos(avg_t) + avg_y*sin(avg_t);
            r32 normal_x = cos(avg_t);
            r32 normal_y = sin(avg_t);

            r32 x0;
            r32 y0;
            r32 x1;
            r32 y1;
            {
                if (abs(normal_y) > abs(normal_x))
                {
                    x0 = 0.0f;
                    x1 = in_width;
                    y0 = (avg_r-x0*normal_x)/normal_y;
                    y1 = (avg_r-x1*normal_x)/normal_y;
                }
                else
                {
                    y0 = 0.0f;
                    y1 = in_height;
                    x0 = (avg_r-y0*normal_y)/normal_x;
                    x1 = (avg_r-y1*normal_y)/normal_x;
                }
            }

            glBegin(GL_LINES);
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
            glEnd();

            glPointSize(10.0f);
            glBegin(GL_POINTS);
            glColor4f(1.0f, 0.2f, 0.1f, 1.0f);
            glVertex2f(avg_x, avg_y);
            glEnd();
        }
    }
    VDBE();
    #endif

    int lines_found = 0;
    const int bins_t = 128;
    const int bins_r = 128;
    const r32 r_max = sqrt((r32)(in_width*in_width+in_height*in_height));
    const r32 r_min = -r_max;
    asci_HoughCell histogram[bins_t*bins_r];
    asci_HoughCell histogram_maxima[bins_t*bins_r];
    s32 dilated_counts[bins_t*bins_r];
    for (int i = 0; i < bins_t*bins_r; i++)
    {
        histogram[i].count = 0;
        histogram[i].avg_t = 0.0f;
        histogram[i].avg_r = 0.0f;

        histogram_maxima[i].count = 0;
        histogram_maxima[i].avg_t = 0.0f;
        histogram_maxima[i].avg_r = 0.0f;
        dilated_counts[i] = 0;
    }

    #define tr_to_i(ti, ri) ((ti)+(ri)*bins_t)

    s32 max_count = 0;
    s32 processed_count = 0;
    s32 count_threshold = 50;
    // @ SIMD
    // @ Lookup tables for cos(t) and sin(t)?
    for (int i = 0; i < feature_count; i++)
    {
        processed_count++;
        asci_Feature f = features[i];
        r32 x = f.x;
        r32 y = f.y;

        r32 t_0 = atan2(f.gy, f.gx);
        if (t_0 < 0.0f)
            t_0 += ASCI_PI;

        int ti_0 = asci_floor_positive(bins_t * t_0 / ASCI_PI);
        #if 1
        for (int ti  = asci_clamp_s32(ti_0-2, 0, bins_t-1);
                 ti <= asci_clamp_s32(ti_0+2, 0, bins_t-1);
                 ti++)
        #else
        int ti = ti_0;
        #endif
        {
            // @ Blue noise dither + temporal filter
            // r32 noise = (asci_xor128() % 1024) / 1024.0f;
            // r32 t = ASCI_PI * (ti + noise) / (r32)bins_t;
            // r32 t = t_0 + (noise/bins_t)*ASCI_PI;
            // r32 t = t_0;
            r32 t = ASCI_PI * ti / (r32)bins_t;
            r32 r = x*cos(t)+y*sin(t);
            int ri = asci_floor_positive(bins_r * (r - r_min) / (r_max - r_min));
            ri = asci_clamp_s32(ri, 0, bins_r-1);
            // ti = asci_clamp_s32(ti, 0, bins_t-1);

            // @ Blue noise dither
            // Leak out as kernel instead of focus on single cell

            asci_HoughCell *hc = &histogram[tr_to_i(ti, ri)];
            hc->avg_t += t;
            hc->avg_r += r;
            s32 new_count = ++hc->count;
            if (new_count > max_count)
                max_count = new_count;
        }
        // i += (asci_xor128() % 2);
    }

    // Compute cell averages
    for (int i = 0; i < bins_t*bins_r; i++)
    {
        if (histogram[i].count > 0)
        {
            histogram[i].avg_t /= (r32)histogram[i].count;
            histogram[i].avg_r /= (r32)histogram[i].count;
        }
    }

    // Find local maxima
    // @ Local maxima block size

    int lm_radius = 5;

    // Dilate histogram
    for (int ri = 0; ri < bins_r; ri++)
    for (int ti = 0; ti < bins_t; ti++)
    {
        #if 1
        int c = histogram[tr_to_i(ti, ri)].count;
        // @ identification with theta
        int nri0 = asci_max_s32(ri-lm_radius, 0);
        int nri1 = asci_min_s32(ri+lm_radius, bins_r-1);
        int nti0 = asci_max_s32(ti-lm_radius, 0);
        int nti1 = asci_min_s32(ti+lm_radius, bins_t-1);
        for (int nri = nri0; nri <= nri1; nri++)
        for (int nti = nti0; nti <= nti1; nti++)
        {
            c = asci_max_s32(c, histogram[tr_to_i(nti, nri)].count);
        }
        dilated_counts[tr_to_i(ti, ri)] = c;
        #else
        r32 c00 = histogram[(ti-1)+(ri-1)*bins_t].count;
        r32 c = c00;
        r32 c10 = histogram[(ti+0)+(ri-1)*bins_t].count; if (c10 > c) c = c10;
        r32 c20 = histogram[(ti+1)+(ri-1)*bins_t].count; if (c20 > c) c = c20;

        r32 c01 = histogram[(ti-1)+(ri+0)*bins_t].count; if (c01 > c) c = c01;
        r32 c11 = histogram[(ti+0)+(ri+0)*bins_t].count; if (c11 > c) c = c11;
        r32 c21 = histogram[(ti+1)+(ri+0)*bins_t].count; if (c21 > c) c = c21;

        r32 c02 = histogram[(ti-1)+(ri+1)*bins_t].count; if (c02 > c) c = c02;
        r32 c12 = histogram[(ti+0)+(ri+1)*bins_t].count; if (c12 > c) c = c12;
        r32 c22 = histogram[(ti+1)+(ri+1)*bins_t].count; if (c22 > c) c = c22;
        dilated_counts[tr_to_i(ti, ri)] = c;
        #endif
    }

    // Nonmaximum suppression
    for (int ri = lm_radius; ri < bins_r-lm_radius; ri++)
    for (int ti = lm_radius; ti < bins_t-lm_radius; ti++)
    {
        s32 c_original = histogram[tr_to_i(ti, ri)].count;
        if (c_original < 1)
            continue;
        s32 c_dilated = dilated_counts[tr_to_i(ti, ri)];
        if (c_original == c_dilated)
        {
            r32 avg_t = 0.0f;
            r32 avg_r = 0.0f;
            s32 sum_n = 0;
            #if 1
            {
                r32 sum_t = 0.0f;
                r32 sum_r = 0.0f;
                for (int nri = ri-lm_radius; nri <= ri+lm_radius; nri++)
                for (int nti = ti-lm_radius; nti <= ti+lm_radius; nti++)
                {
                    asci_HoughCell c = histogram[nti+nri*bins_t];
                    sum_t += c.avg_t*c.count;
                    sum_r += c.avg_r*c.count;
                    sum_n += c.count;
                }
                avg_t = sum_t / (r32)sum_n;
                avg_r = sum_r / (r32)sum_n;
            }
            #else
            {
                asci_HoughCell c00 = histogram[(ti-1)+(ri-1)*bins_t];
                asci_HoughCell c10 = histogram[(ti+0)+(ri-1)*bins_t];
                asci_HoughCell c20 = histogram[(ti+1)+(ri-1)*bins_t];
                asci_HoughCell c01 = histogram[(ti-1)+(ri+0)*bins_t];
                asci_HoughCell c11 = histogram[(ti+0)+(ri+0)*bins_t];
                asci_HoughCell c21 = histogram[(ti+1)+(ri+0)*bins_t];
                asci_HoughCell c02 = histogram[(ti-1)+(ri+1)*bins_t];
                asci_HoughCell c12 = histogram[(ti+0)+(ri+1)*bins_t];
                asci_HoughCell c22 = histogram[(ti+1)+(ri+1)*bins_t];

                s32 n = c00.count+c10.count+c20.count+
                        c01.count+c11.count+c21.count+
                        c02.count+c12.count+c22.count;
                r32 m00 = c00.count/(r32)n; r32 t00 = c00.avg_t; r32 r00 = c00.avg_r;
                r32 m10 = c10.count/(r32)n; r32 t10 = c10.avg_t; r32 r10 = c10.avg_r;
                r32 m20 = c20.count/(r32)n; r32 t20 = c20.avg_t; r32 r20 = c20.avg_r;
                r32 m01 = c01.count/(r32)n; r32 t01 = c01.avg_t; r32 r01 = c01.avg_r;
                r32 m11 = c11.count/(r32)n; r32 t11 = c11.avg_t; r32 r11 = c11.avg_r;
                r32 m21 = c21.count/(r32)n; r32 t21 = c21.avg_t; r32 r21 = c21.avg_r;
                r32 m02 = c02.count/(r32)n; r32 t02 = c02.avg_t; r32 r02 = c02.avg_r;
                r32 m12 = c12.count/(r32)n; r32 t12 = c12.avg_t; r32 r12 = c12.avg_r;
                r32 m22 = c22.count/(r32)n; r32 t22 = c22.avg_t; r32 r22 = c22.avg_r;

                avg_t = t00*m00+t10*m10+t20*m20+t01*m01+t11*m11+t21*m21+t02*m02+t12*m12+t22*m22;
                avg_r = r00*m00+r10*m10+r20*m20+r01*m01+r11*m11+r21*m21+r02*m02+r12*m12+r22*m22;
                sum_n = n;
            }
            #endif

            histogram_maxima[tr_to_i(ti, ri)].avg_t = avg_t;
            histogram_maxima[tr_to_i(ti, ri)].avg_r = avg_r;
            histogram_maxima[tr_to_i(ti, ri)].count = sum_n;

            if (lines_found < max_count && sum_n > count_threshold)
            {
                r32 normal_x = cos(avg_t);
                r32 normal_y = sin(avg_t);
                r32 tangent_x = normal_y;
                r32 tangent_y = -normal_x;

                r32 x0, y0, x1, y1;
                {
                    if (abs(normal_y) > abs(normal_x))
                    {
                        x0 = 0.0f;
                        x1 = in_width;
                        y0 = (avg_r-x0*normal_x)/normal_y;
                        y1 = (avg_r-x1*normal_x)/normal_y;
                    }
                    else
                    {
                        y0 = 0.0f;
                        y1 = in_height;
                        x0 = (avg_r-y0*normal_y)/normal_x;
                        x1 = (avg_r-y1*normal_y)/normal_x;
                    }
                }

                asc_Line line = {0};
                line.t = avg_t;
                line.r = avg_r;
                line.x_min = x0;
                line.y_min = y0;
                line.x_max = x1;
                line.y_max = y1;
                out_lines[lines_found++] = line;
            }
        }
    }

    #ifdef ASCDEBUG
    VDBB("Hough");
    {
        s32 mouse_ti = asci_round_positive((0.5f+0.5f*MOUSEX)*bins_t);
        s32 mouse_ri = asci_round_positive((0.5f-0.5f*MOUSEY)*bins_r);
        r32 mouse_t = 0.0f;
        r32 mouse_r = 0.0f;
        s32 mouse_count = 0;

        vdbClear(0.0f, 0.0f, 0.0f, 1.0f);

        vdbOrtho(0.0f, in_width, 0.0f, in_height);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        {
            for (int i = 0; i < feature_count; i++)
            {
                r32 gg = (r32)features[i].gg;
                r32 gx = (r32)features[i].gx/gg;
                r32 gy = (r32)features[i].gy/gg;
                r32 x = (r32)features[i].x;
                r32 y = (r32)features[i].y;

                glColor4f(0.1f, 0.1f, 0.1f, 1.0f);
                glVertex2f(x, y);
            }
        }
        glEnd();

        vdbOrtho(0.0f, ASCI_PI, r_min, r_max);
        glPointSize(8.0f);
        glBegin(GL_POINTS);
        for (int ri = 1; ri < bins_r-1; ri++)
        for (int ti = 1; ti < bins_t-1; ti++)
        {
            #if 1
            s32 cm = histogram_maxima[tr_to_i(ti, ri)].count;
            if (cm > 150)
            {
                r32 r = r_min + (r_max-r_min) * ri / bins_r;
                r32 t = 0.0f + ASCI_PI * ti / bins_t;
                glColor4f(1.0f, 0.2f, 0.1f, 1.0f);
                glVertex2f(t, r);

                if (ti == mouse_ti && ri == mouse_ri)
                {
                    mouse_t = histogram_maxima[tr_to_i(ti, ri)].avg_t;
                    mouse_r = histogram_maxima[tr_to_i(ti, ri)].avg_r;
                    mouse_count = cm;
                    glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
                    SetTooltip("%d\n%.2f %.2f", cm, mouse_t, mouse_r);
                }
            }
            #else
            r32 r = r_min + (r_max-r_min) * ri / bins_r;
            r32 t = 0.0f + ASCI_PI * ti / bins_t;
            s32 c = histogram[tr_to_i(ti, ri)].count;
            s32 cm = histogram_maxima[tr_to_i(ti, ri)].count;
            if (c < 1)
                continue;
            if (cm > count_threshold)
                glColor4f(1.0f, 0.2f, 0.1f, 1.0f);
            else
                ColorRamp((r32)c/max_count);
            if (ti == mouse_ti && ri == mouse_ri)
            {
                mouse_t = histogram_maxima[tr_to_i(ti, ri)].avg_t;
                mouse_r = histogram_maxima[tr_to_i(ti, ri)].avg_r;
                mouse_count = cm;
                glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
                SetTooltip("%d\n%.2f %.2f", c, mouse_t, mouse_r);
            }
            glVertex2f(t, r);
            #endif
        }
        glEnd();

        r32 normal_x = cos(mouse_t);
        r32 normal_y = sin(mouse_t);
        r32 tangent_x = normal_y;
        r32 tangent_y = -normal_x;

        // Compute terminal points for drawing the line
        r32 x0;
        r32 y0;
        r32 x1;
        r32 y1;
        {
            if (abs(normal_y) > abs(normal_x))
            {
                x0 = 0.0f;
                x1 = in_width;
                y0 = (mouse_r-x0*normal_x)/normal_y;
                y1 = (mouse_r-x1*normal_x)/normal_y;
            }
            else
            {
                y0 = 0.0f;
                y1 = in_height;
                x0 = (mouse_r-y0*normal_y)/normal_x;
                x1 = (mouse_r-y1*normal_y)/normal_x;
            }
        }

        vdbOrtho(0.0f, in_width, 0.0f, in_height);
        glLineWidth(4.0f);
        glBegin(GL_LINES);
        glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
        glVertex2f(x0, y0);
        glVertex2f(x1, y1);
        glEnd();

        int support_count = 0;
        r32 mean_sum = 0.0f;
        r32 var_sum = 0.0f;

        // Draw supporting points
        glPointSize(4.0f);
        vdbAdditiveBlend();
        glBegin(GL_POINTS);
        glColor4f(0.05f*0.2f, 0.05f*0.5f, 0.05f*1.0f, 1.0f);
        for (int i = 0; i < feature_count; i++)
        {
            int x = features[i].x;
            int y = features[i].y;
            r32 d = abs(normal_x*x + normal_y*y - mouse_r);
            if (d < 40.0f)
            {
                support_count++;
                mean_sum += d;
                var_sum += d*d;
                glVertex2f(x, y);
            }
        }
        glEnd();

        int select_1 = asci_xor128() % feature_count;
        int select_2 = asci_xor128() % feature_count;
        {
            int x1 = features[select_1].x;
            int y1 = features[select_1].y;
            r32 d1 = abs(normal_x*x1 + normal_y*y1 - mouse_r);
            int x2 = features[select_2].x;
            int y2 = features[select_2].y;
            r32 d2 = abs(normal_x*x2 + normal_y*y2 - mouse_r);
            if (d1 < 40.0f && d2 < 40.0f)
            {
                glBegin(GL_LINES);
                glColor4f(1.0f, 0.2f, 0.1f, 1.0f);
                glVertex2f(x1, y1);
                glVertex2f(x2, y2);
                glEnd();
            }
        }

        r32 normal_error_mean = mean_sum/(r32)support_count;
        r32 normal_error_std = sqrt((var_sum/(r32)support_count) -
                                    normal_error_mean*normal_error_mean);
        Text("Processed count: %d", processed_count);
        Text("Lines found: %d", lines_found);
        Text("Mean: %.2f", normal_error_mean);
        Text("Std: %.2f", normal_error_std);
        Text("Inliers: %d", support_count);
        Text("Total: %d", mouse_count);
        Text("Total/Inliers: %.2f", support_count > 0 ? mouse_count / (r32)support_count : 0.0f);
    }
    VDBE();
    #endif

    #ifdef ASCDEBUG
    VDBB("final lines");
    {
        vdbOrtho(0.0f, in_width, 0.0f, in_height);
        glPointSize(2.0f);
        vdbClear(0.0f, 0.0f, 0.0f, 1.0f);
        glBegin(GL_POINTS);
        {
            for (int i = 0; i < feature_count; i++)
            {
                r32 x = (r32)features[i].x;
                r32 y = (r32)features[i].y;

                glColor4f(0.75f, 0.75f, 0.75f, 1.0f);
                glVertex2f(x, y);
            }
        }
        glEnd();

        glLineWidth(5.0f);
        glBegin(GL_LINES);
        for (s32 i = 0; i < lines_found; i++)
        {
            asc_Line line = out_lines[i];
            glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
            glVertex2f(line.x_min, line.y_min);
            glVertex2f(line.x_max, line.y_max);
        }
        glEnd();
    }
    VDBE();
    #endif

    *out_count = lines_found;
}

#undef s32
#undef s16
#undef s08
#undef u32
#undef u16
#undef u08
#undef r32
#endif

// Todo list

// @ Double threshold in sobel filter: upper and lower bound
