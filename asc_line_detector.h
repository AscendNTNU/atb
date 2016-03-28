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

// For debugging
#ifndef USE_GDB
#ifndef GDB
#define GDB(arg1, arg2) ;
#endif
#ifndef GDB_SKIP
#define GDB_SKIP(arg1, arg2) ;
#endif
#endif

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

s32 asci_clamp_s32(s32 x, s32 low, s32 high)
{
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

r32 asci_max(r32 x, r32 y)
{
    if (x > y) return x;
    else return y;
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
    #if 0
    GDB("fisheye tune",
    {
        glLineWidth(1.0f);
        glPointSize(2.0f);
        BlendMode();
        Ortho(0.0f, in_width, 0.0f, in_height);
        Clear(0.0f, 0.0f, 0.0f, 1.0f);
        glBegin(GL_LINES);
        s32 count = 0;
        for (s32 i = 0; i < in_count; i += 16)
        {
            asci_Feature feature = in_features[i];
            r32 xd = feature.x - fisheye_center_x;
            r32 yd = feature.y - fisheye_center_y;
            r32 rd = sqrt(xd*xd+yd*yd);
            r32 theta = (fisheye_fov/2.0f)*rd/fisheye_radius;
            if (theta > pinhole_fov_x/2.0f)
                continue;
            r32 ru = pinhole_f*tan(theta); // Obs! Different definition of ru

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
            }

            s32 ix = asci_round_positive(fisheye_center_x+xu);
            s32 iy = asci_round_positive(fisheye_center_y+yu);

            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glVertex2f(ix, iy);
            glVertex2f(ix+gxu/4.0f, iy+gyu/4.0f);
        }
        glEnd();
        SliderFloat("fisheye radius", &fisheye_radius, 500.0f, 1000.0f);
        SliderFloat("center x", &fisheye_center_x, 200.0f, 1000.0f);
        SliderFloat("center y", &fisheye_center_y, 200.0f, 1000.0f);
    });
    #endif

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

    GDB("fisheye",
    {
        glLineWidth(1.0f);
        glPointSize(2.0f);
        BlendMode();
        Ortho(0.0f, in_width, 0.0f, in_height);
        Clear(0.0f, 0.0f, 0.0f, 1.0f);
        glBegin(GL_POINTS);
        for (s32 i = 0; i < count; i++)
        {
            asci_Feature feature = out_features[i];

            r32 gg = (r32)feature.gg;
            r32 gx = (r32)feature.gx/gg;
            r32 gy = (r32)feature.gy/gg;

            glColor4f(0.5f+0.5f*gx, 0.5f+0.5f*gy, 0.5f, 1.0f);
            glVertex2f(feature.x, feature.y);
        }
        glEnd();
    });

    *out_count = count;
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
    s32 in_width,
    s32 in_height,
    asci_Vote *out_votes,
    s32 *out_count,
    r32 *out_r_min,
    r32 *out_r_max)
{
    if (in_feature_count == 0)
    {
        *out_count = 0;
        *out_r_min = 0.0f;
        *out_r_max = 0.0f;
        return;
    }

    // Initialize min and max values to their opposite bounds
    r32 r_max = -sqrt((r32)(in_width*in_width+in_height*in_height));
    r32 r_min = -r_max;

    s32 count = 0;
    r32 rejection_threshold = 0.2f; // @ Gradient rejection threshold
    for (s32 sample = 0; sample < sample_count; sample++)
    {
        // Draw two random features (edges) from the image
        s32 sample_i1 = asci_xor128() % (in_feature_count);
        s32 sample_i2 = asci_xor128() % (in_feature_count);
        asci_Feature f1 = in_features[sample_i1];
        asci_Feature f2 = in_features[sample_i2];

        // Reject the samples if the gradients differ too much
        {
            r32 dot = f1.gx*f2.gx + f1.gy*f2.gy;
            if (asci_abs_r32(dot) <= rejection_threshold*f1.gg*f2.gg)
            {
                continue;
            }
        }

        // Compute the angle of the normal (t) of the line drawn
        // between the two samples
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

        // Reject the samples if the normal of the connecting line
        // differs from the gradients of the samples too much
        if (f1.gg > 0 && f2.gg > 0) // Should not need this check
        {
            r32 dot1 = (f1.gx*c+f1.gy*s) / f1.gg;
            r32 dot2 = (f2.gx*c+f2.gy*s) / f2.gg;
            r32 adot = 0.5f*(asci_abs_r32(dot1) + asci_abs_r32(dot2));
            if (adot <= rejection_threshold)
            {
                continue;
            }
        }

        // Compute the line's distance to the origin
        r32 r = f1.x*c + f1.y*s;
        if (r > r_max) r_max = r;
        if (r < r_min) r_min = r;

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

    *out_r_min = r_min;
    *out_r_max = r_max;
    *out_count = count;
}

struct asci_HoughCell
{
    r32 avg_r;
    r32 avg_t;
    s32 count;
};

void asci_hough_histogram(
    asci_HoughCell *histogram,
    s32 bins_t,
    s32 bins_r,
    r32 t_min,
    r32 t_max,
    r32 r_min,
    r32 r_max,
    asci_Vote *votes,
    s32 vote_count,
    s32 *out_histogram_max_count)
{
    for (s32 i = 0; i < bins_r*bins_t; i++)
    {
        histogram[i].avg_r = 0.0f;
        histogram[i].avg_t = 0.0f;
        histogram[i].count = 0;
    }

    s32 histogram_max_count = 0;
    for (s32 i = 0; i < vote_count; i++)
    {
        asci_Vote vote = votes[i];
        r32 t = vote.t;
        r32 r = vote.r;
        s32 ti = asci_clamp_s32(bins_t*(t-t_min)/(t_max-t_min), 0, bins_t-1);
        s32 ri = asci_clamp_s32(bins_r*(r-r_min)/(r_max-r_min), 0, bins_r-1);
        asci_HoughCell *cell = &histogram[ti + ri*bins_t];
        cell->avg_r += r;
        cell->avg_t += t;
        cell->count++;
        if (cell->count > histogram_max_count)
            histogram_max_count = cell->count;
    }

    for (s32 i = 0; i < bins_r*bins_t; i++)
    {
        if (histogram[i].count > 0)
        {
            histogram[i].avg_r /= (r32)histogram[i].count;
            histogram[i].avg_t /= (r32)histogram[i].count;
        }
    }
    *out_histogram_max_count = histogram_max_count;
}

#define ASCI_TI_TO_T(ti) (t_min + (t_max-t_min)*ti/bins_t)
#define ASCI_RI_TO_R(ri) (r_min + (r_max-r_min)*ri/bins_r)

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

    GDB_SKIP("sobel features",
    {
        Ortho(0.0f, in_width, 0.0f, in_height);
        glPointSize(2.0f);
        Clear(0.0f, 0.0f, 0.0f, 1.0f);
        BlendMode();
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
    });

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

    // @ Experimental Hough transform
    #if 0
    {
        const int bins_t = 32;
        const int bins_r = 32;
        const r32 r_max = sqrt((r32)(in_width*in_width+in_height*in_height));
        const r32 r_min = -r_max;
        asci_HoughCell histogram[bins_t*bins_r];
        s32 dilated_counts[bins_t*bins_r];
        for (int i = 0; i < bins_t*bins_r; i++)
        {
            histogram[i].count = 0;
            histogram[i].avg_t = 0.0f;
            histogram[i].avg_r = 0.0f;
            dilated_counts[i] = 0;
        }

        s32 max_count = 0;
        for (int i = 0; i < feature_count; i++)
        {
            asci_Feature f = features[i];
            r32 x = f.x;
            r32 y = f.y;
            r32 t_0 = atan2(f.gy, f.gx);
            if (t_0 < 0.0f) t_0 += ASCI_PI;
            int ti_0 = bins_t * t_0 / ASCI_PI;
            for (int ti  = asci_clamp_s32(ti_0-2, 0, bins_t-1);
                     ti <= asci_clamp_s32(ti_0+2, 0, bins_t-1);
                     ti++)
            {
                r32 t = ASCI_PI * ti / (r32)bins_t;
                r32 r = x*cos(t)+y*sin(t);
                int ri = (int)(bins_r * (r - r_min) / (r_max - r_min));
                ri = asci_clamp_s32(ri, 0, bins_r-1);

                s32 new_count = ++histogram[ti+bins_t*ri].count;
                histogram[ti+bins_t*ri].avg_t += t;
                histogram[ti+bins_t*ri].avg_r += r;
                if (new_count > max_count)
                    max_count = new_count;
            }
        }
        for (int i = 0; i < bins_t*bins_r; i++)
        {
            histogram[i].avg_t /= (r32)histogram[i].count;
            histogram[i].avg_r /= (r32)histogram[i].count;
        }

        GDB("eht",
        {
            s32 mouse_ti = asci_round_positive((0.5f+0.5f*input.mouse.x)*bins_t);
            s32 mouse_ri = asci_round_positive((0.5f-0.5f*input.mouse.y)*bins_r);

            Clear(0.0f, 0.0f, 0.0f, 1.0f);

            Ortho(0.0f, in_width, 0.0f, in_height);
            glPointSize(2.0f);
            BlendMode();
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

            Ortho(0.0f, ASCI_PI, r_min, r_max);
            BlendMode();
            glPointSize(8.0f);
            glBegin(GL_POINTS);
            for (int ri = 1; ri < bins_r-1; ri++)
            for (int ti = 1; ti < bins_t-1; ti++)
            {
                r32 r = r_min + (r_max-r_min) * ri / bins_r;
                r32 t = 0.0f + ASCI_PI * ti / bins_t;
                s32 c = histogram[ti+ri*bins_t].count;
                if (c < 1.0f)
                    continue;
                ColorRamp((r32)c/max_count);
                glVertex2f(histogram[ti+ri*bins_t].avg_t,
                           histogram[ti+ri*bins_t].avg_r);
            }
            glEnd();

            if (Button("Detect local maxima"))
            {
                for (int ri = 1; ri < bins_r-1; ri++)
                for (int ti = 1; ti < bins_t-1; ti++)
                {
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
                    dilated_counts[ti+ri*bins_t] = c;
                }

                for (int i = 0; i < bins_t*bins_r; i++)
                {
                    s32 c_original = histogram[i].count;
                    if (c_original <= 1.0f)
                        continue;
                    s32 c_dilated = dilated_counts[i];
                    if (c_original < c_dilated)
                        histogram[i].count = 0;
                }
            }

            r32 mouse_t = ASCI_PI * ((0.5f+0.5f*input.mouse.x)*bins_t) / bins_t;
            r32 mouse_r = r_min + (r_max-r_min) * ((0.5f-0.5f*input.mouse.y)*bins_r) / bins_r;
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

            Ortho(0.0f, in_width, 0.0f, in_height);
            glLineWidth(4.0f);
            glBegin(GL_LINES);
            glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
            glEnd();
        });
    }
    #endif

    // @ Warn about parameters
    if (options.hough_sample_count > ASCI_MAX_VOTE_COUNT)
        options.hough_sample_count = ASCI_MAX_VOTE_COUNT;
    static asci_Vote votes[ASCI_MAX_VOTE_COUNT];
    s32 vote_count = 0;
    r32 r_min = 0.0f;
    r32 r_max = 0.0f;
    asci_hough(
        features,
        feature_count,
        options.hough_sample_count,
        in_width,
        in_height,
        votes,
        &vote_count,
        &r_min,
        &r_max);

    if (vote_count == 0)
    {
        *out_count = 0;
        return;
    }

    // I need to ensure that r_max-r_min atleast as large as the suppression
    // window, since I iterate over the window later. If the range is zero,
    // it means that we either found nothing, or that we only found one type
    // of line.
    if (asci_abs_r32(r_max-r_min) < options.suppression_window_r*2.5f)
    {
        r_max = (options.suppression_window_r/2.0f)*1.25f;
        r_min = -r_max;
    }

    // Quantize the line parameter votes into a histogram
    const s32 bins_t = 32;
    const s32 bins_r = 32;
    static asci_HoughCell histogram[bins_t*bins_r];
    s32 histogram_max_count = 0;
    r32 t_min = 0.0f;
    r32 t_max = ASCI_PI;
    asci_hough_histogram(
        histogram,
        bins_t,
        bins_r,
        t_min, t_max,
        r_min, r_max,
        votes,
        vote_count,
        &histogram_max_count);

    static s32 dilated_counts[bins_t*bins_r];
    GDB("eht2",
    {
        s32 mouse_ti = asci_round_positive((0.5f+0.5f*input.mouse.x)*bins_t);
        s32 mouse_ri = asci_round_positive((0.5f-0.5f*input.mouse.y)*bins_r);

        Clear(0.0f, 0.0f, 0.0f, 1.0f);

        Ortho(0.0f, in_width, 0.0f, in_height);
        glPointSize(2.0f);
        BlendMode();
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

        Ortho(0.0f, ASCI_PI, r_min, r_max);
        BlendMode();
        glPointSize(8.0f);
        glBegin(GL_POINTS);
        for (int ri = 1; ri < bins_r-1; ri++)
        for (int ti = 1; ti < bins_t-1; ti++)
        {
            r32 r = r_min + (r_max-r_min) * ri / bins_r;
            r32 t = 0.0f + ASCI_PI * ti / bins_t;
            s32 c = histogram[ti+ri*bins_t].count;
            if (c < 1)
                continue;
            ColorRamp(c / (0.2f*histogram_max_count));
            if (mouse_ti == ti && mouse_ri == ri)
            {
                glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
                SetTooltip("%d", c);
            }
            glVertex2f(histogram[ti+ri*bins_t].avg_t,
                       histogram[ti+ri*bins_t].avg_r);
        }
        glEnd();

        if (Button("Detect local maxima"))
        {
            for (int ri = 1; ri < bins_r-1; ri++)
            for (int ti = 1; ti < bins_t-1; ti++)
            {
                s32 c00 = histogram[(ti-1)+(ri-1)*bins_t].count;
                s32 c = c00;
                s32 c10 = histogram[(ti+0)+(ri-1)*bins_t].count; if (c10 > c) c = c10;
                s32 c20 = histogram[(ti+1)+(ri-1)*bins_t].count; if (c20 > c) c = c20;

                s32 c01 = histogram[(ti-1)+(ri+0)*bins_t].count; if (c01 > c) c = c01;
                s32 c11 = histogram[(ti+0)+(ri+0)*bins_t].count; if (c11 > c) c = c11;
                s32 c21 = histogram[(ti+1)+(ri+0)*bins_t].count; if (c21 > c) c = c21;

                s32 c02 = histogram[(ti-1)+(ri+1)*bins_t].count; if (c02 > c) c = c02;
                s32 c12 = histogram[(ti+0)+(ri+1)*bins_t].count; if (c12 > c) c = c12;
                s32 c22 = histogram[(ti+1)+(ri+1)*bins_t].count; if (c22 > c) c = c22;
                dilated_counts[ti+ri*bins_t] = c;
            }

            for (int i = 0; i < bins_t*bins_r; i++)
            {
                s32 c_original = histogram[i].count;
                if (c_original < 1)
                    continue;
                s32 c_dilated = dilated_counts[i];
                if (c_original < c_dilated)
                    histogram[i].count = 0;
            }
        }

        r32 mouse_t = ASCI_PI * ((0.5f+0.5f*input.mouse.x)*bins_t) / bins_t;
        r32 mouse_r = r_min + (r_max-r_min) * ((0.5f-0.5f*input.mouse.y)*bins_r) / bins_r;
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

        Ortho(0.0f, in_width, 0.0f, in_height);
        glLineWidth(4.0f);
        glBegin(GL_LINES);
        glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
        glVertex2f(x0, y0);
        glVertex2f(x1, y1);
        glEnd();
    });

    // Compute the suppression window dimensions in number of histogram cells
    s32 window_len_t = 0;
    s32 window_len_r = 0;
    {
        r32 bin_size_t = (t_max-t_min) / bins_t;
        r32 bin_size_r = (r_max-r_min) / bins_r;
        window_len_t = asci_round_positive(options.suppression_window_t / bin_size_t);
        window_len_r = asci_round_positive(options.suppression_window_r / bin_size_r);
        if (window_len_t % 2 != 0) window_len_t++;
        if (window_len_r % 2 != 0) window_len_r++;
        window_len_t = asci_clamp_s32(window_len_t, 0, bins_t-1);
        window_len_r = asci_clamp_s32(window_len_r, 0, bins_r-1);
    }

    s32 lines_found = 0;
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

        // Early-exit if the peak count was less than a threshold
        if (peak_count <= options.peak_exit_threshold*histogram_max_count)
        {
            break;
        }

        // @ Experimental
        const s32 MAX_SUPPORT_FEATURES_COUNT = 4096*8;
        static asci_Feature support_features[MAX_SUPPORT_FEATURES_COUNT];
        s32 support_features_count = 0;

        s32 peak_ti = peak_index % bins_t;
        s32 peak_ri = peak_index / bins_t;
        r32 peak_t = histogram[peak_index].avg_t;
        r32 peak_r = histogram[peak_index].avg_r;
        r32 peak_normal_x = cos(peak_t);
        r32 peak_normal_y = sin(peak_t);

        // Collect the votes in a window around the peak.
        static asci_Vote neighbor_votes[ASCI_MAX_VOTE_COUNT];
        s32 neighbor_count = 0;
        {
            r32 t0 = peak_t - options.suppression_window_t/2.0f;
            r32 t1 = peak_t + options.suppression_window_t/2.0f;
            r32 r0 = peak_r - options.suppression_window_r/2.0f;
            r32 r1 = peak_r + options.suppression_window_r/2.0f;
            bool right_edge_clip = false;
            if (t0 < 0.0f)
            {
                t0 += ASCI_PI;
            }
            if (t1 > ASCI_PI)
            {
                t1 -= ASCI_PI;
                right_edge_clip = true;
            }
            for (s32 i = 0; i < vote_count; i++)
            {
                asci_Vote vote = votes[i];

                // Since angles wrap _and_ the distance negates as a result,
                // we need to do a little dance to determine if the vote is
                // inside the selection window.
                bool within_t = false;
                bool within_r = false;
                if (t0 < t1)
                {
                    if (vote.t >= t0 && vote.t <= t1)
                    {
                        within_t = true;
                        if (vote.r >= r0 && vote.r <= r1)
                        {
                            within_r = true;
                        }
                    }
                }
                else if (right_edge_clip)
                {
                    if (vote.t <= t1)
                    {
                        within_t = true;
                        if (vote.r >= -r1 && vote.r <= -r0)
                        {
                            within_r = true;
                        }
                    }
                    else if (vote.t >= t0)
                    {
                        within_t = true;
                        if (vote.r >= r0 && vote.r <= r1)
                        {
                            within_r = true;
                        }
                    }
                }
                else
                {
                    if (vote.t <= t1)
                    {
                        within_t = true;
                        if (vote.r >= r0 && vote.r <= r1)
                        {
                            within_r = true;
                        }
                    }
                    else if (vote.t >= t0)
                    {
                        within_t = true;
                        if (vote.r >= -r1 && vote.r <= -r0)
                        {
                            within_r = true;
                        }
                    }
                }

                if (within_t && within_r)
                {
                    r32 d1 = peak_normal_x*vote.x1 + peak_normal_y*vote.y1 - peak_r;
                    r32 d2 = peak_normal_x*vote.x2 + peak_normal_y*vote.y2 - peak_r;
                    if (asci_max(asci_abs_r32(d1),asci_abs_r32(d2)) < 100.0f) // @ Better neighborhood collection.
                    {
                        neighbor_votes[neighbor_count] = vote;
                        neighbor_count++;
                    }

                    asci_Feature f1 = {0};
                    f1.x = vote.x1;
                    f1.y = vote.y1;
                    f1.gx = vote.gx1;
                    f1.gy = vote.gy1;
                    support_features[support_features_count++] = f1;

                    asci_Feature f2 = {0};
                    f2.x = vote.x2;
                    f2.y = vote.y2;
                    f2.gx = vote.gx2;
                    f2.gy = vote.gy2;
                    support_features[support_features_count++] = f2;
                }
            }
        }

        // @ Experimental
        #if 0
        {
            r32 x0, y0, x1, y1;
            r32 normal_x = cos(peak_t);
            r32 normal_y = sin(peak_t);
            r32 tangent_x = normal_y;
            r32 tangent_y = -normal_x;
            {
                if (asci_abs_r32(normal_y) > asci_abs_r32(normal_x))
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

            // Sort support features
            struct SortEntry
            {
                r32 key;
                s32 index;
            };
            SortEntry entries[MAX_SUPPORT_FEATURES_COUNT];
            for (s32 fi = 0; fi < support_features_count; fi++)
            {
                r32 l = tangent_x*(support_features[fi].x-x0)+
                        tangent_y*(support_features[fi].y-y0);
                entries[fi].key = l;
                entries[fi].index = fi;
            }
            qsort(entries, support_features_count, sizeof(SortEntry),
            [](const void *pa, const void *pb)
            {
                r32 dl = ((SortEntry*)pa)->key - ((SortEntry*)pb)->key;
                if (dl > 0.0f) return 1;
                else if (dl < 0.0f) return -1;
                else return 0;
            });

            GDB("test sorting", {
                Clear(0.0f, 0.0f, 0.0f, 1.0f);
                Ortho(0.0f, in_width, 0.0f, in_height);
                BlendMode();
                glPointSize(4.0f);
                glLineWidth(1.0f);
                static s32 visible_count = 1;
                glBegin(GL_POINTS);
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                for (s32 i = 0; i < visible_count; i++)
                {
                    s32 index = entries[i].index;
                    s32 x = support_features[index].x;
                    s32 y = support_features[index].y;
                    glVertex2f(x, y);
                }
                glEnd();

                glLineWidth(8.0f);
                glBegin(GL_LINES);
                glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
                glVertex2f(x0, y0);
                glVertex2f(x1, y1);
                glEnd();
                SliderInt("visible", &visible_count, 0, support_features_count-1);
                Text("%.2f", entries[visible_count].key);
            });
        }
        #endif

        // Fit lines to their voting neighborhood by minimizing square
        // vertical or horizontal error. This is implemented as one of
        // two linear least-squares programs, depending on the line
        // orientation.
        // @ Least-squares fitting
        #if 0
        r32 ls_t = peak_t;
        r32 ls_r = peak_r;
        {
            r32 A = 0.0f;
            r32 B = 0.0f;
            r32 C = 0.0f;
            r32 D = 0.0f;
            s32 N = 2*neighbor_count;
            r32 L = 1.0f/in_width;

            if (asci_abs_r32(peak_normal_x) > asci_abs_r32(peak_normal_y))
            {
                // "Vertical" line
                for (s32 i = 0; i < neighbor_count; i++)
                {
                    asci_Vote vote = neighbor_votes[i];
                    r32 x1 = vote.x1*L;
                    r32 y1 = vote.y1*L;
                    r32 x2 = vote.x2*L;
                    r32 y2 = vote.y2*L;
                    A += y1*y1 + y2*y2;
                    B += y1 + y2;
                    C += y1*x1 + y2*x2;
                    D += x1 + x2;
                }

                r32 b = (A*D-B*C)/(L*(A*N-B*B));
                r32 a = (C-b*B*L)/A;
                ls_t = atan(-a);
                ls_r = b*cos(ls_t);
            }
            else
            {
                // "Horizontal" line
                for (s32 i = 0; i < neighbor_count; i++)
                {
                    asci_Vote vote = neighbor_votes[i];
                    r32 x1 = vote.x1*L;
                    r32 y1 = vote.y1*L;
                    r32 x2 = vote.x2*L;
                    r32 y2 = vote.y2*L;
                    A += x1*x1 + x2*x2;
                    B += x1 + x2;
                    C += x1*y1 + x2*y2;
                    D += y1 + y2;
                }

                r32 b = (A*D-B*C)/(L*(A*N-B*B));
                r32 a = (C-b*B*L)/A;
                ls_t = atan(-1.0f/a);
                ls_r = b*sin(ls_t);
            }
        }
        r32 line_t = ls_t;
        r32 line_r = ls_r;
        #else
        r32 line_t = peak_t;
        r32 line_r = peak_r;
        #endif

        r32 normal_x = cos(line_t);
        r32 normal_y = sin(line_t);
        r32 tangent_x = normal_y;
        r32 tangent_y = -normal_x;

        // Compute terminal points for drawing the line
        r32 x0, y0, x1, y1;
        {
            if (asci_abs_r32(normal_y) > asci_abs_r32(normal_x))
            {
                x0 = 0.0f;
                x1 = in_width;
                y0 = (line_r-x0*normal_x)/normal_y;
                y1 = (line_r-x1*normal_x)/normal_y;
            }
            else
            {
                y0 = 0.0f;
                y1 = in_height;
                x0 = (line_r-y0*normal_y)/normal_x;
                x1 = (line_r-y1*normal_y)/normal_x;
            }
        }

        // Compute some error metrics on the line quality
        r32 normal_error_mean = 0.0f;
        r32 normal_error_std = 0.0f;
        if (neighbor_count > 0)
        {
            r32 mean_sum = 0.0f;
            r32 var_sum = 0.0f;
            for (s32 i = 0; i < neighbor_count; i++)
            {
                asci_Vote vote = neighbor_votes[i];
                r32 d1 = normal_x*vote.x1 + normal_y*vote.y1 - line_r;
                r32 d2 = normal_x*vote.x2 + normal_y*vote.y2 - line_r;
                r32 e1 = asci_abs_r32(d1);
                r32 e2 = asci_abs_r32(d2);
                mean_sum += e1 + e2;
                var_sum += e1*e1 + e2*e2; // @ Numerical stability
            }
            r32 N = (r32)(2*neighbor_count);
            normal_error_mean = mean_sum/N;
            normal_error_std = sqrt((var_sum/N) - normal_error_mean*normal_error_mean);
        }

        if (neighbor_count < 5) // @ Low count
            continue;

        // Reject line based on the error metric computed above
        if (normal_error_mean < options.normal_error_threshold &&
            normal_error_std < options.normal_error_std_threshold) // @ Altitude-based threshold
        {
            out_lines[lines_found].t = peak_t;
            out_lines[lines_found].r = peak_r;
            out_lines[lines_found].x_min = x0;
            out_lines[lines_found].y_min = y0;
            out_lines[lines_found].x_max = x1;
            out_lines[lines_found].y_max = y1;
            lines_found++;
        }

        GDB("hough histogram",
        {
            s32 mouse_ti = asci_round_positive((0.5f+0.5f*input.mouse.x)*bins_t);
            s32 mouse_ri = asci_round_positive((0.5f-0.5f*input.mouse.y)*bins_r);

            Ortho(t_min, t_max, r_min, r_max);
            BlendMode();
            Clear(0.0f, 0.0f, 0.0f, 1.0f);
            glPointSize(6.0f);
            glBegin(GL_POINTS);
            {
                for (s32 ri = 0; ri < bins_r; ri++)
                for (s32 ti = 0; ti < bins_t; ti++)
                {
                    // r32 r = r_min + (r_max-r_min)*ri/bins_r;
                    // r32 t = t_min + (t_max-t_min)*ti/bins_t;
                    r32 r = histogram[ti + ri*bins_t].avg_r;
                    r32 t = histogram[ti + ri*bins_t].avg_t;
                    s32 count = histogram[ti + ri*bins_t].count;

                    if (mouse_ti == ti && mouse_ri == ri)
                    {
                        glColor4f(0.4f, 1.0f, 0.4f, 1.0f);
                        SetTooltip("%d %d %d", ti, ri, count);
                    }
                    else
                    {
                        ColorRamp(count / (0.2f*histogram_max_count));
                    }
                    glVertex2f(t, r);
                }
            }
            glEnd();

            glPointSize(6.0f);
            glBegin(GL_POINTS);
            glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
            {
                s32 ti0 = peak_ti - window_len_t/2;
                s32 ti1 = peak_ti + window_len_t/2;
                s32 ri0 = peak_ri - window_len_r/2;
                s32 ri1 = peak_ri + window_len_r/2;
                for (s32 ti = ti0; ti <= ti1; ti++)
                for (s32 ri = ri0; ri <= ri1; ri++)
                {
                    s32 write_t = 0;
                    s32 write_r = 0;
                    if (ti < 0)
                    {
                        write_t = asci_clamp_s32(ti+bins_t, 0, bins_t-1);
                        write_r = asci_clamp_s32(bins_r-1-ri, 0, bins_r-1);
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
                    r32 r = r_min + (r_max-r_min)*write_r/bins_r;
                    r32 t = t_min + (t_max-t_min)*write_t/bins_t;
                    glVertex2f(t, r);
                }
            }
            glEnd();

            glPointSize(14.0f);
            glBegin(GL_POINTS);
            glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
            r32 r = r_min + (r_max-r_min)*peak_ri/bins_r;
            r32 t = t_min + (t_max-t_min)*peak_ti/bins_t;
            glVertex2f(t, r);
            glEnd();
        });

        #ifdef USE_GDB
        GLuint texture = 0;
        #endif
        GDB_SKIP("line estimate",
        {
            // if (!texture)
            //     texture = MakeTexture2D(in_rgb, in_width, in_height, GL_RGB);
            // BlendMode(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            // Clear(0.0f, 0.0f, 0.0f, 1.0f);
            // Ortho(-1.0f, +1.0f, -1.0f, +1.0f);
            // DrawTexture(texture, 0.5f, 0.5f, 0.5f);
            Ortho(0.0f, in_width, 0.0f, in_height);

            glPointSize(2.0f);
            Clear(0.0f, 0.0f, 0.0f, 1.0f);
            BlendMode();
            glBegin(GL_POINTS);
            {
                for (int i = 0; i < feature_count; i++)
                {
                    r32 x = (r32)features[i].x;
                    r32 y = (r32)features[i].y;

                    glColor4f(0.3f, 0.3f, 0.3f, 1.0f);
                    glVertex2f(x, y);
                }
            }
            glEnd();

            glLineWidth(5.0f);
            glBegin(GL_LINES);
            if (normal_error_mean < options.normal_error_threshold &&
                normal_error_std < options.normal_error_std_threshold)
                glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
            else
                glColor4f(0.2f, 0.5f, 1.0f, 1.0f);
            glVertex2f(x0, y0);
            glVertex2f(x1, y1);
            glEnd();

            BlendMode(GL_ONE, GL_ONE);
            glPointSize(8.0f);
            glBegin(GL_POINTS);
            glColor4f(0.7f*0.3f, 0.7f*0.5f, 0.7f*0.8f, 1.0f);
            for (s32 i = 0; i < neighbor_count; i++)
            {
                asci_Vote vote = neighbor_votes[i];
                glVertex2f(vote.x1, vote.y1);
                glVertex2f(vote.x2, vote.y2);
            }
            glEnd();
            Text("mean: %.2f\nvar: %.2f\ncount: %d",
                 normal_error_mean, normal_error_std, neighbor_count);
        });

        // Zero the histogram count of votes inside the suppression window
        {
            s32 ti0 = peak_ti - window_len_t/2;
            s32 ti1 = peak_ti + window_len_t/2;
            s32 ri0 = peak_ri - window_len_r/2;
            s32 ri1 = peak_ri + window_len_r/2;
            for (s32 ti = ti0; ti <= ti1; ti++)
            for (s32 ri = ri0; ri <= ri1; ri++)
            {
                s32 write_t = 0;
                s32 write_r = 0;
                if (ti < 0)
                {
                    write_t = asci_clamp_s32(ti+bins_t, 0, bins_t-1);
                    write_r = asci_clamp_s32(bins_r-1-ri, 0, bins_r-1);
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
    }

    GDB("final lines",
    {
        Ortho(0.0f, in_width, 0.0f, in_height);
        glPointSize(2.0f);
        Clear(0.0f, 0.0f, 0.0f, 1.0f);
        BlendMode();
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
    });

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

// @ Altitude-based threshold:
//   Rejection based on squared normal distance alone will not be robust
//   against changing the camera height, since thick lines will give a large
//   response. Need to either have altitude-based threshold, or use a different
//   error metric. For example, also consider normal distance variance.

// @ Least-square fitting
//   Currently disabled because it works badly in the presence of outliers.
//   Want to atleast partially prune lines before performing minimzation.
//   Want to do some sort of prepass where we look at gaps and outliers...?
//   Prepass RANSAC on x0,y0,x1,y1?
//     Count inliers, outliers (positive detections outside line, and negative
//     detections inside line... somehow?)

// @ Better neighborhood collection
//   We have a tradeoff between rejecting votes that have large spatial
//   seperation from the initial guess, and keeping them to determine
//   the fitness of the line in the end. Jfr. image video0042.
//   This might be fixed by using error metrics for rejection, such as
//   variance _and_ average normal error.

// @ SIMD Sobel
//   Compute average? Compute sum?
//   Skip pushing entire block if almost all less than threshold

// @ Gradient rejection threshold
//   Too high?

// @ Low count
//   Do we also want to reject lines with low neighborhood count
