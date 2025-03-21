#!/usr/bin/env python3

import argparse
import subprocess

import cv2 as cv
import numpy as np

def fine_tune_warp(foreground, background, M, original_foreground_w, original_foreground_h, interpolation, points_type, confidence, use_left_edge, region):
    # Compute corners from old matrix.
    foreground_corners = points_type([
        [[0,                     0]],
        [[original_foreground_w, 0]],
        [[0,                     original_foreground_h]],
        [[original_foreground_w, original_foreground_h]],
    ])
    background_corners = cv.perspectiveTransform(foreground_corners, M)

    # Get left/right edge of foreground image.
    edge_width = int(0.05 * len(foreground[0]))
    if use_left_edge:
        foreground_edge = foreground[:, :edge_width]
    else:
        foreground_edge = foreground[:, -edge_width:]

    # Focus on a specific part of foreground image.
    foreground_edge_critical = foreground_edge[region[0][0]:region[0][1], :]
    if len(region) > 1:
        foreground_edge_critical = np.append(foreground_edge_critical, foreground_edge[region[1][0]:region[1][1], :], 0)

    best_corr = 0.0
    best_error = None
    best_M = None

    max_fine_tuning_pixels = 10
    fine_tuning_step = 1

    for fine_tuning_error_top in range(-max_fine_tuning_pixels, max_fine_tuning_pixels+1, fine_tuning_step):
        fine_tuning_error_bottom_range = [fine_tuning_error_top]
        if len(region) > 1:
            fine_tuning_error_bottom_range = range(-max_fine_tuning_pixels, max_fine_tuning_pixels+1, fine_tuning_step)
        for fine_tuning_error_bottom in fine_tuning_error_bottom_range:
            # Apply error correction to left or right side's Y coord.
            if use_left_edge:
                fine_tuned_foreground_corners = points_type([
                    [[0,                     fine_tuning_error_top]],
                    [[original_foreground_w, 0]],
                    [[0,                     original_foreground_h+fine_tuning_error_bottom]],
                    [[original_foreground_w, original_foreground_h]],
                ])
            else:
                fine_tuned_foreground_corners = points_type([
                    [[0,                     0]],
                    [[original_foreground_w, fine_tuning_error_top]],
                    [[0,                     original_foreground_h]],
                    [[original_foreground_w, original_foreground_h+fine_tuning_error_bottom]],
                ])

            # Calculate new matrix
            new_M, mask = cv.findHomography(fine_tuned_foreground_corners, background_corners, 0, confidence=confidence)

            # Get left/right edge of background image.
            background_scaled_to_foreground = cv.warpPerspective(background, new_M, (original_foreground_w,original_foreground_h), flags=interpolation|cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_REPLICATE)
            if use_left_edge:
                background_edge = background_scaled_to_foreground[:, :edge_width]
            else:
                background_edge = background_scaled_to_foreground[:, -edge_width:]

            # Focus on a specific part of foreground image.
            background_edge_critical = background_edge[region[0][0]:region[0][1], :]
            if len(region) > 1:
                background_edge_critical = np.append(background_edge_critical, background_edge[region[1][0]:region[1][1], :], 0)

            # Only enable for debugging the crop operation
            if False:
                is_success, im_buf_arr = cv.imencode('.png', foreground_edge_critical)
                if not is_success:
                    raise Exception('cv.imencode failure')
                im_buf_arr.tofile('foreground_edge.png')
                is_success, im_buf_arr = cv.imencode('.png', background_edge_critical)
                if not is_success:
                    raise Exception('cv.imencode failure')
                im_buf_arr.tofile('background_edge.png')

            # Compute correlation.
            # https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
            correlation = cv.matchTemplate(background_edge_critical,foreground_edge_critical,cv.TM_CCOEFF_NORMED)
            min_corr, max_corr, min_corr_loc, max_corr_loc = cv.minMaxLoc(correlation)

            if max_corr > best_corr:
                best_corr, best_error, best_M = max_corr, (fine_tuning_error_top, fine_tuning_error_bottom), new_M

    print(f'best error {"left" if use_left_edge else "right"}: {best_error}')

    return best_M

def warp_foreground_to_background(foreground, background, replace_foreground=None, background_scale=1.0, feature_type='orb', max_features=500, cross_check=True, subset_frac=0.2, confidence=0.995, transform_mode='homography', interpolation='linear', fine_tune_left_region=None, fine_tune_right_region=None):
    if replace_foreground is None:
        replace_foreground = foreground

    # Feature detection algorithms use grayscale input.
    foreground_grayscale = cv.cvtColor(foreground, cv.COLOR_BGR2GRAY)
    background_grayscale = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

    # Detect features.
    # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    if feature_type == 'orb':
        detector = cv.ORB_create(max_features)
    elif feature_type == 'sift':
        detector = cv.SIFT_create(max_features)
    elif feature_type == 'brisk':
        detector = cv.BRISK_create()
    elif feature_type == 'akaze':
        detector = cv.AKAZE_create()
    else:
        raise Exception(f'Unsupported feature type {feature_type}')
    foreground_keypoints, foreground_descriptors = detector.detectAndCompute(foreground_grayscale, None)
    background_keypoints, background_descriptors = detector.detectAndCompute(background_grayscale, None)
    print(f'Feature count: foreground {len(foreground_keypoints)}, background {len(background_keypoints)}')

    # Match feature descriptors.
    # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    if feature_type in ('orb', 'brisk', 'akaze'):
        distance_type = cv.NORM_HAMMING
    elif feature_type == 'sift':
        distance_type = cv.NORM_L2
    else:
        raise Exception(f'Unsupported feature type {feature_type}')
    matcher = cv.BFMatcher(distance_type, crossCheck=cross_check)
    matches = matcher.match(foreground_descriptors, background_descriptors, None)
    print(f'Feature match count: {len(matches)}')

    # Sort matches in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Take best subset of matches.
    subset_num = int(subset_frac * len(matches))
    matches = matches[:subset_num]

    # Draw the matches.
    matches_image = cv.drawMatches(foreground, foreground_keypoints, background, background_keypoints, matches, None)

    # Extract points.
    # https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
    points_type = np.float32 if transform_mode in ('homography', 'affine3d') else np.int64
    points_len = 3 if transform_mode == 'affine3d' else 2
    foreground_points = points_type([ foreground_keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,1,points_len)
    background_points = points_type([ background_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,points_len)

    # Handle foreground replacement.
    # We assume that the replacement has the same aspect ratio as the original.
    replace_foreground_scale = (foreground.shape[0] * foreground.shape[1] / (replace_foreground.shape[0] * replace_foreground.shape[1]))**0.5
    original_foreground, foreground = foreground, replace_foreground

    # Fill in RGB data for fully transparent pixels, otherwise the bilinear filtering will produce halos.
    # This is unfortunately rather slow. Maybe there's a faster way?
    # Ideally we should also traverse columns, not just rows. Maybe later.
    # Disabled for now since we use magick for user-facing images.
    #for row in foreground:
    if False:
        valid_rgb = None
        for pixel in row:
            if pixel[3] > 0:
                valid_rgb = pixel[0:3]
            elif valid_rgb is not None:
                pixel[0:3] = valid_rgb
        valid_rgb = None
        for pixel in row[::-1]:
            if pixel[3] > 0:
                valid_rgb = pixel[0:3]
            elif valid_rgb is not None:
                pixel[0:3] = valid_rgb

    # Handle alpha separately to avoid border artifacts.
    # Warp the alpha with a 1px border set to transparent.
    foreground_alpha_border = foreground[1:-1, 1:-1]
    foreground_alpha_border = cv.copyMakeBorder(foreground_alpha_border, top=1, bottom=1, left=1, right=1, borderType=cv.BORDER_CONSTANT, value=(0,0,0,0))

    # Get foreground dimensions.
    original_foreground_h,original_foreground_w = original_foreground.shape[:2]

    # Handle background scaling.
    h,w = background.shape[:2]
    h,w = (h * background_scale, w * background_scale)
    h,w = (int(h), int(w))

    # Handle interpolation mode.
    if interpolation == 'nearest':
        interpolation = cv.INTER_NEAREST
    elif interpolation == 'linear':
        interpolation = cv.INTER_LINEAR
    elif interpolation == 'cubic':
        interpolation = cv.INTER_CUBIC
    elif interpolation == 'lanczos':
        interpolation = cv.INTER_LANCZOS4
    else:
        raise Exception(f'Unsupported interpolation mode {interpolation}')

    if transform_mode == 'homography':
        # Find transformation matrix.
        M, mask = cv.findHomography(foreground_points, background_points, cv.RANSAC, confidence=confidence)

        # Fine-tune based on left and right edge of foreground.
        if fine_tune_left_region is not None:
            M = fine_tune_warp(foreground=original_foreground, background=background, M=M, original_foreground_w=original_foreground_w, original_foreground_h=original_foreground_h, interpolation=interpolation, points_type=points_type, confidence=confidence, use_left_edge=True, region=fine_tune_left_region)
        if fine_tune_right_region is not None:
            M = fine_tune_warp(foreground=original_foreground, background=background, M=M, original_foreground_w=original_foreground_w, original_foreground_h=original_foreground_h, interpolation=interpolation, points_type=points_type, confidence=confidence, use_left_edge=False, region=fine_tune_right_region)

        # Scale output
        M[0][0] *= background_scale * replace_foreground_scale
        M[0][1] *= background_scale * replace_foreground_scale
        M[0][2] *= background_scale
        M[1][0] *= background_scale * replace_foreground_scale
        M[1][1] *= background_scale * replace_foreground_scale
        M[1][2] *= background_scale
        M[2][0] *= replace_foreground_scale
        M[2][1] *= replace_foreground_scale

        # Transform foreground to fit background.
        try:
            foreground_rgb = cv.warpPerspective(foreground, M, (w,h), flags=interpolation, borderMode=cv.BORDER_REPLICATE)
            foreground_alpha = cv.warpPerspective(foreground_alpha_border, M, (w,h), flags=interpolation, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0,0))
        except cv.error:
            foreground_rgb, foreground_alpha = None, None
            print('Size above OpenCV warp limit, skipping OpenCV warp')
    elif transform_mode == 'affine3d':
        # TODO: Affine3D mode isn't properly tested yet.

        # Find transformation matrix.
        _, M, inliers = cv.estimateAffine3D(foreground_points, background_points, confidence=confidence)

        # Scale output
        M[0][0] *= background_scale * replace_foreground_scale
        M[0][1] *= background_scale * replace_foreground_scale
        M[0][2] *= background_scale
        M[1][0] *= background_scale * replace_foreground_scale
        M[1][1] *= background_scale * replace_foreground_scale
        M[1][2] *= background_scale

        # Transform foreground to fit background.
        try:
            foreground_rgb = cv.warpAffine(foreground, M, (w,h), flags=interpolation, borderMode=cv.BORDER_REPLICATE)
            foreground_alpha = cv.warpAffine(foreground_alpha_border, M, (w,h), flags=interpolation, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0,0))
        except cv.error:
            foreground_rgb, foreground_alpha = None, None
            print('Size above OpenCV warp limit, skipping OpenCV warp')
    elif transform_mode in ('affine2d', 'affinepartial2d'):
        # Find transformation matrix.
        if transform_mode == 'affine2d':
            M, inliers = cv.estimateAffine2D(foreground_points, background_points, confidence=confidence)
        elif transform_mode == 'affinepartial2d':
            M, inliers = cv.estimateAffinePartial2D(foreground_points, background_points, confidence=confidence)
        else:
            raise Exception('Mode not supported')

        # Scale output
        M[0][0] *= background_scale * replace_foreground_scale
        M[0][1] *= background_scale * replace_foreground_scale
        M[0][2] *= background_scale
        M[1][0] *= background_scale * replace_foreground_scale
        M[1][1] *= background_scale * replace_foreground_scale
        M[1][2] *= background_scale

        # Transform foreground to fit background.
        try:
            foreground_rgb = cv.warpAffine(foreground, M, (w,h), flags=interpolation, borderMode=cv.BORDER_REPLICATE)
            foreground_alpha = cv.warpAffine(foreground_alpha_border, M, (w,h), flags=interpolation, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0,0))
        except cv.error:
            foreground_rgb, foreground_alpha = None, None
            print('Size above OpenCV warp limit, skipping OpenCV warp')
    else:
        raise Exception('Mode not supported')

    if foreground_rgb is not None:
        foreground_b, foreground_g, foreground_r, _ = cv.split(foreground_rgb)
        _, _, _, foreground_alpha = cv.split(foreground_alpha)
        foreground = cv.merge((foreground_b, foreground_g, foreground_r, foreground_alpha))

        return foreground, M, matches_image
    else:
        return None, M, matches_image

def main():
    parser = argparse.ArgumentParser(prog='subimage-stitcher')

    parser.add_argument('--foreground', required=True,
        help='path to input foreground (will be stitched to background)')
    parser.add_argument('--background', required=True,
        help='path to input background')
    parser.add_argument('--replace-foreground', required=False, default=None,
        help='path to replacement input foreground (can be different resolution from foreground, but must be same aspect ratio)')
    parser.add_argument('--replace-background', required=False, default=None,
        help='path to replacement input background (can be different resolution from background, but must be same aspect ratio)')
    parser.add_argument('--out-warped', required=False, default='warped.png',
        help='path to output warped image')
    parser.add_argument('--out-composite', required=False, default='composite.png',
        help='path to output composite image')
    parser.add_argument('--out-background', required=False, default='background.png',
        help='path to output background image')
    parser.add_argument('--out-stacked', required=False, default='stacked.png',
        help='path to output horizontally stacked image')
    parser.add_argument('--out-keypoints', required=False, default='keypoints.png',
        help='path to output keypoints image')
    parser.add_argument('--feature-type', default='orb',
        choices=['orb', 'sift', 'brisk', 'akaze'], help='Type of features to detect')
    parser.add_argument('--max-features', type=int, default=500,
        help='Number of features to try detecting')
    parser.add_argument('--no-cross-check', action='store_false',
        dest='cross_check', help='Disable cross-check when matching features')
    parser.add_argument('--subset-frac', type=float, default=0.2,
        help='Fraction of detected feature matches to use')
    parser.add_argument('--confidence', type=float, default=0.995,
        help='Confidence level to aim for')
    parser.add_argument('--transform-mode', default='homography',
        choices=['homography', 'affine3d', 'affine2d', 'affinepartial2d'], help='Transform mode')
    parser.add_argument('--fine-tune-left-region', required=False, nargs='*',
        type=str, help='Fine-tuning critical region (left edge) (enter two comma-separated Y coordinates in foreground scale, can be used up to twice)')
    parser.add_argument('--fine-tune-right-region', required=False, nargs='*',
        type=str, help='Fine-tuning critical region (right edge) (enter two comma-separated Y coordinates in foreground scale, can be used up to twice)')
    parser.add_argument('--interpolation', default='linear',
        choices=['nearest', 'linear', 'cubic', 'lanczos'], help='Interpolation mode')
    parser.add_argument('--scale', type=float, default=1.0,
        help='Scale factor (relative to background)')
    parser.add_argument('--blend', type=int, default=5,
        help='Blend radius')
    args = parser.parse_args()

    if args.replace_foreground is None:
        args.replace_foreground = args.foreground
    if args.replace_background is None:
        args.replace_background = args.background

    if args.fine_tune_left_region is not None:
        for n in range(len(args.fine_tune_left_region)):
            args.fine_tune_left_region[n] = args.fine_tune_left_region[n].split(',')
            args.fine_tune_left_region[n] = list([int(v) for v in args.fine_tune_left_region[n]])
    if args.fine_tune_right_region is not None:
        for n in range(len(args.fine_tune_right_region)):
            args.fine_tune_right_region[n] = args.fine_tune_right_region[n].split(',')
            args.fine_tune_right_region[n] = list([int(v) for v in args.fine_tune_right_region[n]])

    # Read input images.
    # We use imdecode instead of imread to work around Unicode breakage on Windows.
    # https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/
    foreground = cv.imdecode(np.fromfile(args.foreground, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    background = cv.imdecode(np.fromfile(args.background, dtype=np.uint8), cv.IMREAD_UNCHANGED)

    # Convert from grayscale to color.
    if len(foreground.shape) < 3:
        foreground = cv.cvtColor(foreground, cv.COLOR_GRAY2BGR)
    if len(background.shape) < 3:
        background = cv.cvtColor(background, cv.COLOR_GRAY2BGR)

    # Add alpha channel.
    if len(foreground.shape) == 3 and foreground.shape[2] == 3:
        foreground = cv.cvtColor(foreground, cv.COLOR_RGB2RGBA)
    if len(background.shape) == 3 and background.shape[2] == 3:
        background = cv.cvtColor(background, cv.COLOR_RGB2RGBA)

    # Do the same for replacement images.
    if args.replace_foreground == args.foreground:
        replace_foreground = foreground
    else:
        replace_foreground = cv.imdecode(np.fromfile(args.replace_foreground, dtype=np.uint8), cv.IMREAD_UNCHANGED)
        if len(replace_foreground.shape) < 3:
            replace_foreground = cv.cvtColor(replace_foreground, cv.COLOR_GRAY2BGR)
        if len(replace_foreground.shape) == 3 and replace_foreground.shape[2] == 3:
            replace_foreground = cv.cvtColor(replace_foreground, cv.COLOR_RGB2RGBA)
    if args.replace_background == args.background:
        replace_background = background
    else:
        replace_background = cv.imdecode(np.fromfile(args.replace_background, dtype=np.uint8), cv.IMREAD_UNCHANGED)
        if len(replace_background.shape) < 3:
            replace_background = cv.cvtColor(replace_background, cv.COLOR_GRAY2BGR)
        if len(replace_background.shape) == 3 and replace_background.shape[2] == 3:
            replace_background = cv.cvtColor(replace_background, cv.COLOR_RGB2RGBA)

    # Transform foreground, generate keypoints debug output.
    warped, M, keypoints = warp_foreground_to_background(foreground, background, replace_foreground=replace_foreground, feature_type=args.feature_type, max_features=args.max_features, cross_check=args.cross_check, subset_frac=args.subset_frac, confidence=args.confidence, transform_mode=args.transform_mode, interpolation=args.interpolation, background_scale=args.scale, fine_tune_left_region=args.fine_tune_left_region, fine_tune_right_region=args.fine_tune_right_region)

    # Detect scale factor of matrix, to assist in upscaling.
    scale_matrix = np.array(M[:2, :2])
    # Area scale factor will be negative if there's a reflection.
    area_scale_factor = np.linalg.det(scale_matrix)
    linear_scale_factor = abs(area_scale_factor) ** 0.5
    print(f'Linear scale factor: {linear_scale_factor}')

    # Scale background.
    background = cv.resize(replace_background, (int(background.shape[1]*args.scale), int(background.shape[0]*args.scale)), interpolation = cv.INTER_LANCZOS4)

    # Write results.
    # We use imencode instead of imwrite to work around Unicode breakage on Windows.
    # https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/

    is_success, im_buf_arr = cv.imencode('.png', background)
    if not is_success:
        raise Exception('cv.imencode failure')
    im_buf_arr.tofile(args.out_background)

    is_success, im_buf_arr = cv.imencode('.png', keypoints)
    if not is_success:
        raise Exception('cv.imencode failure')
    im_buf_arr.tofile(args.out_keypoints)

    # Generate stacked debug output.
    # warped might be None due to https://github.com/opencv/opencv/issues/7544
    # If so, we only use magick for warping.
    if warped is not None:
        stacked = np.hstack([warped, background])

        # Write results.
        # We use imencode instead of imwrite to work around Unicode breakage on Windows.
        # https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/

        is_success, im_buf_arr = cv.imencode('.png', stacked)
        if not is_success:
            raise Exception('cv.imencode failure')
        im_buf_arr.tofile(args.out_stacked)

        is_success, im_buf_arr = cv.imencode('.png', warped)
        if not is_success:
            raise Exception('cv.imencode failure')
        im_buf_arr.tofile(args.out_warped)

    # Do homography via magick, because it handles alpha better.
    if args.transform_mode in ('affine2d', 'affinepartial2d'):
        M = [ M[0], M[1], [0.0, 0.0, 1.0] ]
    subprocess.run(['magick', args.replace_foreground, '-background', 'transparent', '-extent', f'{max(int(background.shape[1]), int(replace_foreground.shape[1]))}x{max(int(background.shape[0]), int(replace_foreground.shape[0]))}', '-virtual-pixel', 'transparent', '-distort', 'Perspective-Projection', f'{M[0][0]}, {M[0][1]}, {M[0][2]} {M[1][0]}, {M[1][1]}, {M[1][2]} {M[2][0]}, {M[2][1]}', '-extent', f'{int(background.shape[1])}x{int(background.shape[0])}', args.out_warped + '.magick.png'], check=True)

    # Generate composite image via magick, because it handles alpha better.
    if warped is not None:
        subprocess.run(['magick', args.out_background, args.out_warped, '-composite', args.out_composite], check=True)
    subprocess.run(['magick', args.out_background, args.out_warped + '.magick.png', '-composite', args.out_composite + '.magick.png'], check=True)
    subprocess.run(['magick', args.out_background, args.out_warped + '.magick.png', '-channel', 'Alpha', '-negate', '-morphology', 'Dilate', f'Diamond:{args.blend*2}', '-negate', '-blur', f'{args.blend*2}x{args.blend}', '-channel', 'All', '-composite', args.out_composite + '.magick-blend.png'], check=True)

if __name__ == '__main__':
    main()
