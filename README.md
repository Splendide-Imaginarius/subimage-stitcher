# Subimage Stitcher

Do you have two images, one of which is a zoomed-in version of the other? Would you like to stitch the zoomed-in image into the zoomed-out image, to boost the resolution of that portion? Subimage Stitcher is designed to do this.

## Comparison to Other Projects

Most other image stitching tools are designed for panoramas, where each input image has a small overlap with its adjacent input images, and each input image has similar scale. They produce an output image whose dimensions and viewpoint are zoomed-out relative to the input images.

Subimage Stitcher is different. It's designed for cases where the input images have large (maybe 100%) overlap, and very different scale. It produces an output image whose dimensions and viewpoint are identical to the outer input image, but with enhanced detail in the portion that corresponds to the inner input image.

Use the right tool for the job!

## Installation

* Install ImageMagick if you don't have it already.
* Clone this Git repo.
* `pip install --user .`

## Usage

```
subimage-stitcher --foreground "inputs/foreground.png" --background "inputs/background.png" --replace-foreground "inputs/replacement-foreground.png" --replace-background "inputs/replacement-background.png" --scale 1.0 --feature-type "sift" --max-features "50000" --subset-frac "0.2" --confidence "0.998" --transform-mode "homography"
```

* The arguments to `--foreground` and `--background` are used for computing the transformation matrix.
* The arguments to `--replace-foreground` and `--replace-background`, if present, are used in composing the output image. If they are not specified, the `--foreground` and `--background` images are used.
* The output image has the dimensions of the `--background` argument, multiplied by the `--scale` argument.
* Three output composite images are produced: one with OpenCV interpolation, one with ImageMagick interpolation, and one with ImageMagick interpolation combined with a smooth blending of the foreground and the background.
* An output warped image is produced; this is the foreground image in the same pose as the composite image, but without the background.
* A keypoints map image is produced for debug purposes; this can be helpful for visualizing how well the feature detection/matching is working.
* An output background image is produced; this is the same as the composite image but without the foreground. (It's identical to the input background image, just potentially at a different scale.)
* An output stacked image is produced, if you want to look at the warped foreground and background images side-by-side.
* If you're seeing bad warping results, the `--feature-type`, `--subset-frac`, and `--no-cross-check` arguments are the most likely to help.
* The ImageMagick composite images are likely to look better to humans, but the OpenCV composite image is likely to work better as input to a subsequent Subimage Stitcher feature-detection pass. You can use the `--replace-foreground` or `--replace-background` arguments to get the best of both worlds.
* Running the input images through Real-ESRGAN first seems to improve the reliabilty of Subimage Stitcher's feature detection.
* If you've already followed the above tips, but one or both of the seams are still off by just a few pixels, you can use the `--fine-tune-left-region` and `--fine-tune-right-region` arguments, like this: `--fine-tune-left-region 500,550 --fine-tune-right-region 400,450` (this will fine-tune the stitching matrix to prioritize the alignment of pixels 500 through 550 of the left seam, and pixels 400 through 450 of the right seam). Pixels are in foreground coordinates. Fine-tuning works by translating the Y coordinates of the corners of the perspective transformation. Typically, you would prioritize a high-contrast edge or a piece of geometry that naturally attracts the eye.

Run `subimage-stitcher --help` to see full command-line options.

## Credits

Copyright 2023 Splendide Imaginarius.

This is not a license requirement, but if you use Subimage Stitcher to process images that you distribute, it would be greatly appreciated if you credit me. Example credits: "This image was processed with Subimage Stitcher by Splendide Imaginarius." Linking back to this Git repository would also be greatly appreciated.

Subimage Stitcher is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Subimage Stitcher is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Subimage Stitcher. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).
