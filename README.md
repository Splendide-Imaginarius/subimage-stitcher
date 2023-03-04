# Subimage Stitcher

Do you have two images, one of which is a zoomed-in version of the other? Would you like to stitch the zoomed-in image into the zoomed-out image, to boost the resolution of that portion? Subimage Stitcher is designed to do this.

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
* If you're seeing bad matrix results, the `--feature-type`, `--subset-frac`, and `--no-cross-check` arguments are the most likely to help.

Run `subimage-stitcher --help` to see full command-line options.

## Credits

Copyright 2023 Splendide Imaginarius.

This is not a license requirement, but if you use Subimage Stitcher to process images that you distribute, it would be greatly appreciated if you credit me. Example credits: "This image was processed with Subimage Stitcher by Splendide Imaginarius." Linking back to this Git repository would also be greatly appreciated.

Subimage Stitcher is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Subimage Stitcher is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Subimage Stitcher. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).
