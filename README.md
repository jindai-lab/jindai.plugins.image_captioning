# jindai.plugins.image_captioning
Image captioning plugin, adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.

## Installation

- Check out `sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning` and place `__init__.py` alongside with its codes.
- Put downloaded model `BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar` and `WORDMAP_coco_5_cap_per_img_5_min_word_freq.json` in a new subfolder called `checkpoint`.
- Place all files in `plugins/image_captioning`.

Then you can use `ImageCaptioning` pipeline stage in Jindai.

## Prerequisite

- `cv2-python` is required.

## License

MIT License
