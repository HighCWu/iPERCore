# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import cv2
import numpy as np
import paddle
import paddle.vision.transforms.functional as TF


class ImageTransformer(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is matched to output_size.
                            If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        images = sample["images"]
        resized_images = []

        for image in images:
            height, width = image.shape[0:2]

            if height != self.output_size or width != self.output_size:
                image = cv2.resize(image, (self.output_size, self.output_size))

            image = image.astype(np.float32)
            image /= 255.0
            image = image * 2 - 1

            image = np.transpose(image, (2, 0, 1))

            resized_images.append(image)

        resized_images = np.stack(resized_images, axis=0)

        sample["images"] = resized_images
        return sample


class ImageNormalizeToTensor(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __call__(self, image):
        if not (TF._is_pil_image(image) or TF._is_numpy_image(image)):
            raise TypeError(
                'pic should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.
                format(type(image)))
        if TF._is_pil_image(image):
            image = np.asarray(image)
        if image.ndim == 2:
            image = image[...,None]
        if image.dtype == np.uint8:
            image = image / 255.0
        h, w, c = image.shape
        if h > c and w > c:
            image = np.transpose(image, (2,0,1))
        image = image * 2 - 1

        return image.astype(np.float32)


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors. 
    """

    def __call__(self, sample):
        sample["images"] = np.asarray(sample["images"]).astype(np.float32)
        sample["smpls"] = np.asarray(sample["smpls"]).astype(np.float32)

        if "masks" in sample:
            sample["masks"] = np.asarray(sample["masks"]).astype(np.float32)

        return sample

