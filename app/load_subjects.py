# %%
import time
import numpy as np
import pandas as pd

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, getTestCase

from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.filters import maximum_filter

from tqdm.auto import tqdm

from pathlib import Path

from toolbox import count_limit_img_contour
from onstart import CONFIG, logger

# %%
# Find all files
folder = CONFIG['subjects_folder']

if not folder.is_dir():
    logger.error('Subjects folder not found.')

subject_folders = dict()
for e in folder.iterdir():
    # Ingore all the files
    if e.is_file():
        continue

    dcms = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(e.as_posix())
    if dcms:
        subject_folders[e.name] = (e, len(dcms), dcms)

subject_folders

logger.debug('Found subjects of {}'.format(
    ['{}: {} dcm files'.format(e, subject_folders[e][1]) for e in subject_folders]))

# %%


class Subject_Manager(object):
    def __init__(self, subject_folders):
        # The subject_folders has solid format,
        # see the subject_folders above.
        self.subject_folders = subject_folders
        self.subjects = [e for e in subject_folders]
        logger.debug('Subject Manager initialized: {}'.format(self.subjects))

    def _check_subject(self, subject):
        # Check the existence of the subject
        if not subject in self.subjects:
            logger.error('Can not access the subject {}'.format(subject))
            return None
        return subject

    def get_array(self, subject):
        # Get the array of the dcm files,
        # the examples have the shape of slices x 512 x 512
        self._check_subject(subject)

        dcms = self.subject_folders[subject][2]
        img_list = [sitk.GetArrayFromImage(sitk.ReadImage(e))
                    for e in tqdm(dcms, 'Reading .dcm files in {}'.format(subject))]
        img_array = np.concatenate(img_list, axis=0)

        logger.debug('Got array for subject: {}, the shape is {}'.format(
            subject, img_array.shape
        ))

        return img_array

    def compute_contour(self, img_array, threshold=None, shrink=False, kernel=np.ones((2, 5, 5)), footprint=np.ones((5, 5, 5))):
        # Mark the region of interest
        # Remove the skull for calculation,
        # using the maximum_filter method.
        img_contour = img_array.copy()

        if threshold is None:
            threshold = 50
            logger.debug(
                'The threshold is not provided, using {} for default value.'.format(threshold))

        mask = maximum_filter(img_array, footprint=footprint)
        img_contour[mask > 200] = 0

        mask = maximum_filter(img_array, footprint=np.ones((10, 10, 10)))
        img_contour[mask > 500] = 0

        # Remove the **small** nodes for better solution.
        mask = img_contour > threshold
        mask = binary_erosion(mask, kernel)
        mask = binary_dilation(mask, kernel)
        img_contour[mask < 1] = 0

        if shrink:
            # img_contour = count_limit_img_contour(img_contour)
            logger.debug('The shrinking method is applied.')
        else:
            logger.debug('The shrinking method is not required.')

        # with open('a.npy', 'wb') as f:
        #     np.save(f, img_array)
        #     np.save(f, img_contour)

        return img_contour

    def get_features(self, subject, img_array=None, img_contour=None):
        # Get or compute the features,
        # the features will be saved for the future useage
        self._check_subject(subject)

        csv = self.subject_folders[subject][0].joinpath(
            '../{}.csv'.format(subject))

        # If we have computed the features,
        # use it.
        allow_use_history = False
        if csv.is_file() and allow_use_history:
            features_table = pd.read_csv(csv, index_col=0)
            logger.debug(
                'Using the features: {}, the other args are ignored.'.format(csv))
            logger.debug('The features are {}'.format(features_table))
            return features_table

        if img_array is None:
            logger.error('The img_array is invalid')

        if img_contour is None:
            logger.error('The img_contour is invalid')

        assert(img_array is not None)
        assert(img_contour is not None)

        # Compute the features
        img_contour = img_contour.copy()
        img_contour[img_contour > 0] = 1

        # Return empty if can NOT find valid regions
        if np.count_nonzero(img_contour) == 0:
            df = pd.DataFrame(
                [{'subject': subject, 'name': 'N.A.', 'value': 0}])
            logger.debug('No valid regions found for {}.'.format(subject))
            return df

        logger.debug('The img_contour contains {} nonzero pixels'.format(
            np.count_nonzero(img_contour)))

        # Get image as sitk format
        # Use 'sitk.GetImageFromArray' to make image and img_mask to make sure they are in the same geometry.
        image = sitk.GetImageFromArray(img_array.copy())
        logger.debug(
            'Got image (sitk) with the shape of {}.'.format(image.GetSize()))

        # Compute features
        img_mask = sitk.GetImageFromArray(img_contour)
        rfe = featureextractor.RadiomicsFeatureExtractor()
        mask = radiomics.imageoperations.getMask(img_mask)

        # Normalize
        # !!! Do not normalize the image,
        # !!! since the template data are not.
        # image = radiomics.imageoperations.normalizeImage(image)

        # Compute Wavelet Image [featureImage, name, {}] x 8 (wavelet-LLH, LLL, ...)
        waveletImages = [e
                         for e in radiomics.imageoperations.getWaveletImage(image, mask)]
        print('--------', len(waveletImages), waveletImages[0])

        # Compute Exponential Image [image, name, {}] x 1 (exponential)
        exponentialImages = [e
                             for e in radiomics.imageoperations.getExponentialImage(image, mask)]
        print('--------', len(exponentialImages), exponentialImages[0])

        # Compute Squareroot Image [image, name, {}] x 1 (squareroot)
        squarerootImages = [e
                            for e in radiomics.imageoperations.getSquareRootImage(image, mask)]
        print('--------', len(squarerootImages), squarerootImages[0])

        rfe.loadImage(image, mask)
        logger.debug(
            'The featureextractor:"{}" loaded image and mask'.format(rfe))

        rfe.enableImageTypeByName('Wavelet', enabled=True)
        logger.debug('RFE Features are {}'.format(rfe.featureClassNames))
        lst = []

        # Features of original x 6
        rfe.disableAllFeatures()
        rfe.enableFeaturesByName(**dict(
            shape=['LeastAxisLength', 'MinorAxisLength',
                   'Maximum3DDiameter',
                   'Maximum2DDiameterColumn'],  # 1, 7, 8 # Will be computed by rfe.computeShape rather than rfe.computeFeatures
            glszm=['ZoneEntropy'],  # 2
            firstorder=['Median'],  # 17
            # ??? Alternative version of Entropy
            glcm=['DifferenceEntropy', 'JointEntropy', 'SumEntropy'],
        ))

        features = rfe.computeFeatures(image, mask, 'original')
        for name in tqdm(features, 'Collecting Features'):
            lst.append((name, features[name]))

        bbox, _ = radiomics.imageoperations.checkMask(image, mask)
        features = rfe.computeShape(image, mask, bbox)
        for name in tqdm(features, 'Collecting Features'):
            lst.append((name, features[name]))

        # Features of exponential x 1
        rfe.disableAllFeatures()
        rfe.enableFeaturesByName(**dict(
            glrlm=['RunEntropy'],  # 5
        ))

        features = rfe.computeFeatures(
            exponentialImages[0][0], mask, exponentialImages[0][1])
        for name in tqdm(features, 'Collecting Features'):
            lst.append((name, features[name]))

        # Features of squareroot x 1
        rfe.disableAllFeatures()
        rfe.enableFeaturesByName(**dict(
            firstorder=['Median'],  # 9
        ))

        features = rfe.computeFeatures(
            squarerootImages[0][0], mask, squarerootImages[0][1])
        for name in tqdm(features, 'Collecting Features'):
            lst.append((name, features[name]))

        # Features of wavelet x ???
        rfe.disableAllFeatures()
        rfe.enableFeaturesByName(**dict(
            glszm=['ZoneEntropy', 'GrayLevelNonUniformity', 'ZoneVariance',
                   'ZoneEntropy', 'SizeZoneNonUniformity'],  # 3, 4, 12, 14, 15, 18
            glrlm=['LongRunEmphasis'],  # 6
            firstorder=['Median', 'InterquartileRange'],  # 13, 17
        ))

        for e in waveletImages:
            features = rfe.computeFeatures(e[0], mask, e[1])
            for name in tqdm(features, e[1]):
                lst.append((name, features[name]))

        df = pd.DataFrame(lst, columns=['name', 'value'])
        df['subject'] = subject
        df = df[['subject', 'name', 'value']]

        logger.debug('Computed features for {} entries.'.format(len(df)))

        df.to_csv(csv.as_posix())
        logger.debug('Saved features to {}'.format(csv))
        logger.debug('The features are {}'.format(df))

        return df


SUBJECT_MANAGER = Subject_Manager(subject_folders)

# %%
# Success
logger.info('Subjects are loaded')

# %%
if __name__ == '__main__':
    for subject in SUBJECT_MANAGER.subjects:
        img_array = SUBJECT_MANAGER.get_array(subject)
        img_contour = SUBJECT_MANAGER.compute_contour(img_array)
        df = SUBJECT_MANAGER.get_features(subject, img_array, img_contour)


# %%
