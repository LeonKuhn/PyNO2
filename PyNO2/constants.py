from os import listdir

import numpy as np

import serial


# Mean absorption of NO2 [cm^2 / molec] in the 430-450nm spectral range
mean_absorption = 5.101179556419726e-19

# Camera gain [e-/ADU]:
gain = 0.53

# Calibration parameters from the S(tau, Sc) calibration fit and the corresponding fit function
a = -3.00493069836e-58
b = 1.072709449e-38


def calibration_factor_func(cell_cd):
    return 1 / (a * cell_cd ** 2 + b * cell_cd)


def spatial_bin(im, s_bin_x, s_bin_y):
    """
    Bins an image digitally. The image size is unaltered, i.e. pixel values are repeated.
    :param im: np.ndarray, image to bin
    :param s_bin: Int, valid binning parameters. i.e. s_bin = 2 applies a 2x2 binning to the image.
    :return: np.ndarray, binned image as String.
    """

    new_shape = (im.shape[0] // s_bin_x, im.shape[1] // s_bin_y)
    shape = (new_shape[0], im.shape[0] // new_shape[0], new_shape[1], im.shape[1] // new_shape[1])

    data_im = im.reshape(shape)

    # Average:
    data_im = data_im.mean(-1).mean(1)

    # Scale the images up:
    data_im = data_im.repeat(s_bin_x, axis=0).repeat(s_bin_y, axis=1)

    return data_im


class DarkLut:
    """
    DarkLut is a class for dark signal look up tables.
    """

    def __init__(self, path: str) -> None:
        """
        :param path: Path to a folder with temperature-sorted dark images as .npy files
        """
        self.path = path

    def __call__(self, temperature, n=1):
        """
        Returns the dark image for a given temperature.
        :param temperature: temperature in °C for which the dark image should be returned
        :return: np.ndarray
        """

        # Get list of all temperatures in this look up table
        temperatures = [float(x[:-4]) for x in listdir(self.path) if x[0].isdigit()]

        # Find the temperature closest to the requested temperature
        closest = min(temperatures, key=lambda x: abs(x - temperature))

        # Return the dark signal image for this temperature
        dark_image = np.load(self.path + "{}.npy".format(closest)).T

        # Bin it:
        dark_image = spatial_bin(dark_image, n, n)
        return dark_image


class CalibrationLut:
    """
    CalibrationLut is a class for calibration look up tables.
    """

    def __init__(self, path):
        """
        :param path: Path to a file of calibration curves as .npy file
        """

        self.path = path
        self.data = np.load(self.path, allow_pickle=True).item()

    def __call__(self, Sc, output="cd"):
        """
        Returns the calibration factor to convert instrument signal to colum densities or optical depth
        for a given cell column density.

        :param Sc: cell column density in molec / cm^2
        :param output:
        "tau" to return signal -> optical depth conversion factor
        else return signal -> column density conversion factor or
        :return: conversion factor as float
        """

        # List of all cell column densities for which this LUT has calibration curves
        entries = self.data.keys()

        # Find the cell cd closest to the requested cell cd
        closest = min(entries, key=lambda x: abs(x - Sc))

        # Get conversion factor for this cell cd
        conversion_factor = 1/self.data[closest]

        if output == "tau":
            # If the conversion factor to optical depth was requested
            conversion_factor *= mean_absorption

        return conversion_factor

    def c2s(self, calib):
        """
        Returns the Sc value of a given calibration factor given by the instrument model
        :param calib: Float, calibration factor
        :return: Float, Sc
        """

        calib = 1/calib

        values = self.data.values()

        closest = min(values, key=lambda x: abs(x - calib))

        Sc = ([key for (key, value) in self.data.items() if value == closest][0])

        return Sc


class FFC:
    """
    FFC is a class for flat field correction.
    """

    def __init__(self, path):
        """
        :param path: Path to an ffc image as a .npy file. Assumes this npy file has been t_exp and dc corrected.
        """
        self.path = path
        self.im = np.load(self.path, allow_pickle=True)

    def __call__(self, n=1):
        """
        :return: Image correction factor. This simply needs to be multiplied with the dark-signal corrected raw image.
        (Note: See the Wikipedia article for flat field correction: https://en.wikipedia.org/wiki/Flat-field_correction)
        """

        im = self.im

        # Bin it:
        im = spatial_bin(im, n, n)

        # Calculate image-averaged value of im_corrected
        m = np.mean(im)

        # Finally return correction factor
        return m / im
        # return 1/im


class NLC:
    """
    NLC is a class for non-linearity correction.
    """

    def __init__(self, *args):
        """
        Initializes the non-linearity correction
        :param args: list of lists. Each list of the list has the format
        [path_to_NLC_fit_file, lower limit, upper limit]
        This way, polynomial fits of the instrument response as function of saturation can be used to correct segments of an image based on their
        saturation levels.
        """

        self.correction_dict = {}

        for segment in args:

            lower, upper = segment[1], segment[2]
            poly = segment[0]
            self.correction_dict.update({(lower, upper): poly})

    def correct(self, pixel):
        """
        Returns a correction image (to divide the target image by)
        :param pixel: the pixel to correct
        :return: correction image as np.ndarray
        """

        condlist = list([(seg[0] <= pixel) & (pixel < seg[1]) for seg, poly in self.correction_dict.items()])
        choicelist = list([poly(pixel) for seg, poly in self.correction_dict.items()])

        return np.select(condlist, choicelist)  # Returns image of correction values

# Camera IDs
cam1_id = "19224942"
cam2_id = "19224934"

# Load the two dark signal look up tables:
dark_lut_1 = DarkLut(
    "Dark_Noise_Evaluations/LUT_19224942_2019_10_17_15_53_05/"
)

dark_lut_2 = DarkLut(
    "Dark_Noise_Evaluations/LUT_19224934_2019_10_17_15_55_35/"
)

NLC_cam1 = NLC(
    [np.poly1d(np.load("NLC/NLC_order-3_cam-1_lower-0_upper-5000_.npy")), 0, 5000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-1_lower-5000_upper-10000_.npy")), 5000, 10000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-1_lower-10000_upper-15000_.npy")), 10000, 15000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-1_lower-15000_upper-20000_.npy")), 15000, 20000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-1_lower-20000_upper-30000_.npy")), 20000, 30000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-1_lower-30000_upper-65536_.npy")), 30000, 2**16]
)

NLC_cam2 = NLC(
    [np.poly1d(np.load("NLC/NLC_order-3_cam-2_lower-0_upper-5000_.npy")), 0, 5000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-2_lower-5000_upper-10000_.npy")), 5000, 10000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-2_lower-10000_upper-15000_.npy")), 10000, 15000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-2_lower-15000_upper-20000_.npy")), 15000, 20000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-2_lower-20000_upper-30000_.npy")), 20000, 30000],
    [np.poly1d(np.load("NLC/NLC_order-3_cam-2_lower-30000_upper-65536_.npy")), 30000, 2**16]
)

# Initialize the Arduino Uno serial connection. This is only necessary for measurements, but not for evaluations
try:
    tempSens = serial.Serial(
        port='COM3',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.EIGHTBITS,
        timeout=0)

    print("connected to: " + tempSens.portstr)
except:
    tempSens = None

