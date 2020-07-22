from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys; sys.path.append("..")
from constants import gain, calibration_factor_func

# Script parameters:
sbin_factors = [1, 2, 4]    # The choices for the spatial binning parameter, one line for one value
prebin = 70

# Some plot formatting:
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

plt.rcParams.update({'font.size': 10})
plt.rcParams["figure.figsize"] = cm2inch(12, 12 / 1.618 * 1.5)


def prepare(window):

    # Set appropriate settings:
    # Show "raw" measurement images, not processed ones. This is so that the detector saturation can be taken from
    # the "Average" ROIs. Also, don't ignore the signals from these ROIs. Finally, make sure the quotient image is not
    # calibrated, as the calibration will take place in the "run" routine.

    window.checkBox_show_processed.setChecked(False)        # Show raw measurement images
    window.checkBox_ignore_ROI_signals.setChecked(False)    # Do not ignore signals
    window.radioButton_units_1.setChecked(True)             # Use tilde tau images

    # Go back and forth once in the image stack to update ROIs after these changes:
    window.horizontalSlider_stack_upper.setValue(window.horizontalSlider_stack_upper.value() + 1)
    window.horizontalSlider_stack_upper.setValue(window.horizontalSlider_stack_upper.value() + -1)

    # For each sbin in sbin_factors, an array of average-ROI subimages vs. time will be extracted in the "run" function
    window.s.free_memory = []
    for _ in sbin_factors:
        window.s.free_memory.append([])


def run(window):

    for i, sbin in enumerate(sbin_factors):

        ## Set the spatial binning parameter
        # Write directly to Settings instance to set x-spatial binning, avoids unneeded call to peak_view
        window.s.sbin_x.__set__(sbin)

        # Change the y-spatial binning via GUI element to automatically call peak_view and update quotient image
        window.spinBox_sbin_y.setValue(sbin)

        # Extract ROI data
        ROI_Data = window.avg_ROI(2)
        window.s.free_memory[i].append(ROI_Data)


def trail(window):

    # Instantiate matplotlib figure and get axis from it
    plt.figure()
    ax = plt.gca()

    # The maximum of the maximal temporal binning parameter (divided by prebin)
    n_t_max_h = len(window.s.free_memory[0])

    # Grab cell cd from the Settings instance and calculate the calibration factor from it
    cell_cd = window.s.cell_cd.v
    calib = calibration_factor_func(cell_cd)

    binning_factors = []    # This will hold the array of total binning factors for each sbin

    for i, sbin in enumerate(sbin_factors):

        # Stack the ROI sub-images along depth (axis 2)
        window.s.free_memory[i] = np.dstack(window.s.free_memory[i])

        label = "$N_s = {}$".format(sbin)

        detection_limits = []
        binning_factors.append([])

        series = window.s.free_memory[i]

        for t in range(1, n_t_max_h):

            # average over t images and calculate the std:
            pixels = np.nanmean(series[:, :, 0:t], axis=2)
            detection_limits.append(np.std(pixels))
            binning_factors[i].append(t * prebin * sbin**2)

        ax.plot(binning_factors[i], calib * np.asarray(detection_limits), label=label)

    # Extract the saturation of the two raw measurement images. This is for calculating the pure photon noise line
    sat1 = window.s.sc1.average.v
    sat2 = window.s.sc2.average.v

    # Pure photon noise for a single pair of images should be:
    ph_noise = sqrt(1/gain * (1/sat1 + 1/sat2))

    ## Select which binning factors to plot on the ordinate for pure photon noise line.
    # This should be the range from the first binning factor for the first sbin value to the last binning factor
    # of the last sbin value.
    binning_factors_to_plot = np.arange(binning_factors[0][0], binning_factors[-1][-1])

    ax.plot(binning_factors_to_plot,
            calib * ph_noise/np.sqrt(binning_factors_to_plot),
            label="Pure photon noise", color="black")

    # Plot formatting etc.
    plt.title("Allan plot".format(window.s.read_path.v))
    plt.xlabel("Total binning factor $N = N_t \cdot N_s^2$")
    plt.ylabel(r"Detection limit [$\mathrm{molec \ cm}^{-2}$]")

    plt.xscale("log")
    plt.yscale("log")

    plt.grid(which="both")

    ax.legend()
    plt.tight_layout()
    plt.show()
