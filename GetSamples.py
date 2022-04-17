import numpy as np
from astroquery.gaia import Gaia
import pandas as pd

Gaia.launch_job_async(
    "SELECT TOP 1000000 gaia.source_id, "
    "gaia.ra, "
    "gaia.dec, "
    "gaia.phot_g_mean_mag, "
    "gaia.teff, "
    "gaia.feh, "
    "gaia.logg "
    "FROM gaiaedr3.gaia_source_simulation as gaia "  # this needs to be the real gaia DR3.
    "WHERE gaia.phot_g_mean_mag BETWEEN 6 AND 17"
    "AND gaia.teff BETWEEN %1.0f AND %1.0f" % (3000, 12000) +
    "AND gaia.logg BETWEEN %1.0f AND %1.0f" % (3.5, 4.9) +
    "AND gaia.feh BETWEEN %1.0f AND %1.0f" % (-1.5, 1.5), dump_to_file=True, output_format='csv', name="all_gaia_stars")


Gaia.launch_job_async(
    "SELECT TOP 10000 gaia.source_id, "
    "gaia.ra, "
    "gaia.dec, "
    "gaia.phot_g_mean_mag, "
    "gaia.teff, "
    "gaia.feh, "
    "gaia.logg "
    "FROM gaiaedr3.gaia_source_simulation as gaia "  # this needs to be the real gaia DR3.
    "WHERE gaia.phot_g_mean_mag BETWEEN 6 AND 17"
    "AND gaia.teff BETWEEN %1.0f AND %1.0f" % (3000, 12000) +
    "AND gaia.logg BETWEEN %1.0f AND %1.0f" % (3.5, 4.9) +
    "AND gaia.feh BETWEEN %1.0f AND %1.0f" % (-1.5, 1.5), dump_to_file=True, output_format='csv', name="fake_binaries"
    "ORDER BY RANDOM()")


# remove stars from the control sample that are in the science sample using source_id.
# Needs to be done with GAIA science sample.