# This file is part of analysis_ap.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["make_simbad_link"]

from IPython.display import display, Markdown
import astropy.coordinates as coord
from astroquery.simbad import Simbad
import astropy.units as u


def make_simbad_link(ra, dec, radius_arcsec=3.0):
    """Search Simbad for associated sources within a 3 arcsecond region.

    Parameters
    ----------
    ra : 'float'
        Ra from source.

    dec : 'float'
        Dec from source.

    radius_arcsec : 'float'
        Search radius submitted to Simbad in arcseconds.
        Default radius is 3 arcseconds.

    Returns
    -------
    results_table : `astropy.table.table.Table`
        A table of Simbad search results.
    """

    search_results = f"http://simbad.cds.unistra.fr/simbad/sim-coo?Coord={ra}+{dec}" \
                     f"&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=" \
                     f"{radius_arcsec}&Radius.unit=arcsec&submit=submit+query&CoordList="
    display(Markdown(f"[Link to Simbad search]({search_results})"))

    source_coords = coord.SkyCoord(ra, dec, frame="icrs", unit=(u.deg, u.deg))
    customSimbad = Simbad()
    customSimbad.TIMEOUT = 600
    customSimbad.add_votable_fields("otype(V)")
    results_table = customSimbad.query_region(
        source_coords, radius=radius_arcsec*u.arcsecond
    )

    if results_table is not None:

        return results_table

    else:
        print(f"No matched sources within {radius_arcsec} arcseconds.")

        return None
