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

__all__ = ["makeSimbadLink"]

from IPython.display import display, Markdown
import astropy.coordinates as coord
from astroquery.simbad import Simbad
import astropy.units as u


def makeSimbadLink(source, arcsec_radius=3.0, make_table=False):
    """Search Simbad for associated sources within a 3 arcsecond region
        Maximum search radius should not be greater than 5 arcseconds.

    Parameters
    ----------
    source : 'pandas.core.series.Series'
        Source pandas dataframe that include the ra and dec of the source.

    arcsec_radius : 'float'
        Search radius submitted to Simbad in arcseconds.
        Default radius is 3 arcseconds.

        Returns
    -------
    results_table : `astropy.table.table.Table`
        A table of Simbad search results from astro_query.
    """
    ra = source["ra"]
    dec = source["decl"]
    search_results = f"http://simbad.cds.unistra.fr/simbad/sim-coo?Coord={ra}+{dec}" \
                     f"&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=" \
                     f"{arcsec_radius}&Radius.unit=arcsec&submit=submit+query&CoordList="
    display(Markdown(f"[Link to Simbad search]({search_results})"))

    try:
        source_coords = coord.SkyCoord(ra, dec, frame="icrs", unit=(u.deg, u.deg))
        results_table = Simbad.query_region(
            source_coords, radius=f"0d0m{arcsec_radius}s"
        )
        print(results_table)
        if results_table is not None:
            results_table["MAIN_ID", "RA", "DEC"].pprint_all()

            return results_table

        else:
            raise ValueError

    except ValueError:
        print(f"No matched sources within {arcsec_radius} arcseconds.")
