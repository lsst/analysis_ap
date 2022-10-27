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

__all__ = ("HitsDiaPlot",)

from lsst.analysis.tools.actions.plot.diaSkyPlot import DiaSkyPanel, DiaSkyPlot
from lsst.analysis.tools.actions.vector import DownselectVector, RangeSelector
from lsst.analysis.tools.interfaces import AnalysisPlot


class HitsDiaPlot(AnalysisPlot):
    """Specialized plotter for the DECam HiTS dataset
    for plotting RA/Dec of DiaSources.

    It makes two panels that make the HiTS fields look pretty.
    """

    def setDefaults(self):
        super().setDefaults()

        # Filter data into `ra1s` and `dec1s` for one panel,
        # and `ra2s` and `dec2s` for the other`
        self.process.buildActions.ra1s = DownselectVector(
            vectorKey="ra", selector=RangeSelector(column="dec", maximum=-2)
        )
        self.process.buildActions.dec1s = DownselectVector(
            vectorKey="dec", selector=RangeSelector(column="dec", maximum=-2)
        )
        self.process.buildActions.ra2s = DownselectVector(
            vectorKey="ra", selector=RangeSelector(column="dec", minimum=-2)
        )
        self.process.buildActions.dec2s = DownselectVector(
            vectorKey="dec", selector=RangeSelector(column="dec", minimum=-2)
        )

        self.produce = DiaSkyPlot()

        # Right panel: single HiTS field
        panelRight = DiaSkyPanel()

        panelRight.xlabel = "RA (deg)"
        panelRight.ylabel = "Dec (deg)"
        panelRight.ra = "ra1s"
        panelRight.dec = "dec1s"
        panelRight.rightSpinesVisible = False
        panelRight.size = 0.1
        panelRight.alpha = 0.2
        subplot2gridShape = (100, 100)
        subplot2gridLoc = (0, 0)
        panelRight.subplot2gridShapeRow = subplot2gridShape[0]
        panelRight.subplot2gridShapeColumn = subplot2gridShape[1]
        panelRight.subplot2gridLocRow = subplot2gridLoc[0]
        panelRight.subplot2gridLocColumn = subplot2gridLoc[1]
        panelRight.subplot2gridRowspan = 90
        panelRight.subplot2gridColspan = 50

        self.produce.panels["panel_right"] = panelRight

        # Left panel: two overlapping HiTS fields
        panelLeft = DiaSkyPanel()

        panelLeft.xlabel = "RA (deg)"
        panelLeft.ylabel = "Dec (deg)"
        panelLeft.ra = "ra2s"
        panelLeft.dec = "dec2s"
        panelLeft.leftSpinesVisible = False
        panelLeft.size = 0.1
        panelLeft.alpha = 0.2
        subplot2gridShape = (100, 100)
        subplot2gridLoc = (0, 55)
        panelLeft.subplot2gridShapeRow = subplot2gridShape[0]
        panelLeft.subplot2gridShapeColumn = subplot2gridShape[1]
        panelLeft.subplot2gridLocRow = subplot2gridLoc[0]
        panelLeft.subplot2gridLocColumn = subplot2gridLoc[1]
        panelLeft.subplot2gridRowspan = 50
        panelLeft.subplot2gridColspan = 45

        self.produce.panels["panel_left"] = panelLeft
