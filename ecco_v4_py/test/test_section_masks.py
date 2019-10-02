
from __future__ import division, print_function
import ecco_v4_py as ecco


def test_section_endpoints():
    """Ensure that the listed available sections are actually there
    """

    for section in ecco.get_available_sections():
        assert ecco.get_section_endpoints(section) is not None
