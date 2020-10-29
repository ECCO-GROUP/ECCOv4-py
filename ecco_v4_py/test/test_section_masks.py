
import ecco_v4_py as ecco

from .test_common import llc_mds_datadirs, get_test_vectors

def test_section_endpoints():
    """Ensure that the listed available sections are actually there
    """

    for section in ecco.get_available_sections():
        assert ecco.get_section_endpoints(section) is not None

def test_calc_all_sections(get_test_vectors):
    """Ensure that we can compute all section masks...
    not sure how to test these exactly...
    """

    ds = get_test_vectors

    for section in ecco.get_available_sections():
        pt1,pt2 = ecco.get_section_endpoints(section)
        maskC,maskW,maskS = ecco.get_section_line_masks(pt1,pt2,ds)
