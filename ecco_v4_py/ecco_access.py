"""
Dummy module to alert users of ecco_access function migration to new package.
"""

def ecco_podaac_access(query,version='v4r4',grid=None,time_res='all',\
                StartDate=None,EndDate=None,snapshot_interval=None,\
                mode='download_ifspace',download_root_dir=None,**kwargs):
    
    raise NameError("The ecco_podaac_access function has migrated to the "\
                    +"new ecco_access package.\n"\
                    +"For more information see https://ecco-access.readthedocs.io.\n"\
                    +"This function is deprecated in ecco_v4_py as of v1.8.0.")
    
    return None


def ecco_podaac_to_xrdataset(query,version='v4r4',grid=None,time_res='all',\
                             StartDate=None,EndDate=None,snapshot_interval=None,\
                             mode='download_ifspace',download_root_dir=None,**kwargs):
    
    raise NameError("The ecco_podaac_to_xrdataset function has migrated to the "\
                    +"new ecco_access package.\n"\
                    +"For more information see https://ecco-access.readthedocs.io.\n"\
                    +"This function is deprecated in ecco_v4_py as of v1.8.0.")
    
    return None
