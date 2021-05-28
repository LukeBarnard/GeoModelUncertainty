from astropy.time import Time, TimeDelta
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import glob
import pandas as pd
import scipy.stats as st
import sunpy.coordinates.sun as sn
from palettable.colorbrewer.qualitative import Dark2_5, Set1_3

import huxt as H
import huxt_inputs as HIN
import huxt_analysis as HA

class Observer:
    
    @u.quantity_input(longitude=u.deg)
    def __init__(self, model, longitude, el_min=4.0, el_max=60.0, color='k', name='Observer'):
        
        ert_ephem = model.get_observer('EARTH')
        
        self.time = ert_ephem.time 
        self.r = ert_ephem.r
        self.lon = ert_ephem.lon*0 + longitude
        self.lat = ert_ephem.lat
        self.el_min = el_min
        self.el_max = el_max
        # Force longitude into 0-360 domain
        id_over = self.lon > 360*u.deg
        id_under = self.lon < 0*u.deg
        if np.any(id_over):
            self.lon[id_over] = self.lon[id_over] - 360*u.deg
        if np.any(id_under):
            self.lon[id_under] = self.lon[id_under] + 360*u.deg
            
        cme = model.cmes[0]
        self.cme_flank = self.compute_flank_profile(cme)
        self.fp = self.compute_fixed_phi(cme)
        self.hm = self.compute_harmonic_mean(cme)
        self.sse = self.compute_sse(cme)
        self.elp = self.compute_elp(cme)
        
        # Restrict flank tracking and geometric modelling to specified elongation range
        id_above = self.cme_flank['el'] > self.el_max
        self.cme_flank.loc[id_above, ['lon', 'r', 'el']] = np.NaN
        self.fp.loc[id_above, ['beta', 'r_apex', 'v_apex']] = np.NaN
        self.hm.loc[id_above, ['beta', 'r_apex', 'v_apex']] = np.NaN
        self.sse.loc[id_above, ['beta', 'r_apex', 'v_apex']] = np.NaN
        self.elp.loc[id_above, ['beta', 'r_apex', 'v_apex']] = np.NaN
        
        self.color = color  # Color for plotting
        self.name = name  # Name for plotting
        
    def compute_flank_profile(self, cme):
        """
        Compute the time elongation profile of the flank of a ConeCME in HUXt. The observer longtidue is specified
        relative to Earth but otherwise matches Earth's coords.

        Parameters
        ----------
        cme: A ConeCME object from a completed HUXt run (i.e the ConeCME.coords dictionary has been populated).
        Returns
        -------
        obs_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STA's perspective, including the
                    time, elongation, position angle, and HEEQ radius and longitude.
        """
        
        times = Time([coord['time'] for i, coord in cme.coords.items()])
        model_time = [coord['model_time'].value for i, coord in cme.coords.items()]
        
        # Compute observers location using earth ephem, adding on observers longitude offset from Earth
        # and correct for runover 2*pi
        flank = pd.DataFrame(index=np.arange(times.size), columns=['time','model_time', 'el', 'r', 'lon'])
        flank['time'] = times.jd
        flank['model_time'] = model_time

        for i, coord in cme.coords.items():
                
            if len(coord['r']) == 0:
                flank.loc[i, ['lon', 'r', 'el']] = np.NaN
                continue

            r_obs = self.r[i]
            x_obs = self.r[i] * np.cos(self.lon[i])
            y_obs = self.r[i] * np.sin(self.lon[i])
            
            lon_cme = coord['lon']
            r_cme = coord['r']
            id_front = coord['front_id']==1.0
            lon_cme = lon_cme[id_front]
            r_cme = r_cme[id_front]
            
            if np.any(np.isfinite(r_cme)):
                
                x_cme = r_cme * np.cos(lon_cme)
                y_cme = r_cme * np.sin(lon_cme)

                #############
                # Compute the observer CME distance
                # All bodies and CME in same plane, so don't worry about latitudes.
                x_cme_s = x_cme - x_obs
                y_cme_s = y_cme - y_obs
                s = np.sqrt(x_cme_s**2 + y_cme_s**2)

                numer = (r_obs**2 + s**2 - r_cme**2).value
                denom = (2.0 * r_obs * s).value
                e_obs = np.arccos(numer / denom)

                # Find the flank coordinate and update output
                id_obs_flank = np.argmax(e_obs)       
                flank.loc[i, 'lon'] = lon_cme[id_obs_flank].value
                flank.loc[i, 'r'] = r_cme[id_obs_flank].to(u.km).value
                flank.loc[i, 'el'] = np.rad2deg(e_obs[id_obs_flank])
                
        keys = ['lon', 'r', 'el']
        flank[keys] = flank[keys].astype(np.float64)
        return flank
        
    def compute_fixed_phi(self, cme):
        
        size = self.cme_flank.shape[0]
        fp_keys = ['time','model_time','beta','r_apex', 'v_apex', 'lon']
        geomod = pd.DataFrame(index=np.arange(size), columns=fp_keys)
        geomod['time'] = self.cme_flank['time']
        geomod['model_time'] = self.cme_flank['model_time']
        
        pi = np.pi*u.rad
        piby2 = pi/2.0
        d0 = self.r  #Observer distance
        el = self.cme_flank['el'].values*u.deg
        beta = self.lon - cme.longitude
        id_over = beta > pi
        if np.any(id_over):
            beta[id_over] = 2*pi - beta[id_over]

        psi = pi - beta - el
        r = d0*np.sin(el) / np.sin(psi)
        
        geomod['r_apex'] = r.to(u.km).value
        geomod['v_apex'] = np.gradient(geomod['r_apex'], geomod['model_time'])
        # Drop last point, as can be skewed by having less data
        geomod['beta'] = beta.to(u.deg).value
        geomod['lon'] = cme.longitude.to(u.deg).value
        geomod[fp_keys] = geomod[fp_keys].astype(np.float64)        
        return geomod
    
    def compute_harmonic_mean(self, cme):
        
        size = self.cme_flank.shape[0]
        hmm_keys = ['time','model_time','beta','r_apex', 'v_apex', 'r_center', 'radius', 'lon']
        geomod = pd.DataFrame(index=np.arange(size), columns=hmm_keys)
        geomod['time'] = self.cme_flank['time']
        geomod['model_time'] = self.cme_flank['model_time']
        
        pi = np.pi*u.rad
        piby2 = pi/2.0
        d0 = self.r # Observer distance
        el = self.cme_flank['el'].values*u.deg
        beta = self.lon - cme.longitude
        id_over = beta > pi
        if np.any(id_over):
            beta[id_over] = 2*pi - beta[id_over]

        psi = pi - beta - el
        r_apex = 2*d0*np.sin(el) / (np.sin(psi) + 1)
        
        # Calc radius and center, special case of sse, Eq4 Moestl2012
        radius = r_apex / 2.0
        r_center = r_apex - radius
                
        geomod['r_apex'] = r_apex.to(u.km).value
        geomod['r_center'] = r_center.to(u.km).value
        geomod['radius'] = radius.to(u.km).value
        
        geomod['v_apex'] = np.gradient(geomod['r_apex'], geomod['model_time'])
        geomod['beta'] = beta.to(u.deg).value
        geomod['lon'] = cme.longitude.to(u.deg).value
        geomod[hmm_keys] = geomod[hmm_keys].astype(np.float64)        
        return geomod
    
    def compute_sse(self, cme):
        
        # Make dataframe for output
        size = self.cme_flank.shape[0]
        sse_keys = ['time','model_time','beta','r_apex', 'v_apex', 'r_center', 'radius', 'lon']
        geomod = pd.DataFrame(index=np.arange(size), columns=sse_keys)
        geomod['time'] = self.cme_flank['time']
        geomod['model_time'] = self.cme_flank['model_time']

        pi = np.pi*u.rad
        piby2 = pi/2.0
        d0 = self.r  #Sun-observer distance
        half_width = cme.width / 2.0
        el = self.cme_flank['el'].values*u.deg
        beta = self.lon - cme.longitude
        id_over = beta > pi
        if np.any(id_over):
            beta[id_over] = 2*pi - beta[id_over]

        psi = pi - beta - el
        top = d0 * np.sin(el) * (1 + np.sin(half_width))
        bottom = (np.sin(psi) + np.sin(half_width))
        r_apex = top / bottom
        
        # Calc radius from apex distance and halfwidth. Eq 4 of Moestl2012
        radius = r_apex * np.sin(half_width) / (1 + np.sin(half_width))
        # Calc SSE center
        r_center = r_apex - radius
                
        geomod['r_apex'] = r_apex.to(u.km).value
        geomod['r_center'] = r_center.to(u.km).value
        geomod['radius'] = radius.to(u.km).value
        
        geomod['v_apex'] = np.gradient(geomod['r_apex'], geomod['model_time'])
        geomod['beta'] = beta.to(u.deg).value
        geomod['lon'] = cme.longitude.to(u.deg).value
        geomod[sse_keys] = geomod[sse_keys].astype(np.float64)        
        return geomod
    
    def compute_elp(self, cme):
        
        # Make dataframe for output
        size = self.cme_flank.shape[0]
        elp_keys = ['time','model_time','beta','r_apex', 'v_apex', 'r_center', 'lon', 'f', 'r_a', 'r_b']
        geomod = pd.DataFrame(index=np.arange(size), columns=elp_keys)
        geomod['time'] = self.cme_flank['time']
        geomod['model_time'] = self.cme_flank['model_time']
        
        pi = np.pi*u.rad
        piby2 = pi/2.0
        f = 0.7
        d0 = self.r  #Sun-observer distance
        el = self.cme_flank['el'].values*u.deg
        half_width = cme.width / 2.0
        beta = self.lon - cme.longitude
        id_over = beta > pi
        if np.any(id_over):
            beta[id_over] = 2*pi - beta[id_over]

        omega = pi - el - beta
        # Make sure omega is <90.0, otherwise outside angle has been specified.
        omega[omega > piby2] = pi - omega[omega > piby2]
        
        phi = np.arctan((f**2) * np.tan(omega))
        theta = np.arctan((f**2) * np.tan(half_width))
        
        o_phi = np.sqrt(((f*np.cos(phi))**2 + (np.sin(phi))**2))
        o_theta = np.sqrt(((f*np.cos(theta))**2 + (np.sin(theta))**2))

        top = d0 * np.sin(el) * np.sin(half_width) * o_phi * o_theta
        
        psi = piby2 + theta - half_width
        term1 = np.sin(psi)*np.sin(omega)*o_phi
        zeta = piby2 + phi - omega
        term2 = np.sin(zeta)*np.sin(half_width)*o_theta
        bottom = term1 + term2
        
        r_b = top / bottom
        r_a = r_b / f
        r_half_width = r_b / o_theta
        
        r_center = r_half_width * np.sin(psi) / np.sin(half_width)
        r_apex = r_center + r_b
        
        geomod['r_apex'] = r_apex.to(u.km).value
        geomod['r_center'] = r_center.to(u.km).value
        geomod['r_a'] = r_a.to(u.km).value
        geomod['r_b'] = r_b.to(u.km).value
        
        geomod['v_apex'] = np.gradient(geomod['r_apex'], geomod['model_time'])
        geomod['beta'] = beta.to(u.deg).value
        geomod['lon'] = cme.longitude.to(u.deg).value
        geomod['f'] = f
        geomod[elp_keys] = geomod[elp_keys].astype(np.float64)
        return geomod
    
def fixed_phi(cme, r_obs, lon_obs, el):
    """
    Return the FP point for a specified, CME, observer location, and elongation.
    """    
    pi = np.pi*u.rad
    piby2 = pi/2.0
    beta = lon_obs - cme.longitude
    if beta > pi:
        beta = 2*pi - beta

    psi = pi - beta - el
    r_apex = r_obs*np.sin(el) / np.sin(psi)
    lon_apex = cme.longitude.copy()
    return lon_apex, r_apex


def harmonic_mean(cme, r_obs, lon_obs, el):
    """
    Return the HM circle parameters for the HM model for a specified, CME, observer location, and elongation.
    """ 
    pi = np.pi*u.rad
    piby2 = pi/2.0
    beta = lon_obs - cme.longitude
    if beta > pi:
        beta = 2*pi - beta

    psi = pi - beta - el
    r_apex = 2*r_obs*np.sin(el) / (np.sin(psi) + 1)

    # Calc radius and center, special case of sse, Eq4 Moestl2012
    radius = r_apex / 2.0
    r_center = r_apex - radius
    lon_apex = cme.longitude.copy()
    return lon_apex, r_apex, r_center, radius


def self_similar_expansion(cme, r_obs, lon_obs, el):
    """
    Return the SSE circle parameters for the SSE model for a specified, CME, observer location, and elongation.
    """
    pi = np.pi*u.rad
    piby2 = pi/2.0
    half_width = cme.width / 2.0
    beta = lon_obs - cme.longitude
    if beta > pi:
        beta = 2*pi - beta

    psi = pi - beta - el
    top = r_obs * np.sin(el) * (1 + np.sin(half_width))
    bottom = (np.sin(psi) + np.sin(half_width))
    r_apex = top / bottom

    # Calc radius from apex distance and halfwidth. Eq 4 of Moestl2012
    radius = r_apex * np.sin(half_width) / (1 + np.sin(half_width))
    # Calc SSE center
    r_center = r_apex - radius
    lon_apex = cme.longitude.copy()
    return lon_apex, r_apex, r_center, radius


def elcon(cme, r_obs, lon_obs, el):
    """
    Return the ElCon elipse parameters for the ElCon model for a specified, CME, observer location, and elongation.
    """
    pi = np.pi*u.rad
    piby2 = pi/2.0
    f = 0.7
    half_width = cme.width / 2.0
    beta = lon_obs - cme.longitude
    if beta > pi:
        beta = 2*pi - beta

    omega = pi - el - beta
    # Make sure omega is <90.0, otherwise outside angle has been specified.
    omega[omega > piby2] = pi - omega[omega > piby2]

    phi = np.arctan((f**2) * np.tan(omega))
    theta = np.arctan((f**2) * np.tan(half_width))

    o_phi = np.sqrt(((f*np.cos(phi))**2 + (np.sin(phi))**2))
    o_theta = np.sqrt(((f*np.cos(theta))**2 + (np.sin(theta))**2))

    top = r_obs * np.sin(el) * np.sin(half_width) * o_phi * o_theta

    psi = piby2 + theta - half_width
    term1 = np.sin(psi)*np.sin(omega)*o_phi
    zeta = piby2 + phi - omega
    term2 = np.sin(zeta)*np.sin(half_width)*o_theta
    bottom = term1 + term2

    r_b = top / bottom
    r_a = r_b / f
    r_half_width = r_b / o_theta

    r_center = r_half_width * np.sin(psi) / np.sin(half_width)
    r_apex = r_center + r_b
    lon_apex = cme.longitude.copy()
    return lon_apex, r_apex, r_center, r_a, r_b


def get_project_dirs():
    """
    Function to pull out the directories of boundary conditions, ephemeris, and to save figures and output data.
    """
    # Find the config.dat file path
    files = glob.glob('config.dat')

    # Extract data and figure directories from config.dat
    with open(files[0], 'r') as file:
        lines = file.read().splitlines()
        root = lines[0].split(',')[1]
        dirs = {line.split(',')[0]: os.path.join(root, line.split(',')[1]) for line in lines[1:]}

    # Just check the directories exist.
    for val in dirs.values():
        if not os.path.exists(val):
            print('Error, invalid path, check config.dat: ' + val)

    return dirs


def build_cme_scenarios():
    """
    Function to build the average, fast, and extreme CME scenarios used in the modelling.
    These are built from the percentiles of CME speed and width distributions of the 
    HELCATS GCS fits in WP3 KINCAT (https://www.helcats-fp7.eu/catalogues/wp3_kincat.html)
    """
    project_dirs = get_project_dirs()
    column_names = ['ID', 'pre_date', 'pre_time', 'last_date', 'last_time', 'carlon',
                    'stolon', 'stolat', 'tilt', 'ssprat', 'h_angle', 'speed',  'mass']
    data = pd.read_csv(project_dirs['HELCATS_data'], names=column_names, skiprows=4, delim_whitespace=True)

    # Compute the CME scenarios using the percentiles of the speed and half angle distributions
    scenario_percentiles = {'average': 0.5, 'fast': 0.85, 'extreme': 0.95}

    # Setup output file. Overwrite if exists.
    out_filepath = os.path.join(project_dirs['out_data'], "CME_scenarios.hdf5")
    if os.path.isfile(out_filepath):
        print("Warning: {} already exists. Overwriting".format(out_filepath))
        os.remove(out_filepath)

    out_file = h5py.File(out_filepath, 'w')


    # Iter through scenarios and save CME properties to outfile                                         
    for key, percentile in scenario_percentiles.items():

        speed = data['speed'].quantile(percentile)
        width = 2*data['h_angle'].quantile(percentile)

        cme_group = out_file.create_group(key)
        cme_group.create_dataset('percentile', data=percentile)                              
        dset = cme_group.create_dataset('speed', data=speed)
        dset.attrs['unit'] = (u.km/u.s).to_string()
        dset = cme_group.create_dataset('width', data=width)
        dset.attrs['unit'] = (u.deg).to_string()


    out_file.close()
    return


def load_cme_scenarios():
    """
    Load in the CME scenarios from their HDF5 file and return them in a dictionary.
    """
    
    project_dirs = get_project_dirs()
    datafile_path = os.path.join(project_dirs['out_data'], 'CME_scenarios.hdf5')
    datafile = h5py.File(datafile_path, 'r')
    cme_scenarios = {}
    for key in datafile.keys():
        cme = datafile[key]
        speed = cme['speed'][()] * u.Unit(cme['speed'].attrs['unit'])
        width = cme['width'][()] * u.Unit(cme['width'].attrs['unit'])
        cme_scenarios[key] = {'speed': speed, 'width': width}

    datafile.close()
    
    return cme_scenarios


def compute_apex_profile(model, cme):
    """
    Compute the kinematics of the CME apex. 
    """
    time  = Time([coord['time'] for i, coord in cme.coords.items()])
    model_time = np.array([coord['model_time'].value for i, coord in cme.coords.items()])*u.s

    apex = pd.DataFrame(index=np.arange(time.size), columns=['time','model_time', 'r', 'lon'])
    apex['time'] = time.jd
    apex['model_time'] = model_time

    id_model_lon = np.argmin(np.abs(model.lon - cme.longitude))
    count = 0
    for i, coord in cme.coords.items():

        # Pull out only the CME front
        id_front = coord['front_id'] == 1.0
        cme_r = coord['r'][id_front]
        cme_lon = coord['lon'][id_front]

        if cme_lon.size != 0:
            # Find the coordinates of the apex
            id_apex_lon = np.argmin(np.abs(cme_lon - cme.longitude))

            lon_apex = cme_lon[id_apex_lon]
            r_apex = cme_r[id_apex_lon]

            apex.loc[i, 'r'] = r_apex.to(u.km).value
            apex.loc[i, 'lon'] = lon_apex.value

    keys = ['time', 'model_time','lon', 'r']
    apex[keys] = apex[keys].astype(np.float64)        
    # Now compute speed of apex.
    apex['v'] = np.gradient(apex['r'], apex['model_time'])
    return apex


def get_model_initialisation_times(n_samples=100, t_start='2008-01-01T00:00:00', t_stop='2016-01-01T00:00:00'):
    """
    Compute n random initialisation times between t_start and t_stop
    """
    np.random.seed(47041547)
    time_min = Time(t_start, format='isot')
    time_max = Time(t_stop, format='isot')
    start_times = np.random.uniform(time_min.jd, time_max.jd, n_samples)

    # BUG FIX - two times can be arbitrarily close with the above code,
    # and elements 48 and 99 are within an hour of each other. Replace 99
    # with a new date that is more than a day away from any other.
    #
    # This is hacky, but a lot of processing time has been spent on the other 
    # runs already, so on balance it makes sense to fix just this one element.
    
    search_for_update = True
    while search_for_update:
        # Get candidate replacement
        ts_update = np.random.uniform(time_min.jd, time_max.jd, 1)

        # Check the candidate is more than 1 day away from closest other time
        diff =  start_times - ts_update
        min_diff = np.min(np.abs(diff))
        if min_diff >= 1.0:
            # Found candidate, exit while. 
            search_for_update = False

    start_times[99] = ts_update
    # END BUG FIX

    start_times = Time(start_times, format='jd')
    return start_times


def create_output_hdf5():
    """
    Open a HDF5 file for saving the modelling results and observer data into
    """
    project_dirs = get_project_dirs()
    out_filepath = os.path.join(project_dirs['out_data'], "CME_scenarios_simulation_results.hdf5")
    if os.path.isfile(out_filepath):
        print("Warning: {} already exists. Overwriting".format(out_filepath))
        os.remove(out_filepath)

    out_file = h5py.File(out_filepath, 'w')
    return out_file


def setup_huxt(start_time, uniform_wind=False):
    """
    Initialise HUXt in Ecliptic plane at start_time.
    """
    
    cr_num = np.fix(sn.carrington_rotation_number(start_time))
    ert = H.Observer('EARTH', start_time)
    vr_in = HIN.get_MAS_long_profile(cr_num, ert.lat.to(u.deg), verbose=False)
    
    if uniform_wind:
        vr_in = vr_in*0 + 400*u.km/u.s
        
    model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                   lon_start=300*u.deg, lon_stop=60*u.deg, simtime=5*u.day, dt_scale=1)
    return model


def hdf5_save_cme(parent_group, cme):
    """
    Save the CME initial conditions
    """
    cme_group = parent_group.create_group('cme')
    
    # Save initial CME properties 
    for k, v in cme.__dict__.items():            
        if k not in ["coords", "frame"]:
            dset = cme_group.create_dataset(k, data=v.value)
            dset.attrs['unit'] = v.unit.to_string()
    return


def hdf5_save_arrival_stats(parent_group, cme):
    """
    Compute the cme arrival statistics and save.
    """
    arrival_group = parent_group.create_group('arrival_stats')
    
    # Compute arrival and transit time and save to file
    arrival_stats = cme.compute_arrival_at_body('EARTH')
    arrival_keys = ['t_arrive', 't_transit', 'hit_lon', 'hit_rad']
    for key in arrival_keys:
        val = arrival_stats[key]
        if key=='t_arrive':
            arrival_group.create_dataset(key, data=val.isot)
        else:
            dset = arrival_group.create_dataset(key, data=val)
            dset.attrs['unit'] = val.unit.to_string()
    return


def hdf5_save_apex_profile(parent_group, model, cme):
    """
    Compute the apex profile and save to file
    """
    # Compute apex kinematics and save to file
    apex_group = parent_group.create_group('cme_apex')
    
    apex = compute_apex_profile(model, cme)
    for key, unit in zip(['time', 'model_time', 'lon', 'r', 'v'], [u.d, u.s, u.rad, u.km, u.km/u.s]):
        dset = apex_group.create_dataset(key, data=apex[key].values)
        dset.attrs['unit'] = unit.to_string()
    return


def hdf5_save_solarwind_boundary_var(parent_group, model, cme):
    """
    Compute the variability of the solar wind solution across CME longitudes
    and at all radii at timestep before CME launch, and save to the hdf5 output.
    """
    # Find launch index of CME, and get inner boundary V at 1 timestep prior to launch
    id_launch = np.argmin(np.abs(model.time_out - cme.t_launch))
    v_bound = model.v_grid[id_launch-1, :, :]

    # Compute v_bound variability over the CME longitudes
    # Center lons on CME, and put between -pi and pi
    lons = np.copy(model.lon)
    lons = lons - cme.longitude
    lons[lons > np.pi*u.rad] = lons[lons > np.pi*u.rad] - 2*np.pi*u.rad
    id_sort = np.argsort(lons)
    lons = lons[id_sort]
    v_bound = v_bound[:, id_sort]

    # Get standard deviation of inner boundary and also standard deviation only over CME longs
    id_cme_lons = (lons >= (cme.longitude - cme.width/2.0)) & (lons <= (cme.longitude + cme.width/2.0))
    
    # Standard deviation in V over all domain
    v_std = np.std(v_bound[:, id_cme_lons])
    dset = parent_group.create_dataset('v_std', data=v_bound_std)
    dset.attrs['unit'] = v_bound_std.unit.to_string()
    
    # Standard deviation in V over only the inner boundary
    v_b_std = np.std(v_bound[0, id_cme_lons])
    dset = parent_group.create_dataset('v_b_std', data=v_b_std)
    dset.attrs['unit'] = v_bound_std.unit.to_string()
    
    # Standard deviation in longitudinal difference in V
    dv_std = np.std(np.diff(v_bound[:, id_cme_lons], axis=1))
    dset = parent_group.create_dataset('dv_std', data=dv_std)
    dset.attrs['unit'] = dv_std.unit.to_string()
    
    # Standard deviation in longitudinal difference in V only over the inner boundary
    dv_b_std = np.std(np.diff(v_bound[0, id_cme_lons]))
    dset = parent_group.create_dataset('dv_b_std', data=dv_b_std)
    dset.attrs['unit'] = dv_b_std.unit.to_string()

    return


def hdf5_save_solarwind_timeseries(parent_group, model):
    """
    Compute solar wind time series at Earth and save to hdf5 output.
    """
    # Save the solar wind profile at Earth.
    insitu_group = parent_group.create_group('earth_monitor')
    earth_ts = HA.get_earth_timeseries(model)
    param_units = {'time':u.d, 'model_time':u.s, 'r':u.solRad, 'lon':u.deg, 'vsw':u.km/u.s}
    for key, val in earth_ts.items():
            
        # Put time in JD
        if key == 'time':
            val = Time(pd.to_datetime(val.values)).jd

        dset = insitu_group.create_dataset(key, data=val)
        dset.attrs['unit'] = param_units[key].to_string()
    return


def hdf5_save_observer(parent_group, observer):
    """
    Save an observers data to HDF5
    """
    obs_group = parent_group.create_group(observer.name)
    
    # Save the observers ephemeris over the obs
    dset = obs_group.create_dataset('time_obs', data=observer.time.jd)
    dset.attrs['unit'] = u.d.to_string()
    for name, param in zip(['r', 'lon', 'lat'], [observer.r, observer.lon, observer.lat]):
        if name == 'r':
            param = param.to(u.km)
            
        dset = obs_group.create_dataset(name+'_obs', data=param.value)
        dset.attrs['unit'] = param.unit.to_string()

    # Now save the elongation profile and flank coordinate.
    for key, unit in zip(['time', 'model_time', 'el', 'r', 'lon'], [u.d, u.s, u.deg, u.km, u.rad]):
        dset = obs_group.create_dataset(key+'_flank', data=observer.cme_flank[key].values)
        dset.attrs['unit'] = unit.to_string()
        
    # Now save the geometric models.
    # FIXED PHI
    fp_group = obs_group.create_group('fp')
    param_keys = {'time':u.d, 'model_time':u.s, 'beta':u.deg, 'r_apex':u.km, 'v_apex':u.km/u.s, 'lon':u.rad}
    for key, unit in param_keys.items():
        dset = fp_group.create_dataset(key, data=observer.fp[key].values)
        dset.attrs['unit'] = unit.to_string()
    
    # HARMONIC MEAN
    hm_group = obs_group.create_group('hm')
    param_keys = {'time':u.d, 'model_time':u.s, 'beta':u.deg, 'r_apex':u.km, 'v_apex':u.km/u.s,
                  'lon':u.rad, 'r_center':u.km, 'radius':u.km}
    for key, unit in param_keys.items():
        dset = hm_group.create_dataset(key, data=observer.hm[key].values)
        dset.attrs['unit'] = unit.to_string()
        
    # SELF SIMILAR EXPANSION
    sse_group = obs_group.create_group('sse')
    param_keys = {'time':u.d, 'model_time':u.s, 'beta':u.deg, 'r_apex':u.km, 'v_apex':u.km/u.s,
                  'lon':u.rad, 'r_center':u.km, 'radius':u.km}
    for key, unit in param_keys.items():
        dset = sse_group.create_dataset(key, data=observer.sse[key].values)
        dset.attrs['unit'] = unit.to_string()
    
    # ELIPTICAL
    elp_group = obs_group.create_group('elp')
    param_keys = {'time':u.d, 'model_time':u.s, 'beta':u.deg, 'r_apex':u.km, 'v_apex':u.km/u.s,
                  'lon':u.rad, 'r_center':u.km, 'f':u.dimensionless_unscaled, 'r_a':u.km, 'r_b':u.km}
    for key, unit in param_keys.items():
        dset = elp_group.create_dataset(key, data=observer.elp[key].values)
        dset.attrs['unit'] = unit.to_string()

    return


def plot_huxt(time, model, observer_list, add_observer=True, add_flank=True):
    """
    Plot the HUXt solution at a specified time, and (optionally) overlay the modelled flank location and field of view
    of a specified observer.
    :param time: The time to plot. The closest value in model.time_out is selected.
    :param model: A HUXt instance with the solution in.
    :param observer_list: A list of Observer instances with the modelled flank.
    :param add_flank: If True, add the modelled flank.
    :param add_fov: If True, highlight the observers field of view.
    :return:
    """
    cme = model.cmes[0]
    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.viridis
    v_sub = model.v_grid.value[id_t, :, :].copy()
    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.NaN
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(200, 800 + 10, 10)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Add on CME boundaries and Observer
    cme_lons = cme.coords[id_t]['lon']
    cme_r = cme.coords[id_t]['r'].to(u.solRad)
    if np.any(np.isfinite(cme_r)):
        # Pad out to close the profile.
        cme_lons = np.append(cme_lons, cme_lons[0])
        cme_r = np.append(cme_r, cme_r[0])
        ax.plot(cme_lons, cme_r, '-', color='darkorange', linewidth=3)
    
    ert = model.get_observer('EARTH')
    ax.plot(ert.lon[id_t], ert.r[id_t], 'co', markersize=16, label='Earth')            

    # Add on the observer
    if add_observer:
        for observer in observer_list:
            ax.plot(observer.lon[id_t], observer.r[id_t], 's', color=observer.color, markersize=16, label=observer.name)

            if add_flank:
                flank_lon = observer.cme_flank.loc[id_t, 'lon']
                flank_rad = observer.cme_flank.loc[id_t, 'r']*u.km.to(u.solRad)
                ax.plot(flank_lon, flank_rad, '.', color=observer.color, markersize=10, zorder=4)
                # Add observer-flank line
                ro = observer.r[id_t]
                lo = observer.lon[id_t]
                ax.plot([lo.value, flank_lon], [ro.value, flank_rad], '--', color=observer.color, zorder=4)

    ax.set_ylim(0, 240)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.patch.set_facecolor('slategrey')

    fig.subplots_adjust(left=0.05, bottom=0.16, right=0.95, top=0.99)
    # Add color bar
    pos = ax.get_position()
    dw = 0.005
    dh = 0.045
    left = pos.x0 + dw
    bottom = pos.y0 - dh
    wid = pos.width - 2 * dw
    cbaxes = fig.add_axes([left, bottom, wid, 0.03])
    cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
    cbar1.set_label('Solar Wind speed (km/s)')
    cbar1.set_ticks(np.arange(200, 810, 100))
    return fig, ax


def plot_huxt_multi(ax, time, model, observer_list, add_observer=True, add_flank=True):
    """
    Plot the HUXt solution at a specified time, and (optionally) overlay the modelled flank location and field of view
    of a specified observer.
    :param time: The time to plot. The closest value in model.time_out is selected.
    :param model: A HUXt instance with the solution in.
    :param observer_list: A list of Observer instances with the modelled flank.
    :param add_flank: If True, add the modelled flank.
    :param add_fov: If True, highlight the observers field of view.
    :return:
    """
    cme = model.cmes[0]
    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.viridis
    v_sub = model.v_grid.value[id_t, :, :].copy()
    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.NaN
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(200, 800 + 10, 10)
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Add on CME boundaries and Observer
    cme_lons = cme.coords[id_t]['lon']
    cme_r = cme.coords[id_t]['r'].to(u.solRad)
    if np.any(np.isfinite(cme_r)):
        # Pad out to close the profile.
        cme_lons = np.append(cme_lons, cme_lons[0])
        cme_r = np.append(cme_r, cme_r[0])
        ax.plot(cme_lons, cme_r, '-', color='darkorange', linewidth=3)
    
    ert = model.get_observer('EARTH')
    ax.plot(ert.lon[id_t], ert.r[id_t], 'co', markersize=12, label='Earth')            

    # Add on the observer
    if add_observer:
        for observer in observer_list:
            ax.plot(observer.lon[id_t], observer.r[id_t], 's', color=observer.color, markersize=12, label=observer.name)

            if add_flank:
                flank_lon = observer.cme_flank.loc[id_t, 'lon']
                flank_rad = observer.cme_flank.loc[id_t, 'r']*u.km.to(u.solRad)
                ax.plot(flank_lon, flank_rad, '.', color=observer.color, markersize=10, zorder=4)
                # Add observer-flank line
                ro = observer.r[id_t]
                lo = observer.lon[id_t]
                ax.plot([lo.value, flank_lon], [ro.value, flank_rad], '--', color=observer.color, zorder=4)

    ax.set_ylim(0, 240)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.patch.set_facecolor('slategrey')
    return


def produce_huxt_ensemble(n=100):
    """
    Generate the 100 member ensemble of HUXt runs for each CME scenario. 
    Save the data to a HDF5 file. Generate a plot of the CME arrival at Earth.
    """
    
    cme_scenarios = load_cme_scenarios()

    # Get array of random model initialisation times
    np.random.seed(47041547)
    start_times = get_model_initialisation_times(n_samples=n)

    # Set which longitudes to observe the CME from. 90'
    observer_longitudes = (360 + np.arange(-10, -100, -10))*u.deg

    # Open HDF5 for output statistics
    out_file = create_output_hdf5()

    # Loop over cme scenarios
    for scenario_key, scenario in cme_scenarios.items():

        cme_group = out_file.create_group(scenario_key)

        # Now loop through initialisations
        for i, start_time in enumerate(start_times):

            run_group = cme_group.create_group("run_{:03d}".format(i))

            # Initialise HUXt at this start time
            model = setup_huxt(start_time)

            # Set up the ConeCME
            t_launch = (1*u.hr).to(u.s)
            cme = H.ConeCME(t_launch=t_launch, longitude=0*u.deg, latitude=model.latitude.to(u.deg),
                            width=scenario['width'], v=scenario['speed'], thickness=1*u.solRad)

            # Solve HUXt and get CME object
            model.solve([cme])
            cme = model.cmes[0]
            
            # Save CME details to file
            hdf5_save_cme(run_group, cme)
            hdf5_save_arrival_stats(run_group, cme)
            hdf5_save_apex_profile(run_group, model, cme)
            hdf5_save_solarwind_timeseries(run_group, model)
            hdf5_save_solarwind_boundary_var(run_group, model, cme)
            
            # Save the observer longitudes 
            dset = run_group.create_dataset("observer_lons", data=observer_longitudes.value)
            dset.attrs['unit'] = observer_longitudes.unit.to_string()
            # Setup group to hold all observers
            all_observers_group = run_group.create_group('observers')
            # Loop through observer longitudes, get each observer, save to file
            for obs_lon in observer_longitudes:
                obs_name = "Observer {:3.2f}".format(obs_lon.value)
                observer = Observer(model, obs_lon, el_min=4.0, el_max=40.0, name=obs_name)
                hdf5_save_observer(all_observers_group, observer)         

            out_file.flush()

            # Make summary plot of arrival
            arrival_stats = cme.compute_arrival_at_body('EARTH')
            if arrival_stats['hit']:
                proj_dirs = get_project_dirs()
                hit_id = arrival_stats['hit_id']
                fig, ax = plot_huxt(model.time_out[hit_id], model, [], add_observer=False)
                fig_name = 'run_{:03d}_{}_arrival.png'.format(i, scenario_key)
                fig_path = os.path.join(proj_dirs['HUXt_figures'], fig_name)
                fig.savefig(fig_path)
                plt.close('all')

    out_file.close()
    return


def plot_kinematics_example_multi_observer():

    for wind_type in ['structured', 'uniform']:

        # Set up HUXt to run all three scenarios in uniform wind, then plot the kinematics out.
        fig, ax = plt.subplots(figsize=(10, 10))
        ax1 = plt.subplot(331, projection='polar')
        ax2 = plt.subplot(332, projection='polar')
        ax3 = plt.subplot(333, projection='polar')
        axt = [ax1, ax2, ax3]
        ax4 = plt.subplot(334)
        ax5 = plt.subplot(335)
        ax6 = plt.subplot(336)
        axm = [ax4, ax5, ax6]
        ax7 = plt.subplot(337)
        ax8 = plt.subplot(338)
        ax9 = plt.subplot(339)
        axb = [ax7, ax8, ax9]

        cme_scenarios = load_cme_scenarios()

        t_arr_stats = []
        r_arr_stats = []

        for i, s_key in enumerate(['average', 'fast', 'extreme']):

            scenario = cme_scenarios[s_key]
            start_time = Time('2008-06-10T00:00:00')
            cr_num = np.fix(sn.carrington_rotation_number(start_time))
            ert = H.Observer('EARTH', start_time)
            vr_in = HIN.get_MAS_long_profile(cr_num, ert.lat.to(u.deg))
            if wind_type == 'uniform':
                vr_in = vr_in*0 + 400*(u.km/u.s)

            model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                           lon_start=300*u.deg, lon_stop=60*u.deg, simtime=5*u.day, dt_scale=1)

            t_launch = (1*u.hr).to(u.s)
            cme = H.ConeCME(t_launch=t_launch, longitude=0*u.deg, latitude=model.latitude.to(u.deg),
                            width=scenario['width'], v=scenario['speed'], thickness=0.1*u.solRad)

            # Solve HUXt and get CME object
            model.solve([cme])
            cme = model.cmes[0]

            #Track the CMEs apex kinematics
            apex = compute_apex_profile(model, cme)

            # Observe the CME from L4 and L5
            l4obs = Observer(model, 60.0*u.deg, el_min=4.0, el_max=60.0, color='r', name='L4')
            l5obs = Observer(model, 300.0*u.deg, el_min=4.0, el_max=60.0, color='b', name='L5')

            observer_list = [l4obs, l5obs]

            # Plot solution when CME gets to 100rs
            id_t = np.argmin(np.abs(apex['r'] - 130*u.solRad.to(u.km)))
            plot_huxt_multi(axt[i], model.time_out[id_t], model, observer_list, add_observer=True, add_flank=True)

            # Plot the apex and geometric model kinematics profiles        
            t = apex['model_time'].values*u.s
            r = apex['r'].values*u.km
            v = apex['v'].values*u.km/u.s
            id_max = r == np.nanmax(r)
            r[id_max] = np.NaN*r.unit
            v[id_max] = np.NaN*v.unit
            axm[i].plot(t.to(u.d), r.to(u.solRad), 'k-', label='Apex')
            axb[i].plot(t.to(u.d), v, 'k-', label='Apex')

            for obs in [l4obs, l5obs]:
                t = obs.sse['model_time'].values*u.s
                r = obs.sse['r_apex'].values*u.km
                v = obs.sse['v_apex'].values*u.km/u.s
                axm[i].plot(t.to(u.d), r.to(u.solRad), '-', color=obs.color, label=obs.name+" SSE", linewidth=2)
                axb[i].plot(t.to(u.d), v, '-', color=obs.color, label=obs.name+" SSE", linewidth=2)

            arrival_stats = cme.compute_arrival_at_body('Earth')
            t_arr_stats.append(arrival_stats['t_transit'].value)
            r_arr_stats.append(arrival_stats['hit_rad'].value)

        t_max = np.max(t_arr_stats)
        r_max = np.max(r_arr_stats)
        v_max = 1500
        for a in axm:
            a.set_xlim(0, t_max)
            a.set_ylim(30, r_max)
            a.set_xticklabels([])
            a.legend(loc=4, handlelength=0.5, fontsize=14)
            a.tick_params(direction='in')

        for a in axb:
            a.set_xlim(0, t_max)
            a.set_ylim(300, v_max)
            a.legend(loc=1, handlelength=0.5, fontsize=14)
            a.tick_params(direction='in')
            a.set_xlabel('Model time (days)')

        # Remove uncessary y labels
        for am, ab in zip(axm[1:], axb[1:]):
            am.set_yticklabels([])
            ab.set_yticklabels([])

        axm[0].set_ylabel('Apex distance (Rs)')
        axb[0].set_ylabel('Apex speed (km/s)')

        for a, lab in zip(axt, ['Average', 'Fast', 'Extreme']):
            a.text(0.25, 0.9, lab, horizontalalignment='center', transform=a.transAxes, fontsize=18 ,bbox=dict(facecolor='white'))

        fig.subplots_adjust(left=0.1, bottom=0.06, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
        
        # Save 
        proj_dirs = get_project_dirs()        
        fig_name = 'two_observers_sse_{}.pdf'.format(wind_type)
        fig_path = os.path.join(proj_dirs['paper_figures'], fig_name)
        fig.savefig(fig_path, format='pdf')
        plt.close('all')
        
    return

    
def plot_kinematic_example_multi_model():

    for wind_type in ['structured', 'uniform']:

        # Set up HUXt to run all three scenarios in uniform wind, then plot the kinematics out.
        fig, ax = plt.subplots(figsize=(10, 10))
        ax1 = plt.subplot(331, projection='polar')
        ax2 = plt.subplot(332, projection='polar')
        ax3 = plt.subplot(333, projection='polar')
        axt = [ax1, ax2, ax3]
        ax4 = plt.subplot(334)
        ax5 = plt.subplot(335)
        ax6 = plt.subplot(336)
        axm = [ax4, ax5, ax6]
        ax7 = plt.subplot(337)
        ax8 = plt.subplot(338)
        ax9 = plt.subplot(339)
        axb = [ax7, ax8, ax9]

        cme_scenarios = load_cme_scenarios()

        t_arr_stats = []
        r_arr_stats = []

        for i, s_key in enumerate(['average', 'fast', 'extreme']):

            scenario = cme_scenarios[s_key]
            start_time = Time('2008-06-10T00:00:00')
            cr_num = np.fix(sn.carrington_rotation_number(start_time))
            ert = H.Observer('EARTH', start_time)
            vr_in = HIN.get_MAS_long_profile(cr_num, ert.lat.to(u.deg))
            if wind_type == 'uniform':
                vr_in = vr_in*0 + 400*(u.km/u.s)

            model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                           lon_start=300*u.deg, lon_stop=60*u.deg, simtime=5*u.day, dt_scale=1)

            t_launch = (1*u.hr).to(u.s)
            cme = H.ConeCME(t_launch=t_launch, longitude=0*u.deg, latitude=model.latitude.to(u.deg),
                            width=scenario['width'], v=scenario['speed'], thickness=0.1*u.solRad)

            # Solve HUXt and get CME object
            model.solve([cme])
            cme = model.cmes[0]

            #Track the CMEs apex kinematics
            apex = compute_apex_profile(model, cme)

            # Observe the CME from L4 and L5
            l5obs = Observer(model, 300.0*u.deg, el_min=4.0, el_max=60.0, color='b', name='L5')

            observer_list = [l5obs]

            # Plot solution when CME gets to 100rs
            id_t = np.argmin(np.abs(apex['r'] - 130*u.solRad.to(u.km)))
            plot_huxt_multi(axt[i], model.time_out[id_t], model, observer_list, add_observer=True, add_flank=True)

            # Plot the apex and geometric model kinematics profiles        
            t = apex['model_time'].values*u.s
            r = apex['r'].values*u.km
            v = apex['v'].values*u.km/u.s
            id_max = r == np.nanmax(r)
            r[id_max] = np.NaN*r.unit
            v[id_max] = np.NaN*v.unit
            axm[i].plot(t.to(u.d), r.to(u.solRad), 'k-', label='Apex')
            axb[i].plot(t.to(u.d), v, 'k-', label='Apex')

            names = ['FP', 'HM', 'SSE', 'ElCon']
            for j, obs in enumerate([l5obs.fp, l5obs.hm, l5obs.sse, l5obs.elp]):
                t = obs['model_time'].values*u.s
                r = obs['r_apex'].values*u.km
                v = obs['v_apex'].values*u.km/u.s
                axm[i].plot(t.to(u.d), r.to(u.solRad), '-', color=Dark2_5.mpl_colors[j], label=names[j], linewidth=2)
                axb[i].plot(t.to(u.d), v, '-', color=Dark2_5.mpl_colors[j], label=names[j], linewidth=2)

            arrival_stats = cme.compute_arrival_at_body('Earth')
            t_arr_stats.append(arrival_stats['t_transit'].value)
            r_arr_stats.append(arrival_stats['hit_rad'].value)

        t_max = np.max(t_arr_stats)
        r_max = np.max(r_arr_stats)
        v_max = 1500
        for a in axm:
            a.set_xlim(0, t_max)
            a.set_ylim(30, r_max)
            a.set_xticklabels([])
            a.legend(loc=4, handlelength=0.5, fontsize=14)
            a.tick_params(direction='in')

        for a in axb:
            a.set_xlim(0, t_max)
            a.set_ylim(300, v_max)
            a.legend(loc=1, handlelength=0.5, fontsize=14)
            a.tick_params(direction='in')
            a.set_xlabel('Model time (days)')

        # Remove uncessary y labels
        for am, ab in zip(axm[1:], axb[1:]):
            am.set_yticklabels([])
            ab.set_yticklabels([])

        axm[0].set_ylabel('Apex distance (Rs)')
        axb[0].set_ylabel('Apex speed (km/s)')

        for a, lab in zip(axt, ['Average', 'Fast', 'Extreme']):
            a.text(0.25, 0.9, lab, horizontalalignment='center', transform=a.transAxes, fontsize=18 ,bbox=dict(facecolor='white'))


        fig.subplots_adjust(left=0.1, bottom=0.06, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
        
        # Save 
        proj_dirs = get_project_dirs()        
        fig_name = 'one_observer_all_geomods_{}.pdf'.format(wind_type)
        fig_path = os.path.join(proj_dirs['paper_figures'], fig_name)
        fig.savefig(fig_path, format='pdf')
        plt.close('all')
    
    return


def plot_kinematics_subset():
    """
    Function to produce a 4x3 plot showing examples of the comparison of the 
    true CME apex and geometricially modelled CME apex for a range of geometric models
    and observer locations
    """
    
    data_path = "C:/Users/yq904481/research/repos/GeoModelUncertainty/data/out_data/CME_scenarios_simulation_results.hdf5"
    data = h5py.File(data_path, 'r')

    fig, ax = plt.subplots(4, 3, figsize=(9, 11.5))
    axr = ax.ravel()

    scenario = data['average']
    rs = 1*u.AU.to(u.km)

    obs_keys  = ['Observer 350.00', 'Observer 310.00', 'Observer 270.00']
    gm_keys = ['fp', 'hm', 'sse', 'elp']
    gm_cols = {gk:Dark2_5.mpl_colors[i] for i, gk in enumerate(gm_keys)}
    gm_labels = {'fp':'FP', 'hm':'HM', 'sse':'SSE', 'elp':'ElCon'}

    for i, gk in enumerate(gm_keys):

        for j, ok in enumerate(obs_keys):

            for run_key, run in scenario.items():

                t = run['cme_apex/model_time'][()]
                r = run['cme_apex/r'][()]

                gm_r_path = "/".join(['observers', ok, gk,'r_apex'])
                rf = run[gm_r_path][()]
                ax[i, j].plot(r/rs, rf/rs, '-', color=gm_cols[gk])

            # Label the observer longitude
            lon = ok.split(' ')[1]
            lon = lon.split('.')[0]
            label = "HEE Lon ${}^\circ$".format(lon)
            ax[i,j].text(0.95, 0.05, label, horizontalalignment='right', fontsize=14, transform=ax[i, j].transAxes)

    for a in axr:
        a.plot([0,1], [0,1],'k--', linewidth=2)
        a.set_xlim(0.1, 0.9)
        a.set_ylim(0.1, 0.9)
        a.set_aspect('equal')
        ticks = np.arange(0.2, 1.0, 0.2)
        a.set_xticks(ticks)
        a.set_yticks(ticks)
        a.tick_params(direction='in')
        #break

    for i, rl in enumerate(['A', 'B', 'C']):
        for j in range(4):
            label = "{}{:d})".format(rl, j+1)
            ax[j, i].text(0.025, 0.925, label, horizontalalignment='left', fontsize=14, transform=ax[j, i].transAxes)

    for ac in ax[:-1]:
        for a in ac:
            a.set_xticklabels([])

    for ac in ax:
        for a in ac[1:]:
            a.set_yticklabels([])

    for a in ax[-1, :]:
        a.set_xlabel('HUXt apex (Au)')

    for a, gk in zip(ax[:, 0], gm_keys):
        label = gk.upper() + ' apex (Au)'
        a.set_ylabel(gm_labels[gk])

    fig.subplots_adjust(left=0.085, bottom=0.05, right=0.99, top=0.99, wspace=0.015, hspace=0.015)
    
    # Save and close
    proj_dirs = get_project_dirs()        
    fig_name = 'kinematics_subset.pdf'
    fig_path = os.path.join(proj_dirs['paper_figures'], fig_name)
    fig.savefig(fig_path, format='pdf')
    plt.close('all')
    data.close()
    return


def plot_error_series_and_distribution():
    """
    Function to produce a plot showing the radial evolution of the errors of 
    geometric modelling and the distriubtion of these integrated errors for
    the 100 runs. 
    """
    
    data_path = "C:/Users/yq904481/research/repos/GeoModelUncertainty/data/out_data/CME_scenarios_simulation_results.hdf5"
    data = h5py.File(data_path, 'r')
    
    gm_labels = {'fp':'FP', 'hm':'HM', 'sse':'SSE', 'elp':'ElCon'}

    for err_type in ['mean', 'mean_abs']:

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        scenario = data['average']
        scale = 1*u.AU.to(u.km)

        r_path = "/".join(['cme_apex','r'])
        obs = 'Observer 270.00'
        gm = 'elp'
        rg_path = "/".join(['observers', obs, gm, 'r_apex'])

        r_int_limit = 0.5
        sum_err = []
        lines = []

        for r_key, run in scenario.items():

            # Get the HUXt and GM apex distances
            r = run[r_path][()]/scale
            rg = run[rg_path][()]/scale

            # Find only the valid values in each and compute the error and absolute error
            id_good = np.isfinite(r) & np.isfinite(rg)
            r = r[id_good]
            rg = rg[id_good]
            err = rg - r
            if err_type == 'mean_abs':
                err = np.abs(err)

            # Integrate the errors up to r_int_limit, save to array
            id_sub = r <= r_int_limit
            err_intg = np.trapz(err[id_sub], r[id_sub])
            sum_err.append(err_intg)

            # Update the plot with these data
            ax[0].plot(r, err, '-', color='darkgrey', zorder=1)
            h = ax[0].plot(r[id_sub], err[id_sub], '-', color='darkgrey', zorder=1)
            lines.append(h[0])

        # Add integration limit to axes
        ax[0].vlines(r_int_limit, -1, 1, colors='k', linestyles=':', linewidth=2)

        # Update line colors according to integrated error
        sum_err = np.array(sum_err)
        if err_type == 'mean_abs':
            e_min = 0
            e_max = sum_err.max()
            norm = mpl.colors.Normalize(vmin=e_min, vmax=e_max)
            cmap = mpl.cm.viridis
            for h, err in zip(lines, sum_err):
                h.set_color(cmap(norm(err)))
                
            ylabel = "{} abs. apex error, $\eta _{{{}}}$, (Au)".format(gm_labels[gm], gm_labels[gm])
            ax[0].set_ylabel(ylabel)

        elif err_type == 'mean':
            e_min = sum_err.min()
            e_max = sum_err.max()
            e_supermax = np.max([np.abs(e_min), e_max])
            norm = mpl.colors.Normalize(vmin=-e_supermax, vmax=e_supermax)
            cmap = mpl.cm.PiYG
            for h, e in zip(lines, sum_err):
                h.set_color(cmap(norm(e)))
            
            ylabel = "{} apex error, $\epsilon _{{{}}}$, (Au)".format(gm_labels[gm], gm_labels[gm])
            ax[0].set_ylabel(ylabel)


        ax[0].set_ylim(-0.1, 0.25)
        ax[0].set_xlabel('HUXt apex (Au)')
        
        

        # Add histogram to last panel
        bins = np.arange(-0.01, 0.02, 0.002)
        ax[1].hist(sum_err, bins, density=True, color='skyblue')
        # Add mean error
        avg_err = np.mean(sum_err)
        if err_type == 'mean_abs':
            ax[1].vlines(avg_err, 0, 400, colors='r', linestyles='--', linewidth=2, label='$\\langle H_{{{}}} \\rangle$'.format(gm_labels[gm]))
            ax[1].set_xlabel('Integrated {} abs. apex error, $H_{{{}}}$'.format(gm_labels[gm], gm_labels[gm]))    
        elif err_type == 'mean':
            ax[1].vlines(avg_err, 0, 400, colors='r', linestyles='--', linewidth=2, label='$\\langle E_{{{}}} \\rangle$'.format(gm_labels[gm]))
            ax[1].set_xlabel('Integrated {} apex error, $E_{{{}}}$'.format(gm_labels[gm], gm_labels[gm]))
        
        # Format axes
        ax[1].set_xlim(-0.0045, 0.019)
        ax[1].set_ylim(0, 175)        
        ax[1].set_ylabel('Density')    
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position('right')
        ax[1].legend()

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.925, top=0.85, wspace=0.01)

        # Add the colorbar to ax[0] for the errors
        pos = ax[0].get_position()
        dw = 0.005
        dh = 0.005
        left = pos.x0 + dw
        bottom = pos.y1 + dh
        wid = pos.width - 2 * dw
        hi_cbaxes = fig.add_axes([left, bottom, wid, 0.02])
        smp = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar1 = fig.colorbar(smp, cax=hi_cbaxes, orientation='horizontal')
        if err_type == 'mean_abs':
            cbar1.ax.set_xlabel('Integrated {} abs. apex error, $H_{{{}}}$'.format(gm_labels[gm], gm_labels[gm]))    
        elif err_type == 'mean':
            cbar1.ax.set_xlabel('Integrated {} apex error, $E_{{{}}}$'.format(gm_labels[gm], gm_labels[gm]))
            
        cbar1.ax.xaxis.tick_top()
        cbar1.ax.xaxis.set_label_position('top')
        
        # save and close
        proj_dirs = get_project_dirs()        
        fig_name = "{}_integration_example.pdf".format(err_type)
        fig_path = os.path.join(proj_dirs['paper_figures'], fig_name)
        fig.savefig(fig_path, format='pdf')
        plt.close('all')
            
    data.close()
    return


def plot_error_vs_longitude():
    """
    Function to produce a plot the mean integrated path error and mean absolute integrated path error
    for each geometric model as a function of observer longitude from Earth.
    """
    
    data_path = "C:/Users/yq904481/research/repos/GeoModelUncertainty/data/out_data/CME_scenarios_simulation_results.hdf5"
    data = h5py.File(data_path, 'r')
    
    scale = 1*u.AU.to(u.km)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    scenario_keys = ['average', 'fast', 'extreme']
    gm_keys = ['fp', 'hm', 'sse', 'elp']
    gm_cols = {gk:Dark2_5.mpl_colors[i] for i, gk in enumerate(gm_keys)}
    gm_style = {'fp':'x-', 'hm':'s-', 'sse':'^-', 'elp':'d-' }
    gm_labels = {'fp':'FP', 'hm':'HM', 'sse':'SSE', 'elp':'ElCon'}
    observer_lons = data['average/run_000/observer_lons'][()]
    observer_keys = ["Observer {:3.2f}".format(l) for l in observer_lons]

    for i, sk in enumerate(scenario_keys):

        scenario = data[sk]

        mean_results = np.zeros((len(gm_keys), len(observer_keys)))
        mean_abs_results = np.zeros((len(gm_keys), len(observer_keys)))

        mean_unc = np.zeros((len(gm_keys), len(observer_keys)))
        mean_abs_unc = np.zeros((len(gm_keys), len(observer_keys)))

        for j, gk in enumerate(gm_keys):

            for k, ok in enumerate(observer_keys):


                r_path = "/".join(['cme_apex','r'])
                rg_path = "/".join(['observers', ok, gk, 'r_apex'])
                r_int_limit = 0.5
                sum_err = []
                sum_abs_err = []

                for r_key, run in scenario.items():

                    # Get the HUXt and GM apex distances
                    r = run[r_path][()]/scale
                    rg = run[rg_path][()]/scale

                    # Find only the valid values in each and compute the error and absolute error
                    id_good = np.isfinite(r) & np.isfinite(rg)
                    r = r[id_good]
                    rg = rg[id_good]
                    err = rg - r
                    abs_err = np.abs(err)

                    # Integrate the errors up to r_int_limit, save to array
                    id_sub = r <= r_int_limit
                    err_intg = np.trapz(err[id_sub], r[id_sub])
                    sum_err.append(err_intg)

                    abs_err_intg = np.trapz(abs_err[id_sub], r[id_sub])
                    sum_abs_err.append(abs_err_intg)

                mean_results[j, k] = np.mean(sum_err)
                mean_abs_results[j, k] = np.mean(sum_abs_err)

                mean_unc[j, k] = 2*st.sem(sum_err)
                mean_abs_unc[j, k] = 2*st.sem(sum_abs_err)

            ax[0, i].errorbar(observer_lons, mean_results[j, :], yerr=mean_unc[j, :], fmt=gm_style[gk], color=gm_cols[gk], linewidth=2, label=gm_labels[gk])
            ax[1, i].errorbar(observer_lons, mean_abs_results[j, :], yerr=mean_abs_unc[j, :], fmt=gm_style[gk], color=gm_cols[gk], linewidth=2, label=gm_labels[gk])

        ax[0, i].text(0.3, 0.92, sk.capitalize(), horizontalalignment='left', fontsize=18, transform=ax[0, i].transAxes)
        ax[1, i].text(0.3, 0.92, sk.capitalize(), horizontalalignment='left', fontsize=18, transform=ax[1, i].transAxes)

    for a in ax.ravel():
        a.set_xlim(265, 355)
        a.legend(loc=2, handlelength=1.0)
        a.tick_params(direction='in')

    for a in ax[0, :]:
        a.set_ylim(-0.03, 0.11)
        a.set_xticklabels([])

    for a in ax[:, 1:].ravel():
        a.set_yticklabels([])

    for a in ax[1, :]:
        a.set_ylim(-0.0, 0.11)
        a.set_xlabel('Observer Longitude (deg)')

    ax[0, 0].set_ylabel('Mean integrated error, $\\langle E \\rangle$')
    ax[1, 0].set_ylabel('Mean integrated absolute error, $\\langle H \\rangle$')

    fig.subplots_adjust(left=0.07, bottom=0.065, right=0.99, top=0.99, wspace=0.02, hspace=0.02)    
    
    # save and close
    proj_dirs = get_project_dirs()        
    fig_name = 'err_vs_lon.pdf'
    fig_path = os.path.join(proj_dirs['paper_figures'], fig_name)
    fig.savefig(fig_path, format='pdf')
    plt.close('all')
    data.close()
    return

def plot_elevohi_error_violins():

    project_dirs = get_project_dirs()
    data_avg = pd.read_csv(project_dirs['ELEvoHI_average'], delim_whitespace=True)
    data_fst = pd.read_csv(project_dirs['ELEvoHI_fast'], delim_whitespace=True)
    data_ext = pd.read_csv(project_dirs['ELEvoHI_extreme'], delim_whitespace=True)

    observer_lons = np.sort(data_avg['sep'].unique())

    labels = ['Average', 'Fast', 'Extreme']
    colors = [mpl.cm.tab10.colors[i] for i in range(len(labels))]
    color_dict = {lab:mpl.cm.tab10.colors[i] for i, lab in enumerate(labels)}


    fig, ax = plt.subplots(3, 2, figsize=(12,10))
    axr = ax.ravel()
    axlft = ax[:, 0]
    axrgt = ax[:, 1]

    mae_handles = []
    me_handles = []
    for i, data in enumerate([data_avg, data_fst, data_ext]):

        mae_data = []
        me_data = []

        for ol in observer_lons:

            id_obs = data['sep'] == ol
            mae_data.append(data.loc[id_obs, 'mae_t'])
            me_data.append(data.loc[id_obs, 'me_t'])


        h = axlft[i].violinplot(mae_data, positions=observer_lons, widths=5, showmeans=True)
        mae_handles.append(h)

        h = axrgt[i].violinplot(me_data, positions=observer_lons, widths=5, showmeans=True)
        me_handles.append(h)

    for a in axr:
        a.set_xlim(261, 359)    
        a.tick_params(direction='in')

    for a, h, label in zip(axlft, mae_handles, ['Average', 'Fast', 'Extreme']):
        a.set_ylim(0, 38)
        a.set_ylabel('$|\\Delta t_{eeh}|$ (hours)')
        a.text(0.99, 0.9, label, horizontalalignment='right', transform=a.transAxes, fontsize=18)

        # Color the violins
        for pc in h['bodies']:
            pc.set_facecolor(color_dict[label])
            pc.set_edgecolor('black')

        for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = h[partname]
            vp.set_edgecolor(color_dict[label])

    for a, h, label in zip(axrgt, me_handles, ['Average', 'Fast', 'Extreme']):
        a.set_ylim(-38, 20)
        a.set_ylabel('$\\Delta t_{eeh}$ (hours)')
        a.yaxis.tick_right()
        a.yaxis.set_label_position('right')
        a.text(0.01, 0.9, label, horizontalalignment='left', transform=a.transAxes, fontsize=18)

        # Color the violins
        for pc in h['bodies']:
            pc.set_facecolor(color_dict[label])
            pc.set_edgecolor('black')

        for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = h[partname]
            vp.set_edgecolor(color_dict[label])


    for a in ax[0:2, :].ravel():
        a.set_xticklabels([])

    ax[2, 0].set_xlabel('Observer HEE Longitude')
    ax[2, 1].set_xlabel('Observer HEE Longitude')

    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.93, top=0.98, hspace=0.02, wspace=0.015)
    fig_name = 'mae_me_violins_vs_lon.pdf'
    fig_path = os.path.join(project_dirs['paper_figures'], fig_name)
    fig.savefig(fig_path)
    return
    

def plot_elevohi_mean_errors():
        
    project_dirs = get_project_dirs()
    data_avg = pd.read_csv(project_dirs['ELEvoHI_average'], delim_whitespace=True)
    data_fst = pd.read_csv(project_dirs['ELEvoHI_fast'], delim_whitespace=True)
    data_ext = pd.read_csv(project_dirs['ELEvoHI_extreme'], delim_whitespace=True)

    fig, ax = plt.subplots(1, 2, figsize=(14,7))
    labels = ['Average', 'Fast', 'Extreme']
    colors = {lab:mpl.cm.tab10.colors[i] for i, lab in enumerate(labels)}
    fmt = ['s-', 'x--', '^:']
    for c, data in enumerate([data_avg, data_fst, data_ext]):


        observer_lons = np.sort(data['sep'].unique())

        mae = np.zeros(observer_lons.shape)
        mae_sem = np.zeros(observer_lons.shape)
        me = np.zeros(observer_lons.shape)
        me_sem = np.zeros(observer_lons.shape)
        for i, ol in enumerate(observer_lons):

            id_obs = data['sep'] == ol
            me[i] = data.loc[id_obs, 'me_t'].mean()
            me_sem[i] = 2*data.loc[id_obs, 'me_t'].sem()
            mae[i] = data.loc[id_obs, 'mae_t'].mean()
            mae_sem[i] = 2*data.loc[id_obs, 'mae_t'].sem()


        ax[0].errorbar(observer_lons, mae, yerr=mae_sem, fmt=fmt[c], color=colors[labels[c]], label=labels[c])
        ax[1].errorbar(observer_lons, me, yerr=me_sem, fmt=fmt[c], color=colors[labels[c]], label=labels[c])

    for a in ax:
        a.set_xlim(265, 355)
        a.set_xlabel('Observer longitude (deg)')
        a.legend(loc='upper center')
        a.tick_params(direction='in')

    ax[0].set_ylabel('Mean absolute arrival time error, $\\langle |\\Delta t_{eeh}| \\rangle _{bws}$, (hours)')

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position('right')
    ax[1].set_ylabel('Mean arrival time error, $\\langle \\Delta t_{eeh} \\rangle _{bws}$, (hours)')

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.93, top=0.99, wspace=0.01)
    fig_name = 'mean_err_vs_lon.pdf'
    fig_path = os.path.join(project_dirs['paper_figures'], fig_name)
    fig.savefig(fig_path)
    return


def get_observer_los(ro, lo, el):
    """
    Function to compute a matplotlib patch to higlight an observers field of view. 
    ro = radius of observer (in solRad)
    lo = longitude of observer (in rad)
    el_min = minimum elongation of the field of view
    el_max = maximum elongation of the field of view
    """
    xo = ro*np.cos(lo)
    yo = ro*np.sin(lo)
            
    rp = ro*np.tan(el)
    if (lo < 0*u.rad) | (lo > np.pi*u.rad):
        lp = lo + 90*u.deg
    else:
        lp = lo - 90*u.deg

    if lp > 2*np.pi*u.rad:
        lp = lp - 2*np.pi*u.rad

    xp = rp*np.cos(lp)
    yp = rp*np.sin(lp)

    # Wolfram equations for intersection of line with circle
    rf = 475*u.solRad  # set this to a large value outside axis lims so FOV shading spans model domain
    dx = (xp - xo)
    dy = (yp - yo)
    dr = np.sqrt(dx**2 + dy**2)
    det = (xo*yp - xp*yo)
    discrim = np.sqrt((rf*dr)**2 - det**2)

    if (lo < 0*u.rad) | (lo > np.pi*u.rad):
        xf = (det*dy + np.sign(dy)*dx*discrim) / (dr**2)
        yf = (-det*dx + np.abs(dy)*discrim) / (dr**2)
    else:
        xf = (det*dy - np.sign(dy)*dx*discrim) / (dr**2)
        yf = (-det*dx - np.abs(dy)*discrim) / (dr**2)

    lf = np.arctan2(yf, xf)
    
    return lf, rf


def plot_geomodel_schematic():
    """
    Figure 1 of the article showing a schematic of the different classes of 
    geometric model for each CME scenario.
    """
    
    cme_scenarios = load_cme_scenarios()

    gm_keys = ['fp', 'hm', 'sse', 'elp']
    gm_cols = {gk:Dark2_5.mpl_colors[i] for i, gk in enumerate(gm_keys)}
    gm_style = {'fp':'*', 'hm':'-', 'sse':'--', 'elp':':' }

    fig, ax = plt.subplots(1, 3, figsize=(15, 10), subplot_kw={"projection": "polar"})

    for i, scenario_key in enumerate(['average', 'fast', 'extreme']):

        scenario = cme_scenarios[scenario_key]

        start_time = Time('2008-06-10T00:00:00')

        ert = H.Observer('EARTH', start_time)
        lons, dlon, nlon = H.longitude_grid()
        vr_in = 400*(u.km/u.s)*np.ones(lons.size)
        cr_num = np.fix(sn.carrington_rotation_number(start_time))

        model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                       lon_start=300*u.deg, lon_stop=60*u.deg, simtime=3*u.day, dt_scale=1)

        t_launch = (1*u.hr).to(u.s)
        cme = H.ConeCME(t_launch=t_launch, longitude=0*u.deg, latitude=model.latitude.to(u.deg),
                        width=scenario['width'], v=scenario['speed'], thickness=0.1*u.solRad)

        model.solve([cme])
        cme = model.cmes[0]

        # Observe the CME from L4 and L5
        # Add on Earth, Observer, and Observers Line of sight of flank
        ert = model.get_observer('EARTH')
        ax[i].plot(ert.lon[0], ert.r[0], 'co', markersize=16, label='Earth')            

        observer = Observer(model, 300.0*u.deg, el_min=4.0, el_max=60.0, color='b', name='L5')
        ax[i].plot(observer.lon[0], observer.r[0], 's', color=observer.color, markersize=16, label='Observer')

        # Get the LOS
        r_obs = observer.r[0]
        lon_obs = observer.lon[0]
        el = np.deg2rad(40)*u.rad
        lf, rf = get_observer_los(r_obs, lon_obs, el)
        ax[i].plot([lon_obs.value, lf.value], [r_obs.value, rf.value], '--', color=observer.color, zorder=0)

        # Add on the geometric models
        #Fixed Phi
        lon_apex, r_apex = fixed_phi(cme, r_obs, lon_obs, el)
        ax[i].plot(lon_apex.value, r_apex.value, '*', markersize=14, color=gm_cols['fp'])

        # Harmonic mean
        lon_apex, r_apex, r_center, radius = harmonic_mean(cme, r_obs, lon_obs, el)
        x_c = r_center*np.cos(lon_apex)
        y_c = r_center*np.sin(lon_apex)
        hmm_circ = plt.Circle((x_c.value, y_c.value), radius.value, color=gm_cols['hm'], fill=False, linewidth=3, linestyle=gm_style['hm'], zorder=0, transform=ax[i].transData._b)
        ax[i].add_artist(hmm_circ)

        # Self similar expansion
        lon_apex, r_apex, r_center, radius = self_similar_expansion(cme, r_obs, lon_obs, el)
        x_c = (r_center*np.cos(lon_apex)).value
        y_c = (r_center*np.sin(lon_apex)).value
        sse_circ = plt.Circle((x_c, y_c), radius.value, color=gm_cols['sse'], fill=False, linewidth=3, linestyle=gm_style['sse'], zorder=1, transform=ax[i].transData._b)
        ax[i].add_artist(sse_circ)

        # ElCon
        lon_apex, r_apex, r_center, r_a, r_b = elcon(cme, r_obs, lon_obs, el)
        x_c = (r_center*np.cos(lon_apex)).value
        y_c = (r_center*np.sin(lon_apex)).value
        l_c = cme.longitude.value
        elp_elip = mpl.patches.Ellipse(xy=[x_c, y_c], width=2*r_b.value, height=2*r_a.value, angle=l_c, color=gm_cols['elp'], fill=False,
                                       linewidth=3, linestyle=gm_style['elp'], zorder=2, transform=ax[i].transData._b)
        ax[i].add_artist(elp_elip)


    for a, lab in zip(ax, ['Average', 'Fast', 'Extreme']):
        a.set_xlim(-np.pi/2, np.pi/2)
        a.set_ylim(0, 240)
        a.set_yticklabels([])
        a.set_xticklabels([])
        a.patch.set_facecolor('whitesmoke')

        # Workaround to include patches in the legend
        a.plot([], [], gm_style['fp'], markersize=14, color=gm_cols['fp'], label='FP')
        a.plot([], [], linestyle=gm_style['hm'], color=gm_cols['hm'], linewidth=3, label='HM')
        a.plot([], [], linestyle=gm_style['sse'], color=gm_cols['sse'], linewidth=3, label='SSE')
        a.plot([], [], linestyle=gm_style['elp'], color=gm_cols['elp'], linewidth=3, label='ElCon')

        a.legend(loc='lower left', bbox_to_anchor=(0.525, 0.0), framealpha=1.0)

        # Add on the angles.
        a.plot([0, observer.lon[0].value], [0, observer.r[0].value], 'k-', zorder=1)
        a.plot([0, 0], [0, observer.r[0].value], 'k-', zorder=1)

        # Label elon
        a.text(observer.lon[0].value+0.025, 0.88*observer.r[0].value, "$\\epsilon$", horizontalalignment='left', fontsize=20)
        x = (observer.r[0]*np.cos(observer.lon[0])).value
        y = (observer.r[0]*np.sin(observer.lon[0])).value
        lon = observer.lon[0]
        beta = np.abs((lon.to(u.deg).value - 360))
        elo = el.to(u.deg).value
        psi = 180 - beta - elo
        theta1 = psi
        theta2 = psi+elo
        arc = mpl.patches.Arc((x, y), 75, 75, theta1=theta1, theta2=theta2, fill=False, edgecolor='k', alpha=1.0, linewidth=2, transform=a.transData._b, zorder=1)
        a.add_artist(arc)

        # Label beta
        a.text(observer.lon[0].value+0.1, 0.1*observer.r[0].value, "$\\beta$", horizontalalignment='left', fontsize=18) 
        theta1 = -beta
        theta2 = 0
        arc = mpl.patches.Arc((0, 0), 75, 75, theta1=theta1, theta2=theta2, fill=False, edgecolor='k', alpha=1.0, linewidth=2, transform=a.transData._b, zorder=1)
        a.add_artist(arc)

        a.text(0.26, 0.95, lab, horizontalalignment='left', transform=a.transAxes, fontsize=24 ,bbox=dict(facecolor='white'))

    fig.subplots_adjust(left=-0.15, bottom=-0.0, right=1.15, top=0.99, wspace=-0.45)
    project_dirs = get_project_dirs()
    fig_name = 'geomodel_schematic.pdf'
    fig_path = os.path.join(project_dirs['paper_figures'], fig_name)
    fig.savefig(fig_path)
    return


if __name__ == "__main__":
    
    build_cme_scenarios()
    produce_huxt_ensemble(n=100)
    plot_kinematics_example_multi_observer()
    plot_kinematic_example_multi_model()
    plot_kinematics_subset()
    plot_error_series_and_distribution()
    plot_error_vs_longitude()
    plot_elevohi_error_violins()
    plot_elevohi_mean_errors()
    plot_geomodel_schematic()