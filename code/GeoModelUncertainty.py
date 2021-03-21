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
        id_good = (self.cme_flank['el'] >= self.el_min) & (self.cme_flank['el'] <= self.el_max)
        self.cme_flank = self.cme_flank[id_good]
        self.fp = self.fp[id_good]
        self.hm = self.hm[id_good]
        self.sse = self.sse[id_good]
        self.elp = self.elp[id_good]
        
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
    time_min = Time(t_start, format='isot')
    time_max = Time(t_stop, format='isot')
    start_times = np.random.uniform(time_min.jd, time_max.jd, n_samples)
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
    obs_group.create_dataset('time', data=observer.time.jd)
    for name, param in zip(['r', 'lon', 'lat'], [observer.r, observer.lon, observer.lat]):
        dset = obs_group.create_dataset(name+'_obs', data=param.value)
        dset.attrs['unit'] = param.unit.to_string()

    # Now save the elongation profile and flank coordinate.
    for key, unit in zip(['model_time', 'el', 'r', 'lon'], [u.s, u.deg, u.km, u.rad]):
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

            # Setup group to hold all observers 
            all_observers_group = run_group.create_group('observers')
            dset = all_observers_group.create_dataset("observer_lons", data=observer_longitudes.value)
            dset.attrs['unit'] = observer_longitudes.unit.to_string()
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

if __name__ == "__main__":
    
    build_cme_scenarios()
    produce_huxt_ensemble(n=2)