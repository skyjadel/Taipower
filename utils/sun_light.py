# This module calculate the relative integrated sun light density upon a surface over one day.
# The assumption is ideal: There is no clouds or any other things which blocked the sun light on this day.
# One should only import the function ##calculate_all_day_sunlight##. 

import ephem
import numpy as np
from datetime import datetime, timedelta
import pytz

strftime = datetime.strftime
strptime = datetime.strptime

class _Observer(ephem.Observer):
    def __init__(self, site_position):
        super().__init__()
        self.lon, self.lat, self.elevation = [site_position['lon'],
                                             site_position['lat'],
                                             site_position['elevation']]

def _taiwan_time_to_UTC(time_str):
    tw = pytz.timezone('Asia/Taipei')
    taiwan_t = strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=tw)
    UTC_t = taiwan_t.astimezone(pytz.utc)
    return strftime(UTC_t, '%Y-%m-%d %H:%M:%S')

def _get_sun_az_alt(site, taiwan_date, unit='degree'):
    UTC_date = _taiwan_time_to_UTC(taiwan_date)
    site.date = UTC_date
    v = ephem.Sun(site)
    if unit == 'degree':
        alt = v.alt / np.pi * 180
        az = v.az / np.pi * 180
    elif unit == 'rad':
        alt = v.alt
        az = v.az
    return alt, az

# The arguments are:
# solar_energy_site_center: A dictionary with 3 keys: 'lon'(str), 'lat'(str), 'elevation'(float). It describes the position of solar energy site.
# date(str): A string with shows the date you want to calculate with. Should be in yyyy-mm-dd form. 
# solar_cell_direction(list): A list contains 2 elements: azimuthal angle (default:180) and tilt (default:75) in deg. They should be int or float.  
def calculate_all_day_sunlight(solar_energy_site, date, solar_cell_direction = [180, 75]):
    if type(date) == str:
        date = strptime(date, "%Y-%m-%d")
    test_time = [strftime(date + timedelta(minutes=i * 10), '%Y-%m-%d %H:%M:%S') for i in range(int(1440/10))]
    sun_cell_productions = []
    az_site = solar_cell_direction[0] / 180 * np.pi
    alt_site = solar_cell_direction[1] / 180 * np.pi
    site_vector = [np.sin(az_site) * np.cos(alt_site), np.cos(az_site) * np.cos(alt_site),  np.sin(alt_site)]
    observer = _Observer(solar_energy_site)

    for t in test_time:
        alt, az = _get_sun_az_alt(observer, t, unit='rad')
        this_sun_vector = [np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt),  np.sin(alt)]
        sun_cell_productions.append(np.inner(np.array(site_vector), np.array(this_sun_vector)))
        sun_cell_production = sum([i if i > 0 else 0 for i in sun_cell_productions]) / 6
    return sun_cell_production

def calculate_daytime(site, date):
    time_interval_minutes = 1
    if type(date) == str:
        date = strptime(date, "%Y-%m-%d")
    test_time = [strftime(date + timedelta(minutes=i * time_interval_minutes), '%Y-%m-%d %H:%M:%S') for i in range(int(1440/time_interval_minutes))]
    observer = _Observer(site)

    daytime = 0
    previous_alt = -90
    for t in test_time:
        alt, _ = _get_sun_az_alt(observer, t, unit='rad')
        if alt > 0:
            daytime += time_interval_minutes
        elif previous_alt > 0:
            break
        previous_alt = alt
    return daytime / 60
