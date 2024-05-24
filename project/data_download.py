import cdsapi
import netCDF4 as nc
import numpy as np
import os

supported_era5_variables = [
    'total_precipitation', 
    '10m_v_component_of_wind', 
    '10m_u_component_of_wind'
    ]

def download_era5_data(c, years='2019', variable='total_precipitation'):
    os.makedirs('download/era5', exist_ok=True)
    var_name = ''
    if variable == 'total_precipitation':
        var_name = 'tp'
    elif variable == '10m_v_component_of_wind':
        var_name = 'v10'
    elif variable == '10m_u_component_of_wind':
        var_name = 'u10'
    else:
        print("Variable not supported!")
        print("Supported variables:", supported_era5_variables)
        return
    
    download_name = ""
    if isinstance(years, list):
        download_name = 'download/era5/' + str(var_name) + '-'  + str(years[0]) + '_' + str(years[-1]) +'.nc'
    else:
        download_name = 'download/era5/' + str(var_name) + '-'  + str(years) +'.nc'

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'year': years,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
            'area': [
                52, 2, 49,
                7,
            ],
        },
        download_name
        )

def filter_cerra_data(file_path, variable, remove=False, north=52, south=49, west=2, east=7):
    if west < 0:
        west = 360 + west
    
    var_name = ''
    if variable == 'total_precipitation':
        var_name = 'tp'
    elif variable == '10m_wind_speed':
        var_name = 'si10'
    else:
        print("Variable not supported!")
        print("Supported variables: total_precipitation, 10m_wind_speed")
        return
    
    year = file_path.split('/')[-1].split('-')[1].split('.')[0]

    data = nc.Dataset(file_path)

    lat_error = np.absolute(data['latitude'][:] - north)
    lon_error = np.absolute(data['longitude'][:] - west)
    cord_error = lat_error + lon_error
    top_index = np.unravel_index(np.argmin(cord_error), lat_error.shape)

    lat_error = np.absolute(data['latitude'][:] - south)
    lon_error = np.absolute(data['longitude'][:] - east)
    cord_error = lat_error + lon_error
    bottom_index = np.unravel_index(np.argmin(cord_error), lat_error.shape)

    filter_lat = np.array(data['latitude'][bottom_index[0]:top_index[0], top_index[1]:bottom_index[1]])
    filter_lon = np.array(data['longitude'][bottom_index[0]:top_index[0], top_index[1]:bottom_index[1]])
    filter_data = np.array(data[var_name][:, bottom_index[0]:top_index[0], top_index[1]:bottom_index[1]])
    datetime = np.array(data["valid_time"][:])

    if not os.path.exists('download/cerra/lat.npy'):
        np.save('download/cerra/lat.npy', filter_lat)
    if not os.path.exists('download/cerra/lon.npy'):
        np.save('download/cerra/lon.npy', filter_lon)
    if not os.path.exists('download/cerra/datetime_' + year + '.npy'):
        np.save('download/cerra/datetime_' + year + '.npy', datetime)
    file_name = file_path.split('/')[-1].split('.')[0]
    np.save('download/cerra/' + file_name + '.npy', filter_data)
    
    print("Data filtered and saved successfully!")
    if remove:
        os.remove(file_path)
        print("Original file removed successfully!")



def download_CERRA_data(c, year='2019', variable='total_precipitation'):
    os.makedirs('download/cerra', exist_ok=True)
    if variable == 'total_precipitation':
        var_name = 'tp'
        c.retrieve(
        'reanalysis-cerra-single-levels',
        {
            'format': 'netcdf',
            'variable': 'total_precipitation',
            'level_type': 'surface_or_atmosphere',
            'data_type': 'reanalysis',
            'product_type': 'forecast',
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
            'leadtime_hour': '1',
        },
        'download/cerra/' + str(var_name) + '-' + str(year) +'.nc')

    elif variable == '10m_wind_speed':
        var_name = 'si10'
        c.retrieve(
        'reanalysis-cerra-single-levels',
        {
            'format': 'netcdf',
            'year': year,
            'variable': variable,
            'level_type': 'surface_or_atmosphere',
            'data_type': 'reanalysis',
            'product_type': 'analysis',
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
        },
        'download/cerra/' + str(var_name) + '-' + str(year) +'.nc')

    else:
        print("Variable not supported!")
        print("Supported variables: total_precipitation, 10m_wind_speed")
        return


if __name__ == '__main__':
    pass
    # c = cdsapi.Client()
    # download_CERRA_data(c, '2020', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2020.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2019', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2019.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2018', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2018.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2017', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2017.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2016', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2016.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2015', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2015.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2014', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2014.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2013', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2013.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2012', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2012.nc', '10m_wind_speed')
    # download_CERRA_data(c, '2011', '10m_wind_speed')
    # filter_cerra_data('download/cerra/si10-2011.nc', '10m_wind_speed')