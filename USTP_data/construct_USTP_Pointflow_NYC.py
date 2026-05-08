"""

trafficstatepoin The spatial dimension is one-dimensional datasets (i.e. point-based/segment-based/area-based datasets)

The processed files are stored in ./USTP_Model
File format after alignment, filtering and preprocessing:
    traffic flow data set
   .geo：geo_id,type,coordinates
   .grid：dyna_id,type,time,row_id,column_id,inflow,outflow
   .rel：rel_id,type,origin_id,destination_id,cost


"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import folium
import re
from shapely.geometry import MultiPolygon
import numpy as np
from shapely.geometry import Point

def read_csv_flexible(path: str, **kwargs) -> pd.DataFrame:
    """
    Some released CSVs are not UTF-8 encoded. Try a small set of common encodings
    before failing to keep dataset construction reproducible across platforms.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err

def time_to_seconds_since_begin(value, begin_dt):
    """
    Convert a time value to seconds since begin_dt.

    Supported inputs:
    - ISO / 'YYYY-mm-dd HH:MM:SS(.ffffff)' timestamps
    - 'M:SS.s' / 'MM:SS.s' style offsets like '10:27.8' (minutes:seconds)
    - numeric seconds
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # offset format like 10:27.8 (minutes:seconds)
    if re.fullmatch(r"\d+:\d{1,2}(\.\d+)?", s):
        minutes_str, seconds_str = s.split(":", 1)
        return int(minutes_str) * 60.0 + float(seconds_str)

    # try datetime parsing
    try:
        dt = pd.to_datetime(s, errors="raise")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    return float((dt.to_pydatetime() - begin_dt).total_seconds())

dataframe2 = gpd.read_file('../UrbanKG_data/Meta_data/NYC/Administrative_data/Area/Area.shp')

dataframe2 = dataframe2.to_crs('EPSG:4326')
seleceted_colums2 = ['OBJECTID', 'zone', 'geometry']
area_dataframe = dataframe2[seleceted_colums2]

area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 1]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 103]
area_dataframe = area_dataframe[area_dataframe['OBJECTID'] != 104]

"""

USTP_Model 1: taxi

"""
##################################################################################################
##################################################################################################
taxi_dataframe = pd.read_csv('./Processed_data/NYC/NYC_taxi.csv')

taxi_numpy = taxi_dataframe[['start_time', 'end_time', 'start_area_id', 'end_area_id']].values

begin_time = '2020-04-01 00:00:00'
format_pattern = '%Y-%m-%d %H:%M:%S'

in_flow = np.zeros([91 * 24 * 2, 263])
out_flow = np.zeros([91 * 24 * 2, 263])

for i in tqdm(range(taxi_numpy.shape[0])):
    time_spannn_out = datetime.strptime(str(taxi_numpy[i][0]), format_pattern) - datetime.strptime(begin_time, format_pattern)
    total_seconds_out = int(time_spannn_out.total_seconds())
    time_step_out =  int(total_seconds_out / 1800)
    if 0<= time_step_out <= 4367:
        out_flow[time_step_out][ int(taxi_numpy[i][2]) - 1] += 1

    time_spannn_in = datetime.strptime(str(taxi_numpy[i][1]), format_pattern) - datetime.strptime(begin_time, format_pattern)
    total_seconds_in = int(time_spannn_in.total_seconds())
    time_step_in = int(total_seconds_in / 1800)
    if 0 <= time_step_in <= 4367:
        in_flow[time_step_in][int(taxi_numpy[i][3])- 1] += 1

begin_time = '2020-04-01T00:00:00Z'
format_pattern = '%Y-%m-%dT%H:%M:%SZ'
now = datetime.strptime(begin_time, format_pattern)

NYCTaxi20200406_dyna = []
NYCTaxi20200406_dyna.append('dyna_id,type,time,entity_id,inflow,outflow')
NYCTaxi20200406_geo = []
NYCTaxi20200406_geo.append('geo_id,type,coordinates')

dyna_id = 0
type = 'state'

for i in tqdm(range(in_flow.shape[1])):
    if i != 0 and i != 102 and i != 103:
        for j in range(in_flow.shape[0]):
            time_origin = now + timedelta(minutes = 30 * j)
            time_write = time_origin.strftime('%Y-%m-%dT%H:%M:%SZ')
            inflow = in_flow[j][i]
            outflow = out_flow[j][i]

            grid_record = str(dyna_id) + ',' + type + ',' + str(time_write) + ',' + str(i+1) + ',' + str(int(inflow)) + ',' + str(int(outflow))
            dyna_id += 1
            NYCTaxi20200406_dyna.append(grid_record)

        geo_record = str(i+1) + ',' + 'Point' + ',"[]"'
        NYCTaxi20200406_geo.append(geo_record)

with open(r'./USTP/NYC/NYCTaxi20200406/NYCTaxi20200406.dyna','w') as f1:
    for i in range(len(NYCTaxi20200406_dyna)):
        f1.write(NYCTaxi20200406_dyna[i])
        f1.write('\n')
f1.close()

with open(r'./USTP/NYC/NYCTaxi20200406/NYCTaxi20200406.geo','w') as f1:
    for i in range(len(NYCTaxi20200406_geo)):
        f1.write(NYCTaxi20200406_geo[i])
        f1.write('\n')
f1.close()

NYCTaxi20200406_rel = []
NYCTaxi20200406_rel.append('rel_id,type,origin_id,destination_id,cost')

rel_id = 0
type = 'geo'

for i in tqdm(range(area_dataframe.shape[0])):
    head_area = area_dataframe.iloc[i].geometry
    for j in range(area_dataframe.shape[0]):
        tail_area = area_dataframe.iloc[j].geometry
        distance = head_area.distance(tail_area)
        NYCTaxi20200406_rel.append(str(rel_id) + ',' +  type + ',' + str(area_dataframe.iloc[i].OBJECTID) + ',' + str(area_dataframe.iloc[j].OBJECTID)
                                   + ',' + str(distance))
        rel_id += 1

with open(r'./USTP/NYC/NYCTaxi20200406/NYCTaxi20200406.rel','w') as f1:
    for i in range(len(NYCTaxi20200406_rel)):
        f1.write(NYCTaxi20200406_rel[i])
        f1.write('\n')
f1.close()

##################################################################################################
##################################################################################################

"""

USTP_data 2: bike

"""

bike_dataframe = read_csv_flexible('./Processed_data/NYC/NYC_bike_road.csv')

for col in ['start_road_id', 'end_road_id']:
    bike_dataframe[col] = pd.to_numeric(bike_dataframe[col], errors='coerce')
bike_dataframe = bike_dataframe[bike_dataframe['start_road_id'].notna()]
bike_dataframe = bike_dataframe[bike_dataframe['end_road_id'].notna()]
bike_dataframe['start_road_id'] = bike_dataframe['start_road_id'].astype(np.int64)
bike_dataframe['end_road_id'] = bike_dataframe['end_road_id'].astype(np.int64)

taxi_numpy = bike_dataframe[['start_time', 'end_time', 'start_road_id', 'end_road_id']].values

begin_time = '2020-04-01 00:00:00.0000'
format_pattern = '%Y-%m-%d %H:%M:%S.%f'
begin_dt = datetime.strptime(begin_time, format_pattern)


unique_strat_road_id = bike_dataframe['start_road_id'].unique()
unique_end_road_id = bike_dataframe['end_road_id'].unique()
c = np.concatenate((unique_strat_road_id, unique_end_road_id))

unique_road_id = np.unique(c)

index_dict = {}
revers_index_dict = {}
for index, element in enumerate(unique_road_id):
    index_dict[index] = element
    revers_index_dict[element] = index

in_flow = np.zeros([91 * 24 * 2, len(index_dict)])
out_flow = np.zeros([91 * 24 * 2, len(index_dict)])

for i in tqdm(range(taxi_numpy.shape[0])):
    sec_out = time_to_seconds_since_begin(taxi_numpy[i][0], begin_dt)
    if sec_out is None:
        continue
    total_seconds_out = int(sec_out)
    time_step_out =  int(total_seconds_out / 1800)
    if 0<= time_step_out <= 4367:
        out_flow[time_step_out][ int(revers_index_dict[taxi_numpy[i][2]]) ] += 1

    sec_in = time_to_seconds_since_begin(taxi_numpy[i][1], begin_dt)
    if sec_in is None:
        continue
    total_seconds_in = int(sec_in)
    time_step_in = int(total_seconds_in / 1800)
    if 0 <= time_step_in <= 4367:
        in_flow[time_step_in][int(revers_index_dict[taxi_numpy[i][3]]) ] += 1

begin_time = '2020-04-01T00:00:00Z'
format_pattern = '%Y-%m-%dT%H:%M:%SZ'
now = datetime.strptime(begin_time, format_pattern)

NYCTaxi20200406_dyna = []
NYCTaxi20200406_dyna.append('dyna_id,type,time,entity_id,inflow,outflow')
NYCTaxi20200406_geo = []
NYCTaxi20200406_geo.append('geo_id,type,coordinates')

dyna_id = 0
type = 'state'

for i in tqdm(range(in_flow.shape[1])):
    for j in range(in_flow.shape[0]):
        time_origin = now + timedelta(minutes = 30 * j)
        time_write = time_origin.strftime('%Y-%m-%dT%H:%M:%SZ')
        inflow = in_flow[j][i]
        outflow = out_flow[j][i]

        grid_record = str(dyna_id) + ',' + type + ',' + str(time_write) + ',' + str(index_dict[i]) + ',' + str(int(inflow)) + ',' + str(int(outflow))
        dyna_id += 1
        NYCTaxi20200406_dyna.append(grid_record)

    geo_record = str(index_dict[i]) + ',' + 'Point' + ',"[]"'
    NYCTaxi20200406_geo.append(geo_record)

with open(r'./USTP/NYC/NYCBike20200406/NYCBike20200406.dyna','w') as f1:
    for i in range(len(NYCTaxi20200406_dyna)):
        f1.write(NYCTaxi20200406_dyna[i])
        f1.write('\n')
f1.close()

with open(r'./USTP/NYC/NYCBike20200406/NYCBike20200406.geo','w') as f1:
    for i in range(len(NYCTaxi20200406_geo)):
        f1.write(NYCTaxi20200406_geo[i])
        f1.write('\n')
f1.close()

NYCTaxi20200406_rel = []
NYCTaxi20200406_rel.append('rel_id,type,origin_id,destination_id,cost')

rel_id = 0
type = 'geo'


road_dafaframe = pd.read_csv('../UrbanKG_data/Processed_data/NYC/NYC_road.csv')
road_dafaframe_filter = road_dafaframe[road_dafaframe['link_id'].isin(unique_road_id)]
road_datanumpy = road_dafaframe_filter[['geometry', 'link_id']].values

for i in tqdm(range(road_datanumpy.shape[0])):
    head_road_linestring = wkt.loads(str(road_datanumpy[i][0]))
    for j in range(road_datanumpy.shape[0]):
        tail_road_linestring = wkt.loads(str(road_datanumpy[j][0]))
        distance = head_road_linestring.distance(tail_road_linestring)

        NYCTaxi20200406_rel.append(str(rel_id) + ',' +  type + ',' + str(road_datanumpy[i][1]) + ',' + str(road_datanumpy[j][1])
                                   + ',' + str(distance))
        rel_id += 1

with open(r'./USTP/NYC/NYCBike20200406/NYCBike20200406.rel','w') as f1:
    for i in range(len(NYCTaxi20200406_rel)):
        f1.write(NYCTaxi20200406_rel[i])
        f1.write('\n')
f1.close()

##################################################################################################
##################################################################################################

"""

USTP_data 3: human

"""

bike_dataframe = read_csv_flexible('./Processed_data/NYC/NYC_human.csv')

for col in ['start_poi_id', 'end_poi_id']:
    bike_dataframe[col] = pd.to_numeric(bike_dataframe[col], errors='coerce')
bike_dataframe = bike_dataframe[bike_dataframe['start_poi_id'].notna()]
bike_dataframe = bike_dataframe[bike_dataframe['end_poi_id'].notna()]
bike_dataframe['start_poi_id'] = bike_dataframe['start_poi_id'].astype(np.int64)
bike_dataframe['end_poi_id'] = bike_dataframe['end_poi_id'].astype(np.int64)

taxi_numpy = bike_dataframe[['start_time', 'end_time', 'start_poi_id', 'end_poi_id']].values

begin_time = '2020-04-01 00:00:00.0000'
format_pattern = '%Y-%m-%d %H:%M:%S.%f'
begin_dt = datetime.strptime(begin_time, format_pattern)


unique_strat_road_id = bike_dataframe['start_poi_id'].unique()
unique_end_road_id = bike_dataframe['end_poi_id'].unique()
c = np.concatenate((unique_strat_road_id, unique_end_road_id))

unique_road_id = np.unique(c)

index_dict = {}
revers_index_dict = {}
for index, element in enumerate(unique_road_id):
    index_dict[index] = element
    revers_index_dict[element] = index

in_flow = np.zeros([91 * 24 * 2, len(index_dict)])
out_flow = np.zeros([91 * 24 * 2, len(index_dict)])

for i in tqdm(range(taxi_numpy.shape[0])):
    sec_out = time_to_seconds_since_begin(taxi_numpy[i][0], begin_dt)
    if sec_out is None:
        continue
    total_seconds_out = int(sec_out)
    time_step_out =  int(total_seconds_out / 1800)
    if 0<= time_step_out <= 4367:
        out_flow[time_step_out][ int(revers_index_dict[taxi_numpy[i][2]]) ] += 1

    sec_in = time_to_seconds_since_begin(taxi_numpy[i][1], begin_dt)
    if sec_in is None:
        continue
    total_seconds_in = int(sec_in)
    time_step_in = int(total_seconds_in / 1800)
    if 0 <= time_step_in <= 4367:
        in_flow[time_step_in][int(revers_index_dict[taxi_numpy[i][3]]) ] += 1

begin_time = '2020-04-01T00:00:00Z'
format_pattern = '%Y-%m-%dT%H:%M:%SZ'
now = datetime.strptime(begin_time, format_pattern)

NYCTaxi20200406_dyna = []
NYCTaxi20200406_dyna.append('dyna_id,type,time,entity_id,inflow,outflow')
NYCTaxi20200406_geo = []
NYCTaxi20200406_geo.append('geo_id,type,coordinates')

dyna_id = 0
type = 'state'

for i in tqdm(range(in_flow.shape[1])):
    for j in range(in_flow.shape[0]):
        time_origin = now + timedelta(minutes = 30 * j)
        time_write = time_origin.strftime('%Y-%m-%dT%H:%M:%SZ')
        inflow = in_flow[j][i]
        outflow = out_flow[j][i]

        grid_record = str(dyna_id) + ',' + type + ',' + str(time_write) + ',' + str(index_dict[i]) + ',' + str(int(inflow)) + ',' + str(int(outflow))
        dyna_id += 1
        NYCTaxi20200406_dyna.append(grid_record)

    geo_record = str(index_dict[i]) + ',' + 'Point' + ',"[]"'
    NYCTaxi20200406_geo.append(geo_record)

with open(r'./USTP/NYC/NYCHuman20200406/NYCHuman20200406.dyna','w') as f1:
    for i in range(len(NYCTaxi20200406_dyna)):
        f1.write(NYCTaxi20200406_dyna[i])
        f1.write('\n')
f1.close()

with open(r'./USTP/NYC/NYCHuman20200406/NYCHuman20200406.geo','w') as f1:
    for i in range(len(NYCTaxi20200406_geo)):
        f1.write(NYCTaxi20200406_geo[i])
        f1.write('\n')
f1.close()

NYCTaxi20200406_rel = []
NYCTaxi20200406_rel.append('rel_id,type,origin_id,destination_id,cost')

rel_id = 0
type = 'geo'


poi_dafaframe = pd.read_csv('../UrbanKG_data/Processed_data/NYC/NYC_poi.csv')
poi_dafaframe_filter = poi_dafaframe[poi_dafaframe['poi_id'].isin(unique_road_id)]
poi_datanumpy = poi_dafaframe_filter[['lng', 'lat', 'poi_id']].values

for i in tqdm(range(poi_datanumpy.shape[0])):
    head_point = Point(poi_datanumpy[i][0], poi_datanumpy[i][1])
    for j in range(poi_datanumpy.shape[0]):
        tail_point = Point(poi_datanumpy[j][0], poi_datanumpy[j][1])
        distance = head_point.distance(tail_point)

        NYCTaxi20200406_rel.append(str(rel_id) + ',' +  type + ',' + str(poi_datanumpy[i][2]) + ',' + str(poi_datanumpy[j][2])
                                   + ',' + str(distance))
        rel_id += 1

with open(r'./USTP/NYC/NYCHuman20200406/NYCHuman20200406.rel','w') as f1:
    for i in range(len(NYCTaxi20200406_rel)):
        f1.write(NYCTaxi20200406_rel[i])
        f1.write('\n')
f1.close()