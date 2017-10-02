import urllib.request
import json
import pprint
import os
import glob
import itertools
import cv2
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from keras.models import load_model 
from keras.preprocessing import image
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.models import model_from_json


def directions(address_start,address_end):
    start = address_start
    end = address_end
    key2 = "&key=" + "YOUR KEY"
    start1 = start.replace(' ','+')
    end1 = end.replace(' ','+')
    base3 = "https://maps.googleapis.com/maps/api/directions/json?origin="  + start1 + "&destination=" + end1 + "&alternatives=true&avoid=highways"
    MyUrl3 = base3 + key2
    googleResponse = urllib.request.urlopen(MyUrl3)
    jsonResponse = json.loads(googleResponse.read())
    t = jsonResponse
    return t

def find_elevation(encoded):
    key4 = "&key=" + "YOUR KEY"
    base4 =  "https://maps.googleapis.com/maps/api/elevation/json?path=enc:" + encoded + "&samples=10"
    MyUrl3 = base4 + key4
    googleResponse4 = urllib.request.urlopen(MyUrl3)
    jsonResponse4 = json.loads(googleResponse4.read())
    encode_response = jsonResponse4
    return encode_response

def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}

    # Coordinates have variable length when encoded, so just keep
    # track of whether we've hit the end of the string. In each
    # while loop iteration, a single coordinate is decoded.
    while index < len(polyline_str):
        # Gather lat/lon changes, store them in a dictionary to apply them later
        for unit in ['latitude', 'longitude']: 
            shift, result = 0, 0

            while True:
                byte = ord(polyline_str[index]) - 63
                index+=1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break

            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)

        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append({'lat': (lat / 100000.0), 'lng':(lng / 100000.0)})

    
    return coordinates

def find_points_and_join(t,option):
    route_dist = float(t['routes'][option]['legs'][0]['distance']['text'][:-3])
    routes = t['routes'][option]['legs'][0]['steps']
    polyline_list = []
    for i in routes:
        pol1 = i['polyline']['points']
        pol2 = decode_polyline(pol1)
        polyline_list.append(pol2)
    polyline_list = list(itertools.chain.from_iterable(polyline_list))
    return polyline_list,route_dist

def rearrange_polyline(polyline_list):
    encode_list = []
    num = len(polyline_list)
    i = 0
    for uu in range(0,num):
        lat = polyline_list[i]['lat']
        lng = polyline_list[i]['lng']
        encode_list.append((lat,lng))
        i+=1
    return encode_list

def encode_polyline(encode_list):
    import polyline
    encoded = polyline.encode(encode_list,5)
    return encoded 


def UTM_coord(encode_list):
    from pyproj import Proj
    isn2004=Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1")

    length = len(encode_list)
    j = 0
    x =[]
    y=[]
    coordinate = []
    for j in range(length):
        UTMx, UTMy = isn2004(encode_list[j][1],encode_list[j][0])
        x.append(UTMx)
        y.append(UTMy)
        coordinate.append((UTMx,UTMy))
    return coordinate

def make_linestring(coordinate):
    from shapely.geometry import LineString
    line = LineString(coordinate)
    return line


def interpolate_polyline(line):
    from pyproj import Proj
    isn2004=Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1")
    frac = np.arange(0,1,0.01)
    i=0
    points = []
    points_streetview = []
    poly_groups = []
    for i in range(len(frac)):
        point = line.interpolate(frac[i],normalized=True)
        x,y = isn2004(point.x,point.y,inverse=True)
        points.append((y,x))
        if i % 2 == 1:
            points_streetview.append((y,x))
        i+=1
    return points, points_streetview


def make_poly_strings(points):
    p = 1
    m = np.arange(1,len(points),2)
    poly_strings = []
    i=0
    for i in range(len(m)-1):
        poly_strings.append(points[m[i]-p:m[i+1]])
        i+=1
    poly_strings[-1].append(points[-1])

    print(len(poly_strings))
    return poly_strings



def retrieve_streetview(points_streetview):
    files = glob.glob("/home/ubuntu/Direction_Images/*")
    for f in files:
        os.remove(f)

    for num, item in enumerate(points_streetview):
        try:
            lat = item[0]
            long = item[1]
            myloc = r"/home/ubuntu/Direction_Images" #replace with your own location
            key = "&key=" + "YOUR KEY"
            base = "https://maps.googleapis.com/maps/api/streetview?size=800x1200&location=" + str(lat) + "," + str(long) + "&heading=220&pitch=-90"
            MyUrl = base + key 
            fi = "%d.jpg" %num
            urllib.request.urlretrieve(MyUrl, os.path.join(myloc,fi))
        except IndexError:
            print('Error')
            continue


def process_and_find_class(file,model):
    myloc = r"/home/ubuntu/Direction_Images"
    image_path = myloc + "/" + file
    img = image.load_img(image_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    predicted_vector = model.predict(x)
    return predicted_vector

def gmap_js_location(poly_string):
    length =len(poly_string)
    i = 0 
    dic = []
    for i in range(length):
        dic.append({'lat': poly_string[i][0], 'lng': poly_string[i][1]})
        i+=1
    return dic

def points_js_location(poly_string):
    dic = {'lat': poly_string[0], 'lng': poly_string[1]}
    return dic

def make_graphics(poly_strings,i_in,option):
    myloc = r"/home/ubuntu/Direction_Images"
    model= load_model('/home/ubuntu/samethod.h5')
    condition_list = []
    dist2 = []
    listed_files = os.listdir(myloc)
    print(len(listed_files))
    js_list = []
    dist2_bad = []
    dist2_good =[]
    elevation_bad =[]
    elevation_good = []
    elevation = []
    dist = []
    i = 0
    for file in listed_files[0:-1]:

        predicted_vector = process_and_find_class(file,model)
        print("Image processed - {}".format(i))
        # encode_list = rearrange_polyline(poly_strings[i])
        encoded = encode_polyline(poly_strings[i])
        encode_response = find_elevation(encoded)

        import geopy.distance

        lat_encode = [encode_response['results'][0]['location']['lat']]
        lng_encode = [encode_response['results'][0]['location']['lng']]
        len_encode = len(encode_response['results'])
        k = 1
        for k in range(1,len_encode):
            if k == 1 & i == 0:
                elevation.append(encode_response['results'][0]['elevation'])
            if k == 1:
                dist.append(0)
            lat_encode.append(encode_response['results'][k]['location']['lat'])
            lng_encode.append(encode_response['results'][k]['location']['lng'])
            elevation.append(encode_response['results'][k]['elevation'])
            # What you were looking for
            dist.append(geopy.distance.vincenty((lat_encode[k-1],lng_encode[k-1]), (lat_encode[k], lng_encode[k])).miles)
            dist2.append(sum(dist))
            k+=1
    
        if predicted_vector[0] == 0:

            javascript = "var flightPlanCoordinates" + str(i+i_in) + "=" + str(gmap_js_location(poly_strings[i])) +";\n"  "var flightPath" + str(i+i_in) +"= new google.maps.Polyline({\npath: flightPlanCoordinates" + str(i+i_in) + ",\n geodesic: true,\n strokeColor: '#FF0000',\n strokeOpacity: 1,\n strokeWeight: 6});\n" + "flightPath" +str(i+i_in)+ ".setMap(map);\n"
            js_list.append(javascript)
            condition_list.append('Bad')
            dist2_bad.append(dist2[-k:])
            elevation_bad.append(elevation[-k:])

        else:
            javascript = "var flightPlanCoordinates" + str(i+i_in) + "=" + str(gmap_js_location(poly_strings[i])) +";\n"  "var flightPath" + str(i+i_in) +"= new google.maps.Polyline({\npath: flightPlanCoordinates" + str(i+i_in) +",\n geodesic: true,\n strokeColor: '#32CD32',\n strokeOpacity: 1,\n strokeWeight: 6});\n" + "flightPath" +str(i+i_in)+ ".setMap(map);\n"
            js_list.append(javascript)
            condition_list.append('Good')
            dist2_good.append(dist2[-k:])
            elevation_good.append(elevation[-k:])
        i += 1



    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Elevation (ft)',fontsize=25)
    ax.set_xlabel('Distance (miles)',fontsize=25)
    ax.tick_params(axis='y', labelsize=22, width = 3)
    ax.tick_params(axis='x', labelsize=22, width = 3)
   
    l = 0
    for l in range(0,len(elevation_good)):
        chart1 =plt.plot(dist2_good[l],elevation_good[l],color='#32CD32',linewidth=4.5)
        l+=1

    m = 0
    for l in range(0,len(elevation_bad)):
        chart1 =plt.plot(dist2_bad[m],elevation_bad[m],'r',linewidth=4.5)
        m+=1

    plt.ylim((0, 150))
    ax.grid(color='k', linestyle='--', linewidth=1)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('/home/ubuntu/application/flaskexample/static/chart' + str(option)+'.png')
    
    start_point = str(points_js_location(poly_strings[0][0]))
    end_point = str(points_js_location(poly_strings[-1][-1]))
    cp = int(len(poly_strings)/2)
    center_point = str(points_js_location(poly_strings[cp][0]))

    dist_end = dist2[-1]
    elevation_range = max(elevation)-min(elevation)
    red_over_green = len(dist2_bad)/len(dist2_good)
    return poly_strings,js_list, center_point,i, start_point,end_point,dist_end, elevation_range, red_over_green
