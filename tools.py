import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, output_file, save
output_notebook()
from bokeh.models import HoverTool,BoxZoomTool,ResetTool, WheelZoomTool,Legend, LegendItem, ColumnDataSource, CDSView, IndexFilter
from bokeh.models.widgets import PreText, Select
from pandas.io.json import json_normalize
from copy import deepcopy
from sys import exit
import math
import bokeh.models as bkm
import bokeh.plotting as bkp

class Geography(object):
    def __init__(self,fs_dict,gs_dict,df_dict,select_recension=None):
        self.fs_dict=fs_dict
        self.gs_dict=gs_dict
        self.df_dict=df_dict
        self.select_recension=select_recension

    def plot_recension(self,fs_dict,gs_dict,df_dict,select_recension):
        
        """This function returns the map based on the entered values as follows:
        fs_dic ~ dictionary of  coastal localities in Omega  -->dict
        gs_dict ~dictionary of  coastal localities in Xi  -->dict
        df_dic ~ dictionary of all localities -->dict
        select_recension ~ either 'Omega ' or -'Xi '  -->string
        pay attention to capitalization 
        """
        
        tools = ["hover","box_select","box_zoom","wheel_zoom","reset"]
        TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
        p = figure(title=self.select_recension, width=1000, height=800, x_range=(1.5, 22), y_range=(35.5, 47),tooltips=TOOLTIPS, tools=tools)
        p.background_fill_color = "beige"
        p.background_fill_alpha = 0.5
        if select_recension== 'Omega':
            Locality={"x":list(self.df_dict["dfTemp"]['longitude']),"y":list(self.df_dict["dfTemp"]['latitude'])}
            source = ColumnDataSource(data=Locality)
            view = CDSView(source=source)
            for i in self.fs_dict.values():
                p.line(i[:,0],i[:,1], color="black",legend="Coasts and boundaries (Omega)",muted_alpha=0.2)
            co="dfTemp"
            p.circle(x='x', y='y',source=source,view=view, fill_color="blue",size=6,fill_alpha=.9,line_color="blue",line_alpha=0.6,legend="Locality (Omega) ",muted_alpha=0.2)
        elif select_recension== 'Xi':
            Locality={"x":list(self.df_dict["dfTempX"]['longitude']),"y":list(self.df_dict["dfTempX"]['latitude'])}
            source = ColumnDataSource(data=Locality)
            view = CDSView(source=source)
            for i in self.gs_dict.values():
                p.line(i[:,0],i[:,1], color="black",legend="Coasts and boundaries (Xi) ",muted_alpha=0.2,line_dash="dashdot")
            co='dfTempX'
            p.circle(np.array(self.df_dict[co]['longitude']),np.array(self.df_dict[co]['latitude']), fill_color="red",size=6, fill_alpha=.9, line_color="red",line_alpha=0.6,legend="Locality (Xi)",muted_alpha=0.2)
        p.legend.click_policy="mute"
        show(p)
            
    def plot_recension_all(self,fs_dict,gs_dict,df_dict):
        
            """This function returns the map based on the entered values as follows:
            fs_dic ~ dictionary of  coastal localities in Omega  -->dict
            gs_dict ~dictionary of  coastal localities in Xi  -->dict
            df_dic ~ dictionary of all localities -->dict
            """
            
            tools = ["hover","crosshair","box_select","box_zoom","wheel_zoom","reset"]
            TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
            p = figure(title=self.select_recension, width=1000, height=800, x_range=(1.5, 22), y_range=(35.5, 47),tools=tools,tooltips=TOOLTIPS)
            for i in self.fs_dict.values():
                p.line(i[:,0],i[:,1], color="dodgerblue",legend="Coasts and boundaries (Omega)",line_width=1.5)
            co="dfTemp"
            Locality={"x":list(self.df_dict["dfTemp"]['longitude']),"y":list(self.df_dict["dfTemp"]['latitude'])}
            source = ColumnDataSource(data=Locality)
            view = CDSView(source=source)
            p.circle(x='x', y='y',source=source,view=view, fill_color="dodgerblue",size=6,fill_alpha=.9,line_color="dodgerblue",line_alpha=0.6,legend="Locality (Omega)",muted_alpha=0.2)
            for i in self.gs_dict.values():
                p.line(i[:,0],i[:,1], color="red",legend="Coasts and boundaries (Xi)",line_dash="dashdot",line_width=1.5)
            co='dfTempX'
            Locality={"x":list(self.df_dict["dfTempX"]['longitude']),"y":list(self.df_dict["dfTempX"]['latitude'])}
            source = ColumnDataSource(data=Locality)
            view = CDSView(source=source)
            p.circle(x='x', y='y',source=source,view=view, fill_color="crimson",size=6, fill_alpha=.9, line_color="red",line_alpha=0.6,legend="Locality (Xi)",muted_alpha=.2)
            p.legend.click_policy="mute"
            show(p)
            
    def plot_compare_recension(self,fs_dict,gs_dict,df_dict):

            """This function returns the comparison map based on the entered values as follows:
            fs_dic ~ dictionary of all coastal localities in Omega
            gs_dic ~ dictionary of all coastal localities in Xi
            df_dic ~ dictionary of all localities -->dict
            """
            
            tools = ["hover","box_select","box_zoom","wheel_zoom","reset"]
            TOOLTIPS = [("index", "$index"),("(x,y)", "($x, $y)")]
            a=df_dict['dfTempX'].reset_index()
            a=a[a.longitude.apply(lambda row: row not in [0])]
            a=a.rename(columns={'longitude':'longitude_Xi','latitude':'latitude_Xi'})
            a=a[['ID','longitude_Xi','latitude_Xi']].dropna()

            b=df_dict['dfTemp'].reset_index()
            b=b[b.longitude.apply(lambda row: row not in [0])]
            b=b.rename(columns={'longitude':'longitude_Omega',"latitude":"latitude_Omega"})
            b=b[['ID','longitude_Omega','latitude_Omega']].dropna()
            c = pd.merge(left=a, right=b,how='inner')
            
            IbEq = c[c["longitude_Xi"]==c["longitude_Omega"]]
            IbEq = IbEq[IbEq["latitude_Xi"]==IbEq["latitude_Omega"]]
            
            r = figure(title='Comparison between Xi (red) and Omega (blue)', width=1000, height=800, x_range=(1.5, 22), y_range=(35.5, 47),tools=tools,tooltips=TOOLTIPS)
            # Xi 
            Locality={"x":list(self.df_dict["dfTempX"]['longitude']),"y":list(self.df_dict["dfTempX"]['latitude'])}
            source = ColumnDataSource(data=Locality)
            view = CDSView(source=source)
            r.circle(x='x', y='y',source=source,view=view, size=5,fill_color='red', fill_alpha=.7,                         line_color='Crimson',line_alpha=0,legend="Localities (Xi)")
            for i in gs_dict.values():
                            r.line(i[:,0],i[:,1], color='Crimson',legend="Coasts and boundaries (Xi) ",line_width=1.5)
            # Omega
            Locality={"x":list(self.df_dict["dfTemp"]['longitude']),"y":list(self.df_dict["dfTemp"]['latitude'])}
            source = ColumnDataSource(data=Locality)
            view = CDSView(source=source)
            r.circle(x='x', y='y',source=source,view=view, size=5,fill_color='blue', fill_alpha=0.8,   line_color='DodgerBlue',line_alpha=0,legend="Localities (Omega)",muted_alpha=0.2)
            for i in fs_dict.values():
                r.line(i[:,0],i[:,1], color='DodgerBlue',legend="Coasts and boundaries (Omega)",line_width=1.5)
            r.circle(np.array(IbEq["longitude_Xi"]),np.array(IbEq["latitude_Xi"]), size=5.5,fill_color='green', fill_alpha=1, line_color='green',line_alpha=0.8,legend="Locality with same coordinates in Xi and Omega" )
            r.segment(x0=c["longitude_Xi"], y0=c["latitude_Xi"], x1=c["longitude_Omega"],
                                  y1=c["latitude_Omega"], color="grey", line_width=1,legend="Distance line")
            r.legend.click_policy="hide"
            r.legend.location = "bottom_right"
            show(r);
            
def reformatCoord(row,longLat, xy='coord_x'):
    
    """Extract the integer and fraction part of the coordinate in Greek form for each row in the dataframe
       for instance 
       >>>row=dfTemp.iloc[26]
        ID                                                                                       2.04.06.02
        category                                                                                       city
        coord        {'long': {'integer': 'ς', 'fraction': 'L'}, 'lat': {'integer': 'λς', 'fraction': 'L'}}
        people                                                                                      Bastuli
        text                                                                                            NaN
        toponym                                                                                   Μενραλία
        type                                                                                       locality
        type_sec                                                                              coast section
        longitude                                                                                       6.5
        latitude                                                                                       36.5
       >>>reformatCoord(row,'long','coord')
       ('ς', 'L')
       >>>reformatCoord(row,'lat','coord')
       ('λς', 'L')
    """
    
    if type(row[xy]) == dict:
        return (row[xy][longLat]['integer'].strip(),row[xy][longLat]['fraction'].strip())
    else:
        return False

def reformatIntFrac(row):
    
    """"Transpose the Greek numerical system proposed by Ptolemy into a modern format using gfrac and gint. Returns a number as longitude or latitude
    a=('λς', 'L')
    >>>reformatIntFrac(a)
    36.5
    """
    
    gfrac={"":0,"ιβ":1/12,"ς":1/6,"δ":1/4,"γ":1/3,"γιβ":5/12,"L":1/2,"Lιβ":7/12,"γο":2/3,"Lδ":3/4,"Lγ":5/6,"Lγιβ":11/12,"η":1/8,"Lς":2/3,"ςL":2/3}
    gint={"":0,"α":1,"β":2,"γ":3,"δ":4,"ε":5,"ς":6,"ζ":7,"η":8,"θ":9,"ι":10,"κ":20,"λ":30,"μ":40}
    if type(row)==tuple:
        try:
            temp_frac = gfrac[row[1]]
        except:
            temp_frac = gint[row[1]]
        try:
            if len(row[0]) == 1:
                temp_int = gint[row[0]]
            elif len(row[0]) == 2:
                temp_int = gint[row[0][0]] + gint[row[0][1]]
            elif row[0] == '':
                temp_int = 0
        except:
            temp_int = None
        return temp_int + temp_frac
    
def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    
    nested_list = deepcopy(nested_list)
    
    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist
            
            
def findTS(s, om):
    for i, r in om.iterrows():
        sci = r["sec_ID"]
        if sci in s:
            return r["type_sec"]
    return "" 

           

def Js2Geodf(df):

    """transform the file format to produce a simple dataframe.
    """
    om=json_normalize(df,"section")
    om=om.dropna(subset=["sec_part"])
    l=[]
    for i,x in om.iterrows():
        k={"type_sec":x["type_sec"]}
        l.append([x["sec_part"]])
#
    listItems=list(flatten_list(l))
    dfout=pd.DataFrame(listItems)
    dfout["type_sec"]=dfout.apply(lambda x: findTS(x["ID"],om),axis=1)
    return(dfout)


## Definitions
# Make figures for recensions:

def makeScale(df, recension):
    '''
    df: DataFrame, which contain the cartographical information
    recension: str, eigher Xi or Omega
    '''
    
    scale = 50
    offsetX = 0.5
    if  math.ceil(np.max(np.array(df["latitude_" + recension]))) == np.max(np.array(df["latitude_" + recension])):
        offsetYup = 0.5
    else:
        offsetYup = 0
    if math.floor(np.min(np.array(df["latitude_" + recension]))) == np.min(np.array(df["latitude_" + recension])):
        offsetYdown = 0.5
    else:
        offsetYdown = 0
        
    maxLat = math.ceil(np.max(np.array(df["latitude_" + recension])))+offsetYup
    maxLong = math.ceil(np.max(np.array(df["longitude_" + recension])))+offsetX
    minLat = math.floor(np.min(np.array(df["latitude_" + recension])))-offsetYdown
    minLong = math.floor(np.min(np.array(df["longitude_" + recension])))-offsetX
    diffLat = int(maxLat-minLat)
    diffLong = int(maxLong-minLong)
    scales = [minLat, maxLat, minLong, maxLong, diffLat, diffLong, scale]
    return scales

def compareScale(scaleXi, scaleOmega):
    '''
    scaleXi: list of floats and ints, list containing for Xi recension [minimum latitude, maximum latitude, minimum longitude, maximum longitude, maximum latitude - minimum latitude, maximum longitude - minimum longitude, scale]
    scaleOmega: list of floats and ints, list containing for Omega recension [minimum latitude, maximum latitude, minimum longitude, maximum longitude, maximum latitude - minimum latitude, maximum longitude - minimum longitude, scale]
    '''
    
    scale = []
    if scaleXi[0] < scaleOmega[0]:
        scale.append(scaleXi[0])
    else:
        scale.append(scaleOmega[0])
        
    if scaleXi[1] > scaleOmega[1]:
        scale.append(scaleXi[1])
    else:
        scale.append(scaleOmega[1])
        
    if scaleXi[2] < scaleOmega[2]:
        scale.append(scaleXi[2])
    else:
        scale.append(scaleOmega[2])
        
    if scaleXi[3] > scaleOmega[3]:
        scale.append(scaleXi[3])
    else:
        scale.append(scaleOmega[3])
        
    if scaleXi[4] > scaleOmega[4]:
        scale.append(scaleXi[4])
    else:
        scale.append(scaleOmega[4])
        
    if scaleXi[5] > scaleOmega[5]:
        scale.append(scaleXi[5])
    else:
        scale.append(scaleOmega[5])
        
    scale.append(scaleOmega[6])
    return scale

def makeRecensions(df, title, recension, ID1, ID2, markColor, drawLine, scales):
    '''
    df: DataFrame, which contain the cartographical information
    title: str, giving the title of the plot
    recension: str, eigher Xi or Omega
    ID1: str or False, use False if no first ID is provided, str provides the the ID of the dataset
    ID2: str or False, use False if no second ID is provided, if the first ID is False, the second must be also False, str provides the the ID of the dataset
    markColor: ColorSpec, determine the color of the annulus mark
    drawLine: boolean, if True, a line is drawn to connect the points, if False no line is drawn
    '''
    
    hover = HoverTool(names=["point"])
    source = ColumnDataSource(data=dict(x=list(df["longitude_" + recension]), y=list(df["latitude_" + recension]), desc=list(df["toponym"])),)
    hover.tooltips = [("toponym", "@desc")]
    
    minLat = scales[0]
    maxLat = scales[1]
    minLong = scales[2]
    maxLong = scales[3]
    diffLat = scales[4]
    diffLong = scales[5]
    scale = scales[6]
    
    loc_var = df.groupby('diff').get_group('var')
    loc_id = df.groupby('diff').get_group('id')
    
    n = figure(title=title, width=diffLong*scale*3, height=diffLat*scale*4, x_range=(minLong, maxLong), y_range=(minLat, maxLat), tools=[hover,'pan', 'wheel_zoom'])
    n.xaxis.axis_label = 'Longitude [°]'
    n.yaxis.axis_label = 'Latitude [°]'
    if drawLine:
        n.line(np.array(df["longitude_" + recension]),np.array(df["latitude_" + recension]),line_alpha=0.6,line_color='black')
    n.circle('x','y',fill_color='black',size=6,fill_alpha=0.4,line_color='black', source=source, name="point")
    if not markColor:
        pass
    else:
        if not (ID1 or ID2):
            n.annulus(loc_var["longitude_" + recension], loc_var["latitude_" + recension], fill_color=markColor, inner_radius=0.04, outer_radius=0.06,fill_alpha=0.7,line_color='black',line_alpha=0)
        if not ID2 and ID1:
            n.annulus(df.loc[ID1, "longitude_" + recension], df.loc[ID1, "latitude_" + recension], fill_color=markColor, inner_radius=0.04, outer_radius=0.06,fill_alpha=0.7,line_color='black',line_alpha=0)
        if (ID1 and ID2):
            n.annulus(df.loc[ID1:ID2, "longitude_" + recension],df.loc[ID1:ID2,"latitude_" + recension],fill_color=markColor,inner_radius=0.04, outer_radius=0.06,fill_alpha=0.9,line_color='black',line_alpha=0)
    return n

def makeComparison(df, title, Compare1, Compare2, drawLine):

    df["longitude_id"] = [df["longitude_" + Compare1][i] if df["longitude_" + Compare1][i] == df["longitude_" + Compare2][i] else np.nan for i in range(len(df))]
    df["latitude_id"] = [df["latitude_" + Compare1][i] if df["latitude_" + Compare1][i] == df["latitude_" + Compare2][i] else np.nan for i in range(len(df))]
    
    scale = 50
    offsetX = 0.5
    if np.max(np.array(df["latitude_" + Compare1])) >= np.max(np.array(df["latitude_" + Compare2])):
        if  math.ceil(np.max(np.array(df["latitude_" + Compare1]))) == np.max(np.array(df["latitude_" + Compare1])):
            offsetYup = 0.5
        else:
            offsetYup = 0
        if math.floor(np.min(np.array(df["latitude_" + Compare1]))) == np.min(np.array(df["latitude_" + Compare1])):
            offsetYdown = 0.5
        else:
            offsetYdown = 0
    else:
        if  math.ceil(np.max(np.array(df["latitude_" + Compare2]))) == np.max(np.array(df["latitude_" + Compare2])):
            offsetYup = 0.5
        else:
            offsetYup = 0
        if math.floor(np.min(np.array(df["latitude_" + Compare2]))) == np.min(np.array(df["latitude_" + Compare2])):
            offsetYdown = 0.5
        else:
            offsetYdown = 0
    
    if np.max(np.array(df["latitude_" + Compare1])) >= np.max(np.array(df["latitude_" + Compare2])):
        maxLat = math.ceil(np.max(np.array(df["latitude_" + Compare1])))+offsetYup
        maxLong = math.ceil(np.max(np.array(df["longitude_" + Compare1])))+offsetX
        minLat = math.floor(np.min(np.array(df["latitude_" + Compare1])))-offsetYdown
        minLong = math.floor(np.min(np.array(df["longitude_" + Compare1])))-offsetX
    else:
        maxLat = math.ceil(np.max(np.array(df["latitude_" + Compare2])))+offsetYup
        maxLong = math.ceil(np.max(np.array(df["longitude_" + Compare2])))+offsetX
        minLat = math.floor(np.min(np.array(df["latitude_" + Compare2])))-offsetYdown
        minLong = math.floor(np.min(np.array(df["longitude_" + Compare2])))-offsetX
    diffLat = int(maxLat-minLat)
    diffLong = int(maxLong-minLong)

    source = bkm.ColumnDataSource(data=df)
    r = bkp.figure(title=title, width=diffLong*scale*3, height=diffLat*scale*4, x_range=(minLong, maxLong), y_range=(minLat, maxLat), tools=['pan', 'wheel_zoom'])
    r.xaxis.axis_label = 'Longitude [°]'
    r.yaxis.axis_label = 'Latitude [°]'
    g1 = bkm.Circle(x="longitude_" + Compare1, y="latitude_" + Compare1,fill_color='red',size=6,fill_alpha=0.4,line_color='darkred')
    g1_r = r.add_glyph(source_or_glyph=source, glyph=g1)
    g1_hover = bkm.HoverTool(renderers=[g1_r],
                             tooltips=[("toponym", "@toponym")])

    g2 = bkm.Circle(x="longitude_" + Compare2, y="latitude_" + Compare2,fill_color='blue',size=6,fill_alpha=0.4,line_color='darkblue')
    g2_r = r.add_glyph(source_or_glyph=source, glyph=g2)
    g2_hover = bkm.HoverTool(renderers=[g2_r],
                             tooltips=[("toponym", "@toponym")])

    g3 = bkm.Circle(x="longitude_" + "id", y="latitude_" + "id",fill_color='grey',size=6,fill_alpha=0.4,line_color='grey')
    g3_r = r.add_glyph(source_or_glyph=source, glyph=g3)
    g3_hover = bkm.HoverTool(renderers=[g3_r],
                            tooltips=[("toponym", "@toponym")])
    
    if drawLine:
        g_l = bkm.Segment(x0="longitude_" + Compare1, y0="latitude_" + Compare1, 
                    x1="longitude_" + Compare2, y1="latitude_" + Compare2, line_color="grey", line_width=1)
        r.add_glyph(source_or_glyph=source, glyph=g_l)
    r.add_tools(g1_hover, g2_hover)
    df.drop(["longitude_id", "latitude_id"], axis=1, inplace = True)
    return r

def makeRecesionsCoast(dfCoast, dfRecension, dfVettones, regionCoord, title, recension):
    '''
    dfCoast: DataFrame, which contain the cartographical information for the coast in question
    dfRecension: DataFrame, which contain the cartographical information for the whole recension in question
    dfVettones: DataFrame, which contain the cartographical information only for the region/people in question
    regionCoord: list of six lists with four floats each, containing the points in order to draw the enclosing line of the region correctly
    title: str, giving the title of the plot
    recension: str, eigher Xi or Omega
    '''
    
    p = figure(title=title, x_axis_label='', y_axis_label='',plot_width=500, plot_height=500,x_range=(4.5,10.5), y_range=(40,43))
    p.line(x=dfCoast['longitude_' + recension],y=dfCoast['latitude_' + recension],line_alpha=0.8, color='grey')
    p.line(regionCoord[0], regionCoord[1], line_alpha=0.8, color='grey')
    p.line(regionCoord[2], regionCoord[3], line_alpha=0.8, color='grey')
    p.line(regionCoord[4], regionCoord[5], line_alpha=0.8, color='grey')
    p.circle(dfRecension['longitude_' + recension],dfRecension['latitude_' + recension],size=5, fill_color='grey', fill_alpha=0.5, line_color='grey',line_alpha=0)
    p.line(dfVettones['longitude_' + recension],dfVettones['latitude_' + recension], line_color='black',line_alpha=0.5)
    p.circle(x=dfVettones["longitude_" + recension],y=dfVettones["latitude_" + recension],color='black',size=5.5)
    p.annulus(dfVettones.loc['2.05.09.04',"longitude_" + recension],dfVettones.loc['2.05.09.04',"latitude_" + recension], fill_color='orange',inner_radius=0.06, outer_radius=0.1,fill_alpha=0.7,line_color='orange',line_alpha=0)
    return p

# If the table have to show only parameters of one ID, the second one have to be None.
def makeTable(df, width, ID1, ID2):
    '''
    df: DataFrame, which contain the cartographical information
    width: float/int, giving the width of the table
    ID1: str, str provides the the ID of the first dataset
    ID2: str or None, use None if no second ID is provided, str provides the the ID of the second dataset
    '''
    
    table = PreText(text="", width=width)
    if not ID2:
        table.text = str(df.loc[[ID1]].T)
    else:
        table.text = str(df.loc[ID1:ID2].T)
    return table


# If the table have to show only parameters of one ID, the second one have to be None.
def makeDf(peoples):
    '''
    df_Omega: DataFrame, which contain the cartographical information of the Omega recension
    df_Xi: DataFrame, which contain the cartographical information of the Xi recension
    peoples: str, people of which the localisation is examine
    ID2: str or None, use None if no second ID is provided, str provides the the ID of the second dataset
    '''
    
    Omega = pd.read_json('./data/OmegaStructure.json',encoding="utf8")
    Xi = pd.read_json('./data/XiStructure.json',encoding="utf8")
    
    dfOmega=Js2Geodf(Omega["chapters"][0])
    dfXi=Js2Geodf(Xi["chapters"][0])
    
    # We transpose the Greek numeral system used by Ptolemy for the coordiantes into a modern format, using decimal. We create two columns for each dataframe: one for the longitude, one for the latitude, thanks to the functions reformatCoord and reformatIntFrac.
    
    ## Omega
    dfTemp = dfOmega.copy()
    dfTemp['longitude_Omega'] = dfOmega.apply(lambda row: reformatCoord(row,'long','coord'),axis=1).apply(reformatIntFrac)
    dfTemp['latitude_Omega'] = dfOmega.apply(lambda row: reformatCoord(row,'lat','coord'),axis=1).apply(reformatIntFrac)
    
    ## Xi
    dfTempX = dfXi.copy()
    dfTempX['longitude_Xi'] = dfXi.apply(lambda row: reformatCoord(row,'long','coord'),axis=1).apply(reformatIntFrac)
    dfTempX['latitude_Xi'] = dfXi.apply(lambda row: reformatCoord(row,'lat','coord'),axis=1).apply(reformatIntFrac)
    
    # From the dataframes we extract the list of the localities belonging to certain peoples.
    
    LocXi = dfTempX[dfTempX.people == peoples]
    LocOmega = dfTemp[dfTemp.people == peoples]
    
    # We merge the two recensions in one single dataframe.
    
    LocXiN = LocXi[['ID','longitude_Xi','latitude_Xi']]
    dfLoc = pd.merge(LocOmega, LocXiN, on='ID')
    dfLoc['diff'] = np.where((dfLoc['longitude_Omega']==dfLoc['longitude_Xi']) & (dfLoc['latitude_Omega']==dfLoc['latitude_Xi']),'id','var')
    dfLoc = dfLoc[['ID','type_sec','people','type','category','toponym','longitude_Xi','latitude_Xi','longitude_Omega','latitude_Omega','diff']].set_index('ID')
    return dfLoc
    
def makeRecensionsMultiPeople(df, title, recension, ID1, ID2, markColor, drawLine, scales, multipeople):
    '''
    df: DataFrames, which contain the cartographical information of all peoples
    title: str, giving the title of the plot
    recension: str, eigher Xi or Omega
    ID1: str or False, use False if no first ID is provided, str provides the the ID of the dataset
    ID2: str or False, use False if no second ID is provided, if the first ID is False, the second must be also False, str provides the the ID of the dataset
    markColor: ColorSpec or False, determine the color of the annulus mark or if False draw no annulus
    drawLine: boolean, if True, a line is drawn to connect the points, if False no line is drawn
    multipeople: False or dict, if False the data only for one people is supposed to drawn, if multiple people are supposed to be drawn provide a dict assigning peoples to the colors to distinguish th peoples 
    '''
    
    minLat = scales[0]
    maxLat = scales[1]
    minLong = scales[2]
    maxLong = scales[3]
    diffLat = scales[4]
    diffLong = scales[5]
    scale = scales[6]
    
    hover = HoverTool(names=["point"])
    hover.tooltips = [("toponym", "@desc")]
    
    n = figure(title=title, width=diffLong*scale*3, height=diffLat*scale*4, x_range=(minLong, maxLong), y_range=(minLat, maxLat), tools=[hover,'pan', 'wheel_zoom'])
    n.xaxis.axis_label = 'Longitude [°]'
    n.yaxis.axis_label = 'Latitude [°]'
    
    loc_var = df.groupby('diff').get_group('var')
    loc_id = df.groupby('diff').get_group('id')
    
    for people in list(multipeople.keys()):
        source = ColumnDataSource(data=dict(x=list(df[df['people'] == people]["longitude_" + recension]), y=list(df[df['people'] == people]["latitude_" + recension]), desc=list(df[df['people'] == people]["toponym"])),)
        if drawLine:
            n.line(np.array(df[df['people'] == people]["longitude_" + recension]),np.array(df[df['people'] == people]["latitude_" + recension]),line_alpha=0.6,line_color='black')
        n.circle('x','y',fill_color=multipeople[people],size=6,fill_alpha=0.4,line_color=multipeople[people], source=source, name="point")
    if not markColor:
        pass
    else:
        if not (ID1 or ID2):
            n.annulus(loc_var["longitude_" + recension], loc_var["latitude_" + recension], fill_color=markColor, inner_radius=0.04, outer_radius=0.06,fill_alpha=0.5,line_color=markColor,line_alpha=0)
        if not ID2 and ID1:
            n.annulus(df.loc[ID1, "longitude_" + recension], df.loc[ID1, "latitude_" + recension], fill_color=markColor, inner_radius=0.04, outer_radius=0.06,fill_alpha=0.5,line_color=markColor,line_alpha=0)
        if (ID1 and ID2):
            n.annulus(df.loc[ID1:ID2, "longitude_" + recension],df.loc[ID1:ID2,"latitude_" + recension],fill_color=markColor,inner_radius=0.04, outer_radius=0.06,fill_alpha=0.5,line_color=markColor,line_alpha=0)
    return n
        
def table2latex(df, datapoints, formatting):
    '''
    df: DataFrame, which contains the cartographical information of all peoples
    datapoints: list of str, which contain the IDs of the datapoints to show 
    formatting: str, can only contain "l" (left aligned), "c" (centered) and "r" (right aligned), number of characters in formatting must be equal to length of datapoints
    '''
    
    df.loc[datapoints].T.to_latex('_'.join(datapoints) + '.tex', column_format=formatting)        




    
