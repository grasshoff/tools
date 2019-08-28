import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, output_file, save
output_notebook()
from bokeh.models import HoverTool,BoxZoomTool,ResetTool, WheelZoomTool,Legend, LegendItem, ColumnDataSource, CDSView, IndexFilter
from pandas.io.json import json_normalize
from copy import deepcopy
from sys import exit

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






    
        
        




    
