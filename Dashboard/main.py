import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
st.set_page_config(layout='wide')

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)







#defination
items=['Apples','Wheat','Barley','Maize',"Tomatoes","Sugar cane","Potatoes","Olives"]
month=["Sep–Oct–Nov",'Jun–Jul–Aug','Dec–Jan–Feb','Mar–Apr–May','Meteorological year']
def load_cropData():
     data=pd.read_csv('.\data\FAOSTAT_data_10-16-2021.csv')
     return data
def load_valueData():
    data=pd.read_csv('.\data\FAOSTAT_GPV.csv')
    return data
def load_tempChange():
    data=pd.read_csv(".\data\FAOSTAT_tempChange.csv")
    return data

# with col2:
# st.title("Data-Visualisation For Agriculture")
st.markdown("<h1 style='text-align: center;'>Data-Visualisation For Agriculture</h1>", unsafe_allow_html=True)


#Image representation
 
image = Image.open('agriculture.jpg')    
new_image = image.resize((2000, 700))
st.image(new_image)   

st.sidebar.title('Parameters')  

left, middle, right = st.columns((3, 5, 2))  

count=1
#working with crop dataset
st.sidebar.subheader('Crop production')
if  st.sidebar.checkbox("Yield quantities based on region" ):
    with middle:
        if st.checkbox("Show  Crop Data"):
            st.write(load_cropData().head(20))  
    count+=1
    select_region=st.sidebar.selectbox('region',['India','Morocco','Russian Federation'],key=count)
    count+=1
    select_item=st.sidebar.selectbox('Item',items,key=count)
    count+=1
    data=load_cropData()
    x=data[data["Item"]==select_item]
    y=x[x["Element"]=="Area harvested"] 
    n=x[x["Element"]=="Production"] 
    z=y[y["Area"]==select_region]
    z1=n[n["Area"]==select_region]
    string="Production/Yield quantities of "+select_item+" in "+select_region


    # with middle:
    #     st.markdown('<p class="big-font"> </p>', unsafe_allow_html=True) 

    fig=go.Figure()
    fig.add_trace( go.Scatter(name="Area Harvested",x=z["Year"], y=z["Value"]))
    fig.add_trace(go.Scatter(name="Production",x=z1["Year"], y=z1["Value"] ))
    fig.update_layout(
    title=string,
    xaxis_title="years",
    yaxis_title="value",
    legend_title="element",)
    with middle:
        st.plotly_chart(fig) 
        st.markdown("""----------------""")


if  st.sidebar.checkbox("Most Produced Commodities"):
    count+=1
    select_region=st.sidebar.radio('region',('India','Morocco','Russian Federation'))
    data=load_cropData()
    region=data[data["Area"]==select_region]
    region=region[region["Element"]=="Production"]
    region=region.groupby(["Item"]).mean()
    region=region.sort_values(by=["Value"],ascending=False)
    region=region[:15]
    string="Most Produced Commodities in "+select_region
    # with middle:
    #     st.markdown('<p class="big-font">Most Produced Commodities in </p>' +select_region, unsafe_allow_html=True)  
    fig=px.pie(region,names=region.index,values=region['Value'],title=string)
    with middle:
        st.plotly_chart(fig)
        st.markdown("""---""")

#working with production dataset
st.sidebar.subheader('Production Value')
if  st.sidebar.checkbox("Value of Agriculture production"):
    count+=1
    select_region=st.sidebar.selectbox('region',['India','Morocco','Russian Federation'],key=count)
    count+=1
    select_item=st.sidebar.selectbox('Item',items,key=count)
    count+=1
    data=load_valueData()
    x=data[data["Item"]==select_item]
    z=x[x["Area"]==select_region]
    string="Gross Production Value (current thousand US$) in "+select_region+" - "+select_item
    # with middle:
    #     st.markdown("Gross Production Value (current thousand US$) in "+select_region+" - "+select_item)
    fig=go.Figure()
    fig.add_trace( go.Scatter(x=z["Year"], y=z["Value"] ))
    fig.update_layout(
    title=string,
    xaxis_title="years",
    yaxis_title="1000$",
  )
    with middle:
        st.plotly_chart(fig)
        st.markdown("""---""")

#working with climate datset
st.sidebar.subheader('Temperature Change')
if  st.sidebar.checkbox("Mean Temperature Change "):
    count+=1
    select_region=st.sidebar.selectbox('region',['India','Morocco','Russian Federation'],key=count)
    count+=1
    select_time=st.sidebar.selectbox('month',month,key=count)
    count+=1
    data=load_tempChange()
    x=data[data["Area"]==select_region]
    y=x[x['Months']==select_time]
    string="Mean Temperature change of "+select_region+" in "+select_time
    # with middle:
    #     st.markdown("Mean Temperature change of "+select_region+" in "+select_time)
    fig=go.Figure()
    fig.add_trace( go.Scatter(x=y["Year"], y=y["Value"] ))
    fig.update_layout(
    title=string,
    xaxis_title="years",
    yaxis_title="°C",
  )
    with middle:
        st.plotly_chart(fig) 
