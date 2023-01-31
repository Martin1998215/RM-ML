import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

hide = """
<style>
div[data-testid="stConnectionStatus"] {
    display: none !important;
</style>
"""

st.markdown(hide, unsafe_allow_html=True)


label = LabelEncoder()
rf = RandomForestRegressor(n_estimators=300)

with st.sidebar:
    selected = option_menu("Menu List",
                           ["Home", "House Rent",
                               "Compare Places", "About Us"],
                           icons=["house-door", "file-bar-graph", "file-bar-graph",
                                  "activity"],
                           default_index=0)


st.sidebar.write('''
**AI For Real Estate**

Predicitive Modeling.

AI For Decision Making and Predictions.

**Locas Technology**.

''')

if selected == "Home":
    st.title("Locas Technology")
    st.write('---')
    st.subheader("Welcome to our ML App")

    from PIL import Image
    img = Image.open("mlcoverpic1.jpg")
    img1 = Image.open("mlcoverpic2.jpg")
    st.write("**Cover Photos**")
    st.image(img, caption="Cover Photo")
    st.image(img1, caption="Cover Photo")

    st.subheader("AI For Real Estate.")
    st.write('''

    # Our Services
    1. Price Predictions on Houses to rent.
    2. Tenants Approvals For whom to let in.

    ''')

    st.write('''

    # Our Contacts
    1. Email: locastechnology@gmail.com
    2. Contacts: 0976 03 57 66

    ''')


if selected == "Test":

    st.subheader("Predict Rent Prices for House.")
    st.write("### Inputs Here....")

    rooms = st.number_input(
        "Number of Rooms", step=1)

    broom = st.number_input("Number of Bedrooms", step=1)
    year = st.number_input("Current Year eg 2023", step=1)
    # year1 = st.date_input("Year", SingleDateValue="Year")
    in_toilet = st.selectbox("Is Toilet Inside?",
                             ["Yes", "No"])
    if in_toilet == "Yes":
        toilet = 1
    else:
        toilet = 0

    # area = st.selectbox("Area Located (Low or Medium or High Cost)?",
    #                     ["Low Cost", "Medium Cost", "High Cost"])
    # if area == "Low Cost":
    #     area1 = 0
    # elif area == "Medium Cost":
    #     area1 = 1
    # else:
    #     area1 = 2

    fence = st.selectbox("Has A Fence?",
                         ["No", "Wire Fence", "Wall Fence"])
    if fence == "No":
        fence1 = 0
    elif fence == "Wire Fence":
        fence1 = 1
    else:
        fence1 = 2

    geyser = st.selectbox("Has A Geyser?",
                          ["No", "Yes"])
    if geyser == "Yes":
        geyser1 = 1
    else:
        geyser1 = 0

    air_con = st.selectbox("Has An Air Con?",
                           ["No", "Yes"])
    if air_con == "Yes":
        ac = 1
    else:
        ac = 0

    ac_num = st.number_input("Number of Air Cons?", step=1)

    tiles = st.selectbox("Has Tiles?",
                         ["No", "Yes"])
    if tiles == "Yes":
        tile1 = 1
    else:
        tile1 = 0

    ceiling = st.selectbox("Has A Ceiling Board?",
                           ["No", "Yes"])
    if ceiling == "Yes":
        ceil = 1
    else:
        ceil = 0

    standalone = st.selectbox("Is It Standalone?",
                              ["No", "Yes"])
    if standalone == "Yes":
        stand = 1
    else:
        stand = 0

    section = st.selectbox(
        "Section Located", [217, "Dcentral", "Dnorth", "DNorthEx", "Dside", "DSVEtx", "EllenBrittel", "KombeDrive",
                            "Highlands", "Libuyu", "Linda", "MA", "Malota", "MB", "MC", "MD", "ME", "Mesinja", "Namatama", "Ngwenya",
                            "NottieBrod", "Railways", "TownArea", "Zecco"])

    if section == 217:
        sect = 0
        area = 2
        av_rent = 1523.4528
        av_sect = 1606.2500
    elif section == "Dcentral":
        sect = 5
        area = 1
        av_rent = 872.9880
        av_sect = 1788.4615
    elif section == "Dnorth":
        sect = 6
        area = 1
        av_rent = 872.9880
        av_sect = 1062.9630
    elif section == "DNorthEx":
        sect = 3
        area = 2
        av_rent = 1523.4528
        av_sect = 1538.8889
    elif section == "Dside":
        sect = 8
        area = 1
        av_rent = 872.9880
        av_sect = 475.0000
    elif section == "DSVEtx":
        sect = 4
        area = 1
        av_rent = 872.9880
        av_sect = 381.8182
    elif section == "EllenBrittel":
        sect = 9
        area = 2
        av_rent = 1523.4528
        av_sect = 1960.0000
    elif section == "KombeDrive":
        sect = 11
        area = 2
        av_rent = 1523.4528
        av_sect = 2183.3333
    elif section == "Highlands":
        sect = 10
        area = 2
        av_rent = 1523.4528
        av_sect = 1381.8182
    elif section == "Libuyu":
        sect = 12
        area = 1
        av_rent = 872.9880
        av_sect = 490.7895
    elif section == "Linda":
        sect = 13
        area = 1
        av_rent = 872.9880
        av_sect = 455.7692
    elif section == "MA":
        sect = 14
        area = 1
        av_rent = 872.9880
        av_sect = 605.1351
    elif section == "Malota":
        sect = 19
        area = 0
        av_rent = 423.2353
        av_sect = 750.0000
    elif section == "MB":
        sect = 15
        area = 1
        av_rent = 872.9880
        av_sect = 494.0000
    elif section == "MC":
        sect = 16
        area = 1
        av_rent = 872.9880
        av_sect = 612.8571
    elif section == "MD":
        sect = 17
        area = 1
        av_rent = 872.9880
        av_sect = 1200.0000
    elif section == "ME":
        sect = 18
        area = 1
        av_rent = 872.9880
        av_sect = 791.6667
    elif section == "Mesinja":
        sect = 21
        area = 1
        av_rent = 872.9880
        av_sect = 737.5000
    elif section == "Namatama":
        sect = 22
        area = 1
        av_rent = 872.9880
        av_sect = 370.5882
    elif section == "Ngwenya":
        sect = 23
        area = 0
        av_rent = 423.2353
        av_sect = 376.3333
    elif section == "NottieBrod":
        sect = 25
        area = 2
        av_rent = 1523.4528
        av_sect = 1377.8571
    elif section == "Railways":
        sect = 26
        area = 1
        av_rent = 872.9880
        av_sect = 1296.5909
    elif section == "TownArea":
        sect = 27
        area = 2
        av_rent = 1523.4528
        av_sect = 963.6364
    elif section == "Zecco":
        sect = 29
        area = 0
        av_rent = 423.2353
        av_sect = 800.0000

    def findSum(arr):
        sum = 0
        for element in arr:
            sum += element
        return sum

    # Find Total: area1, fence1, sum and ac_num
    def findTotal(arr):
        tot = 0
        for element in arr:
            tot += element
        return tot

    sum = findSum([toilet, geyser1, ac, tile1, ceil, stand])
    total = findTotal([area, fence1, sum, ac_num])

    # st.dataframe(dx[["Section", "Ave_Sect"]])

    if st.button("Enter Button"):

        vector = (rooms, broom, year, toilet, area, fence1, geyser1,
                  ac, ac_num, tile1, ceil, stand, sect, sum, av_rent, av_sect, total)

        ev = rf.predict(
            [vector])

        st.success('Estimated Rent Price is: K{}'.format(
            round(ev[0], 2)))
        st.write(vector)

        # show data structure...
        # st.dataframe(dx)

    st.write('---')


if selected == "House Rent":

    rent_model = pickle.load(
        open("C:/Users/Dell/Desktop/rent/rent_model.sav", 'rb'))

    st.title("Rent")
    st.write('---')
    st.subheader("Predict Rent Prices for House.")
    st.write("### Inputs Here....")

    rooms = st.number_input(
        "Number of Rooms", step=1)

    broom = st.number_input("Number of Bedrooms", step=1)
    year = st.number_input("Current Year eg 2023", step=1)
    in_toilet = st.selectbox("Is Toilet Inside?",
                             ["Yes", "No"])
    if in_toilet == "Yes":
        toilet = 1
    else:
        toilet = 0

    area = st.selectbox("Area Located (Low or Medium or High Cost)?",
                        ["Low Cost", "Medium Cost", "High Cost"])
    if area == "Low Cost":
        area1 = 0
    elif area == "Medium Cost":
        area1 = 1
    else:
        area1 = 2

    fence = st.selectbox("Has A Fence?",
                         ["No", "Wire Fence", "Wall Fence"])
    if fence == "No":
        fence1 = 0
    elif fence == "Wire Fence":
        fence1 = 1
    else:
        fence1 = 2

    geyser = st.selectbox("Has A Geyser?",
                          ["No", "Yes"])
    if geyser == "Yes":
        geyser1 = 1
    else:
        geyser1 = 0

    air_con = st.selectbox("Has An Air Con?",
                           ["No", "Yes"])
    if air_con == "Yes":
        ac = 1
    else:
        ac = 0

    ac_num = st.number_input("Number of Air Cons?", step=1)

    tiles = st.selectbox("Has Tiles?",
                         ["No", "Yes"])
    if tiles == "Yes":
        tile1 = 1
    else:
        tile1 = 0

    ceiling = st.selectbox("Has A Ceiling Board?",
                           ["No", "Yes"])
    if ceiling == "Yes":
        ceil = 1
    else:
        ceil = 0

    standalone = st.selectbox("Is It Standalone?",
                              ["No", "Yes"])
    if standalone == "Yes":
        stand = 1
    else:
        stand = 0

    section = st.selectbox(
        "Section Located", [217, "Dcentral", "Dnorth", "DNorthEx", "Dside", "DSVEtx", "EllenBrittel", "KombeDrive",
                            "Highlands", "Libuyu", "Linda", "MA", "Malota", "MB", "MC", "MD", "ME", "Mesinja", "Namatama", "Ngwenya",
                            "NottieBrod", "Railways", "TownArea", "Zecco"])

    if section == 217:
        sect = 0
    elif section == "Dcentral":
        sect = 5
    elif section == "Dnorth":
        sect = 6
    elif section == "DNorthEx":
        sect = 3
    elif section == "Dside":
        sect = 8
    elif section == "DSVEtx":
        sect = 4
    elif section == "EllenBrittel":
        sect = 9
    elif section == "KombeDrive":
        sect = 11
    elif section == "Highlands":
        sect = 10
    elif section == "Libuyu":
        sect = 12
    elif section == "Linda":
        sect = 13
    elif section == "MA":
        sect = 14
    elif section == "Malota":
        sect = 19
    elif section == "MB":
        sect = 15
    elif section == "MC":
        sect = 16
    elif section == "MD":
        sect = 17
    elif section == "ME":
        sect = 18
    elif section == "Mesinja":
        sect = 21
    elif section == "Namatama":
        sect = 22
    elif section == "Ngwenya":
        sect = 23
    elif section == "NottieBrod":
        sect = 25
    elif section == "Railways":
        sect = 26
    elif section == "TownArea":
        sect = 27
    elif section == "Zecco":
        sect = 29

    # summation: toilet, geyser1, ac, tile1, ceil, stand

    def findSum(arr):
        sum = 0
        for element in arr:
            sum += element
        return sum

    # Find Total: area1, fence1, sum and ac_num
    def findTotal(arr):
        tot = 0
        for element in arr:
            tot += element
        return tot

    sum = findSum([toilet, geyser1, ac, tile1, ceil, stand])
    total = findTotal([area1, fence1, sum, ac_num])

    if st.button("Enter Button"):

        vector = (rooms, broom, year, toilet, area1, fence1,
                  geyser1, ac, ac_num, tile1, ceil, stand, sect, sum, total)

        rent_prediction = rent_model.predict([vector])

        st.success('Estimated House Rent Price is : K{}'.format(
            round(rent_prediction[0], 2)))

        st.write(vector)

    st.write('---')
    st.subheader("Disclaimer!")
    st.warning('''Please NOTE: Values generated by the AI or predictive model are not exact.
     They are just predictions based on the training data collected from various places.''')
    st.write('---')
    st.write('''
    **This Is strickly for Livingstone. Much of the data used to train the model
     was collected from Livingstone.**
    ''')
    st.write('''The Model was built to help people evaluate how much houses cost
     to rent and if the price is worthy the house.''')


if selected == "Compare Places":

    rent_model = pickle.load(
        open("C:/Users/Dell/Desktop/rent/rent_model.sav", 'rb'))

    st.title("Compare Places")
    st.write('---')
    st.subheader("Make Comparables Based on Locatins/Sections")

    st.write("### Inputs Here....")

    rooms = st.number_input(
        "Number of Rooms", step=1)

    broom = st.number_input("Number of Bedrooms", step=1)
    year = st.number_input("Current Year eg 2023", step=1)
    in_toilet = st.selectbox("Is Toilet Inside?",
                             ["Yes", "No"])
    if in_toilet == "Yes":
        toilet = 1
    else:
        toilet = 0

    area = st.selectbox("Area Located (Low or Medium or High Cost)?",
                        ["Low Cost", "Medium Cost", "High Cost"])
    if area == "Low Cost":
        area1 = 0
    elif area == "Medium Cost":
        area1 = 1
    else:
        area1 = 2

    fence = st.selectbox("Has A Fence?",
                         ["No", "Wire Fence", "Wall Fence"])
    if fence == "No":
        fence1 = 0
    elif fence == "Wire Fence":
        fence1 = 1
    else:
        fence1 = 2

    geyser = st.selectbox("Has A Geyser?",
                          ["No", "Yes"])
    if geyser == "Yes":
        geyser1 = 1
    else:
        geyser1 = 0

    air_con = st.selectbox("Has An Air Con?",
                           ["No", "Yes"])
    if air_con == "Yes":
        ac = 1
    else:
        ac = 0

    ac_num = st.number_input("Number of Air Cons?", step=1)

    tiles = st.selectbox("Has Tiles?",
                         ["No", "Yes"])
    if tiles == "Yes":
        tile1 = 1
    else:
        tile1 = 0

    ceiling = st.selectbox("Has A Ceiling Board?",
                           ["No", "Yes"])
    if ceiling == "Yes":
        ceil = 1
    else:
        ceil = 0

    standalone = st.selectbox("Is It Standalone?",
                              ["No", "Yes"])
    if standalone == "Yes":
        stand = 1
    else:
        stand = 0

    st.write('---')
    st.write("### Compare Different Locations...")

    section = st.selectbox(
        "Section Located", [217, "Dcentral", "Dnorth", "DNorthEx", "Dside", "DSVEtx", "EllenBrittel", "KombeDrive",
                            "Highlands", "Libuyu", "Linda", "MA [Maramba]", "Malota", "MB [Maramba]",
                            "MC [Maramba]", "MD [Maramba]", "ME [Maramba]", "Mesinja [Maramba]", "Namatama", "Ngwenya",
                            "NottieBrod", "Railways", "TownArea", "Zecco"])

    if section == 217:
        sect = 0
    elif section == "Dcentral":
        sect = 5
    elif section == "Dnorth":
        sect = 6
    elif section == "DNorthEx":
        sect = 3
    elif section == "Dside":
        sect = 8
    elif section == "DSVEtx":
        sect = 4
    elif section == "EllenBrittel":
        sect = 9
    elif section == "KombeDrive":
        sect = 11
    elif section == "Highlands":
        sect = 10
    elif section == "Libuyu":
        sect = 12
    elif section == "Linda":
        sect = 13
    elif section == "MA [Maramba]":
        sect = 14
    elif section == "Malota":
        sect = 19
    elif section == "MB [Maramba]":
        sect = 15
    elif section == "MC [Maramba]":
        sect = 16
    elif section == "MD [Maramba]":
        sect = 17
    elif section == "ME [Maramba]":
        sect = 18
    elif section == "Mesinja [Maramba]":
        sect = 21
    elif section == "Namatama":
        sect = 22
    elif section == "Ngwenya":
        sect = 23
    elif section == "NottieBrod":
        sect = 25
    elif section == "Railways":
        sect = 26
    elif section == "TownArea":
        sect = 27
    elif section == "Zecco":
        sect = 29

    section2 = st.selectbox(
        "Section Located 2", [217, "Dcentral", "Dnorth", "DNorthEx", "Dside", "DSVEtx", "EllenBrittel", "KombeDrive",
                              "Highlands", "Libuyu", "Linda", "MA [Maramba]", "Malota", "MB [Maramba]",
                              "MC [Maramba]", "MD [Maramba]", "ME [Maramba]", "Mesinja [Maramba]", "Namatama", "Ngwenya",
                              "NottieBrod", "Railways", "TownArea", "Zecco"])

    if section2 == 217:
        sect2 = 0
    elif section2 == "Dcentral":
        sect2 = 5
    elif section2 == "Dnorth":
        sect2 = 6
    elif section2 == "DNorthEx":
        sect2 = 3
    elif section2 == "Dside":
        sect2 = 8
    elif section2 == "DSVEtx":
        sect2 = 4
    elif section2 == "EllenBrittel":
        sect2 = 9
    elif section2 == "KombeDrive":
        sect2 = 11
    elif section2 == "Highlands":
        sect2 = 10
    elif section2 == "Libuyu":
        sect2 = 12
    elif section2 == "Linda":
        sect2 = 13
    elif section2 == "MA [Maramba]":
        sect2 = 14
    elif section2 == "Malota":
        sect2 = 19
    elif section2 == "MB [Maramba]":
        sect2 = 15
    elif section2 == "MC [Maramba]":
        sect2 = 16
    elif section2 == "MD [Maramba]":
        sect2 = 17
    elif section2 == "ME [Maramba]":
        sect2 = 18
    elif section2 == "Mesinja [Maramba]":
        sect2 = 21
    elif section2 == "Namatama":
        sect2 = 22
    elif section2 == "Ngwenya":
        sect2 = 23
    elif section2 == "NottieBrod":
        sect2 = 25
    elif section2 == "Railways":
        sect2 = 26
    elif section2 == "TownArea":
        sect2 = 27
    elif section2 == "Zecco":
        sect2 = 29

    section3 = st.selectbox(
        "Section Located 3", [217, "Dcentral", "Dnorth", "DNorthEx", "Dside", "DSVEtx", "EllenBrittel", "KombeDrive",
                              "Highlands", "Libuyu", "Linda", "MA [Maramba]", "Malota", "MB [Maramba]",
                              "MC [Maramba]", "MD [Maramba]", "ME [Maramba]", "Mesinja [Maramba]", "Namatama",
                              "Ngwenya", "NottieBrod", "Railways", "TownArea", "Zecco"])

    if section3 == 217:
        sect3 = 0
    elif section3 == "Dcentral":
        sect3 = 5
    elif section3 == "Dnorth":
        sect3 = 6
    elif section3 == "DNorthEx":
        sect3 = 3
    elif section3 == "Dside":
        sect3 = 8
    elif section3 == "DSVEtx":
        sect3 = 4
    elif section3 == "EllenBrittel":
        sect3 = 9
    elif section3 == "KombeDrive":
        sect3 = 11
    elif section3 == "Highlands":
        sect3 = 10
    elif section3 == "Libuyu":
        sect3 = 12
    elif section3 == "Linda":
        sect3 = 13
    elif section3 == "MA [Maramba]":
        sect3 = 14
    elif section3 == "Malota":
        sect3 = 19
    elif section3 == "MB [Maramba]":
        sect3 = 15
    elif section3 == "MC [Maramba]":
        sect3 = 16
    elif section3 == "MD [Maramba]":
        sect3 = 17
    elif section3 == "ME [Maramba]":
        sect3 = 18
    elif section3 == "Mesinja [Maramba]":
        sect3 = 21
    elif section3 == "Namatama":
        sect3 = 22
    elif section3 == "Ngwenya":
        sect3 = 23
    elif section3 == "NottieBrod":
        sect3 = 25
    elif section3 == "Railways":
        sect3 = 26
    elif section3 == "TownArea":
        sect3 = 27
    elif section3 == "Zecco":
        sect3 = 29

    section4 = st.selectbox(
        "Section Located 4", [217, "Dcentral", "Dnorth", "DNorthEx", "Dside", "DSVEtx", "EllenBrittel",
                              "KombeDrive", "Highlands", "Libuyu", "Linda", "MA [Maramba]", "Malota", "MB [Maramba]", "MC [Maramba]",
                              "MD [Maramba]", "ME [Maramba]", "Mesinja [Maramba]", "Namatama", "Ngwenya",
                              "NottieBrod", "Railways", "TownArea", "Zecco"])

    if section4 == 217:
        sect4 = 0
    elif section4 == "Dcentral":
        sect4 = 5
    elif section4 == "Dnorth":
        sect4 = 6
    elif section4 == "DNorthEx":
        sect4 = 3
    elif section4 == "Dside":
        sect4 = 8
    elif section4 == "DSVEtx":
        sect4 = 4
    elif section4 == "EllenBrittel":
        sect4 = 9
    elif section4 == "KombeDrive":
        sect4 = 11
    elif section4 == "Highlands":
        sect4 = 10
    elif section4 == "Libuyu":
        sect4 = 12
    elif section4 == "Linda":
        sect4 = 13
    elif section4 == "MA [Maramba]":
        sect4 = 14
    elif section4 == "Malota":
        sect4 = 19
    elif section4 == "MB [Maramba]":
        sect4 = 15
    elif section4 == "MC [Maramba]":
        sect4 = 16
    elif section4 == "MD [Maramba]":
        sect4 = 17
    elif section4 == "ME [Maramba]":
        sect4 = 18
    elif section4 == "Mesinja [Maramba]":
        sect4 = 21
    elif section4 == "Namatama":
        sect4 = 22
    elif section4 == "Ngwenya":
        sect4 = 23
    elif section4 == "NottieBrod":
        sect4 = 25
    elif section4 == "Railways":
        sect4 = 26
    elif section4 == "TownArea":
        sect4 = 27
    elif section4 == "Zecco":
        sect4 = 29

    section5 = st.selectbox(
        "Section Located 5", [217, "Dcentral", "Dnorth", "DNorthEx", "Dside", "DSVEtx", "EllenBrittel", "KombeDrive",
                              "Highlands", "Libuyu", "Linda", "MA [Maramba]", "Malota", "MB [Maramba]",
                              "MC [Maramba]", "MD [Maramba]", "ME [Maramba]", "Mesinja [Maramba]", "Namatama", "Ngwenya",
                              "NottieBrod", "Railways", "TownArea", "Zecco"])

    if section5 == 217:
        sect5 = 0
    elif section5 == "Dcentral":
        sect5 = 5
    elif section5 == "Dnorth":
        sect5 = 6
    elif section5 == "DNorthEx":
        sect5 = 3
    elif section5 == "Dside":
        sect5 = 8
    elif section5 == "DSVEtx":
        sect5 = 4
    elif section5 == "EllenBrittel":
        sect5 = 9
    elif section5 == "KombeDrive":
        sect5 = 11
    elif section5 == "Highlands":
        sect5 = 10
    elif section5 == "Libuyu":
        sect5 = 12
    elif section5 == "Linda":
        sect5 = 13
    elif section5 == "MA [Maramba]":
        sect5 = 14
    elif section5 == "Malota":
        sect5 = 19
    elif section5 == "MB [Maramba]":
        sect5 = 15
    elif section5 == "MC [Maramba]":
        sect5 = 16
    elif section5 == "MD [Maramba]":
        sect5 = 17
    elif section5 == "ME [Maramba]":
        sect5 = 18
    elif section5 == "Mesinja [Maramba]":
        sect5 = 21
    elif section5 == "Namatama":
        sect5 = 22
    elif section5 == "Ngwenya":
        sect5 = 23
    elif section5 == "NottieBrod":
        sect5 = 25
    elif section5 == "Railways":
        sect5 = 26
    elif section5 == "TownArea":
        sect5 = 27
    elif section5 == "Zecco":
        sect5 = 29

    def findSum(arr):
        sum = 0
        for element in arr:
            sum += element
        return sum

    # Find Total: area1, fence1, sum and ac_num
    def findTotal(arr):
        tot = 0
        for element in arr:
            tot += element
        return tot

    sum = findSum([toilet, geyser1, ac, tile1, ceil, stand])
    total = findTotal([area1, fence1, sum, ac_num])

    if st.button("Enter Button"):

        vector = (rooms, broom, year, toilet, area1, fence1,
                  geyser1, ac, ac_num, tile1, ceil, stand, sect, sum, total)

        vector2 = (rooms, broom, year, toilet, area1, fence1,
                   geyser1, ac, ac_num, tile1, ceil, stand, sect2, sum, total)

        vector3 = (rooms, broom, year, toilet, area1, fence1,
                   geyser1, ac, ac_num, tile1, ceil, stand, sect3, sum, total)

        vector4 = (rooms, broom, year, toilet, area1, fence1,
                   geyser1, ac, ac_num, tile1, ceil, stand, sect4, sum, total)

        vector5 = (rooms, broom, year, toilet, area1, fence1,
                   geyser1, ac, ac_num, tile1, ceil, stand, sect5, sum, total)

        rent_prediction = rent_model.predict([vector])

        rent_prediction2 = rent_model.predict([vector2])

        rent_prediction3 = rent_model.predict([vector3])

        rent_prediction4 = rent_model.predict([vector4])

        rent_prediction5 = rent_model.predict([vector5])

        X_array = [section, section2, section3, section4, section5]

        Y_array = [rent_prediction, rent_prediction2,
                   rent_prediction3, rent_prediction4, rent_prediction5]

        st.write('---')

        fig_report = plt.figure(
            figsize=(10, 5), dpi=300)

        plt.plot(X_array, Y_array, linewidth=2, color='r', marker='o')

        plt.title("Location Vs Rent Prices")
        plt.xlabel(f"Location: {area}")
        plt.ylabel("Estimated Rent Prices [K]")
        plt.legend(
            [f"Rooms: {rooms}, Bedrooms: {broom}, Self Contained: {in_toilet}, Tiles: {tiles}, Ceiling: {ceiling}, Fence: {fence}, Geyser: {geyser}"
             ]
        )
        plt.grid(True)

        st.pyplot(fig_report)


if selected == "About Us":
    st.title("Locas Technology")
    st.write('---')
    st.subheader("About Us")
