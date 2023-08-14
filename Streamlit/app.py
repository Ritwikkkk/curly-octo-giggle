import pandas as pd
import numpy as np
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

data = {
    'Name': ['A','B','C','D'],
    'Age': [12,13,14,15]
}
df = pd.DataFrame(data)

gd = GridOptionsBuilder.from_dataframe(df)
gd.configure_pagination(enabled=True)
# gd.configure_default_column(editable=True,groupable=True)
gd.configure_side_bar()
gd.configure_selection(selection_mode = "multiple", use_checkbox = True)
gridOptions = gd.build()
gridOptions["columnDefs"][0]["checkboxSelection"] = True
gridOptions["columnDefs"][0]["wrapText"] = True
gridOptions["columnDefs"][0]["autoHeight"] = True
gridOptions["columnDefs"][0]["headerCheckboxSelection"] = True

st.sidebar.title(" Select the Names")
with st.sidebar:
    AgGrid(df,gridOptions=gridOptions,
              update_mode= GridUpdateMode.SELECTION_CHANGED,
                     height = 270,
                     allow_unsafe_jscode=True,
                    #enable_enterprise_modules = True,
                     theme = 'alpine')
