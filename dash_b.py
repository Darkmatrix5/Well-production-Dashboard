import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as ex
from datetime import datetime, date

df = pd.read_csv('volvo_well_data/Volvo_data_csv.csv')

st.title('Volvo Production Dashboard')

st.sidebar.image('equinor_logo.png',caption='Equinor VOLVE field',)

op_well=df[df['FLOW_KIND']=='production']['WELL_BORE_CODE'].unique()
sel_well_prod=st.sidebar.radio('Production Wells',op_well)

inj_well=df[df['FLOW_KIND']=='injection']['WELL_BORE_CODE'].unique()
sel_well_inj=st.sidebar.radio('Injection Wells',inj_well)


# start_date = date(2008, 1, 1)
# end_date = date(2017, 1, 1)

# selected_date = st.sidebar.slider(
#     "Select a date",
#     min_value=start_date,
#     max_value=end_date,
#     value=start_date,
#     format="YYYY-MM-DD"
# )

# st.write(f"You selected: {selected_date}")



# # # Total cumulative production
total_oil = df['BORE_OIL_VOL'].sum() 
total_gas = df['BORE_GAS_VOL'].sum() 
total_water = df['BORE_WAT_VOL'].sum()
# total_wat_inj = df['BORE_WI_VOL'].sum()

c1,c2,c3 = st.columns(3) 
c1.metric(label='Cumulative Oil To Date',value = total_oil,delta=" Sm3") 
c2.metric(label='Cumulative Water To Date',value = total_water,delta=" Sm3") 
c3.metric(label='Cumulative Gas To Date',value = total_gas,delta=" Sm3")
# c3.metric(label='Cum Inj Water To Date',value = total_wat_inj,delta=" Sm3")



st.write("")
st.write("")


# # # PLotting cumulative Production
df['DATEPRD']= pd.to_datetime(df['DATEPRD'])
df = df.sort_values(by='DATEPRD')

df['cum_oil'] = df['BORE_OIL_VOL'].cumsum()
df['cum_water'] = df['BORE_WAT_VOL'].cumsum()
# df['cum_gas'] = df['BORE_GAS_VOL'].cumsum()

df['cum_oil'] = df['cum_oil'].fillna(method='ffill')  # Forward fill
df['cum_water'] = df['cum_water'].fillna(method='ffill')  # Forward fill

fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(df['DATEPRD'],df['cum_oil'],label='Cumulative Oil', color='Brown')
ax.plot(df['DATEPRD'], df['cum_water'], label='Cumulative Water', color='blue')
# ax.plot(df['DATEPRD'], df['cum_gas'], label='Cumulative gas', color='red')

ax.fill_between(df['DATEPRD'], df['cum_oil'], color='brown', alpha=0.3)  # Brown below oil curve
ax.fill_between(df['DATEPRD'], df['cum_water'], color='blue', alpha=0.3) 

fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.3)  # 70% opacity (30% transparency)
ax.set_facecolor('grey')
ax.set_alpha(0.1)

ax.grid(True, color='white',linestyle='--',linewidth=0.5)
ax.set_title('Cumulative Production',fontsize=18,color='white')
ax.set_xlabel('Date', fontsize=14,color='white')
ax.set_ylabel('Cumulative Production (units)',fontsize=14,color='white')
ax.legend()

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

st.pyplot(fig)



st.write("")
st.write("")
st.write("")
st.write("")


# # # all well raw graph
# Store dataframes for each well in a dictionary
well_data = {f"well_{i}": df[df['WELL_BORE_CODE'] == op_well[i]] for i in range(len(op_well))}

fig3,(ax3,ax4)=plt.subplots(2,1,figsize=(12,10),sharex=True)

for i, (well_name, well_df) in enumerate(well_data.items()):
    ax3.plot(well_df['DATEPRD'], well_df['BORE_OIL_VOL'],label=f"Well {op_well[i]}")

for i, (well_name, well_df) in enumerate(well_data.items()):
    ax4.plot(well_df['DATEPRD'], well_df['BORE_WAT_VOL'],label=f"Well {op_well[i]}")

ax3.set_title('Oil Production by Wells', fontsize=16)
ax4.set_title('Water Production by Wells', fontsize=16)
ax4.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Oil Volume (BORE_OIL_VOL)', fontsize=12)
ax4.set_ylabel('Wat Volume (BORE_WAT_VOL)', fontsize=12)
ax3.grid(visible=True, linestyle="--", alpha=0.8)
ax4.grid(visible=True, linestyle="--", alpha=0.8)
ax3.legend()
ax4.legend()

fig3.patch.set_facecolor('#f0f0f0')
ax3.set_facecolor('#e0e0e0')
ax4.set_facecolor('#e0e0e0')

st.pyplot(fig3)



# # # rolling average
fig3,(ax3,ax4)=plt.subplots(2,1,figsize=(12,10),sharex=True)

for i, (well_name, well_df) in enumerate(well_data.items()):
    ax3.plot(well_df['DATEPRD'], well_df['BORE_OIL_VOL'].rolling(60).mean(),label=f"Well {op_well[i]}")

for i, (well_name, well_df) in enumerate(well_data.items()):
    ax4.plot(well_df['DATEPRD'], well_df['BORE_WAT_VOL'].rolling(60).mean(),label=f"Well {op_well[i]}")

ax3.set_title('Oil production by Wells', fontsize=16)
ax4.set_title('Water production by Wells', fontsize=16)
ax4.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Oil Volume (BORE_OIL_VOL)', fontsize=12)
ax4.set_ylabel('Water Volume (BORE_WAT_VOL)', fontsize=12)
ax3.grid(visible=True, linestyle="--", alpha=0.8)
ax4.grid(visible=True, linestyle="--", alpha=0.8)
ax3.legend()
ax4.legend()

fig3.patch.set_facecolor('#f0f0f0')
ax3.set_facecolor('#e0e0e0')
ax4.set_facecolor('#e0e0e0')

st.pyplot(fig3)






st.write("")
st.write("")
st.write("")
st.write("")
st.write("Indivisual well visualization")


# # # Well - wise production plot
fil_df_prod = df[(df['WELL_BORE_CODE'] == sel_well_prod) & (df['FLOW_KIND'] == 'production')]
fil_df_inj = df[(df['WELL_BORE_CODE'] == sel_well_inj) & (df['FLOW_KIND'] == 'injection')]

avg_prod = fil_df_prod['BORE_OIL_VOL'].rolling(50).mean()
avg_inj = fil_df_inj['BORE_WI_VOL'].fillna(method='ffill').rolling(50).mean()

fig1,(ax1,ax2)=plt.subplots(2,1,figsize=(12,10),sharex=True)

# Plot for the producing well
ax1.plot(fil_df_prod['DATEPRD'],avg_prod,label=f"Producing Well: {sel_well_prod}", color="brown")
ax1.set_title(f"Cumulative Oil Production for Well: {sel_well_prod}", fontsize=16,color='white')
ax1.set_ylabel("Oil Volume (units)", fontsize=12,color='white')
ax1.legend(fontsize=12, loc="upper left")
ax1.grid(visible=True, linestyle="--", alpha=0.6)

# Plot for the injecting well
ax2.plot(fil_df_inj['DATEPRD'],avg_inj, label=f"Injecting Well: {sel_well_inj}", color="green")
ax2.set_title(f"Water Injection for Well: {sel_well_inj}", fontsize=16,color='white')
ax2.set_xlabel("Date", fontsize=12,color='white')
ax2.set_ylabel("Water Volume (units)", fontsize=12,color='white')
ax2.legend(fontsize=12, loc="upper left")
ax2.grid(visible=True, linestyle="--", alpha=0.6)

ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')

fig1.patch.set_alpha(0.1)
ax1.set_facecolor((0, 0, 0, 0.05)) 
ax2.set_facecolor((0, 0, 0, 0.05)) 

plt.tight_layout()
st.pyplot(fig1)



st.markdown("<hr>", unsafe_allow_html=True)
