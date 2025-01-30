import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as ex
from datetime import datetime, date

df = pd.read_csv('volvo_well_data/Volvo_data_csv.csv')

st.set_page_config(layout="wide")
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



# # Total cumulative production
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

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df['DATEPRD'],df['cum_oil'],label='Cumulative Oil', color='Brown')
ax.plot(df['DATEPRD'], df['cum_water'], label='Cumulative Water', color='blue')
# ax.plot(df['DATEPRD'], df['cum_gas'], label='Cumulative gas', color='red')

ax.fill_between(df['DATEPRD'], df['cum_oil'], color='brown', alpha=0.3)
ax.fill_between(df['DATEPRD'], df['cum_water'], color='blue', alpha=0.3) 

fig.patch.set_facecolor('black')
fig.patch.set_alpha(0.3)  # 70% opacity (30% transparency)
ax.set_facecolor('grey')
ax.set_alpha(0.1)

ax.grid(True, color='white',linestyle='--',linewidth=0.5)
ax.set_title('Cumulative Production',fontsize=18,color='white')
ax.set_xlabel('Production Timeline', fontsize=14,color='white')
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
print(well_data)

fig3,(ax3,ax4)=plt.subplots(1,2,figsize=(12,4),sharex=True)

for i, (well_name, well_df) in enumerate(well_data.items()):
    ax3.plot(well_df['DATEPRD'], well_df['BORE_OIL_VOL'],label=f"Well {op_well[i]}")

for i, (well_name, well_df) in enumerate(well_data.items()):
    ax4.plot(well_df['DATEPRD'], well_df['BORE_WAT_VOL'],label=f"Well {op_well[i]}")

ax3.set_title('Oil Production by Wells', fontsize=16)
ax4.set_title('Water Production by Wells', fontsize=16)
ax4.set_xlabel('Production Timeline', fontsize=12)
ax3.set_ylabel('Oil Volume (BORE_OIL_VOL)', fontsize=12)
ax4.set_ylabel('Wat Volume (BORE_WAT_VOL)', fontsize=12)
ax3.grid(visible=True, linestyle="--", alpha=0.8)
ax4.grid(visible=True, linestyle="--", alpha=0.8)
ax3.legend(prop={'size': 6})
ax4.legend(prop={'size': 6})

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
ax4.set_xlabel('Production Timeline', fontsize=12)
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
st.markdown(
    """
    <style>
    .center-text{
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>

    <div class="center-text">
        Individual Well Oil and Water Production
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")


# # # Well - wise production plot
fil_df_prod = df[(df['WELL_BORE_CODE'] == sel_well_prod) & (df['FLOW_KIND'] == 'production')]

avg_prod = fil_df_prod['BORE_OIL_VOL'].rolling(50).mean()
avg_wat_prod = fil_df_prod['BORE_WAT_VOL'].rolling(50).mean()

fig1,ax1=plt.subplots(figsize=(13,6),sharex=True)

# Plot for the producing well
ax1.plot(fil_df_prod['DATEPRD'], avg_prod, label=f"Oil Production, Well: {sel_well_prod}", color="brown", linewidth=2)
ax1.plot(fil_df_prod['DATEPRD'],avg_wat_prod,label=f"water Production Well: {sel_well_prod}", color="green",linewidth=2)
ax1.set_title(f"Cumulative Oil Production for Well: {sel_well_prod}", fontsize=16,color='white')
ax1.set_ylabel("Oil Volume (stb/day)", fontsize=12,color='white')
ax1.set_xlabel("Production Timeline", fontsize=12,color='white')
ax1.legend(fontsize=12, loc="upper left")
ax1.grid(visible=True, linestyle="--", alpha=0.6)

ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

fig1.autofmt_xdate()
fig1.patch.set_alpha(0.1)
ax1.set_facecolor((0, 0, 0, 0.05))

plt.tight_layout()
st.pyplot(fig1)




st.write("")
st.write("")
st.write("")
st.write("")
st.markdown(
    """
    <style>
    .center-text{
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>

    <div class="center-text">
        Plot of Injection well
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# # # Injection well plot
fil_df_inj = df[(df['WELL_BORE_CODE'] == sel_well_inj) & (df['FLOW_KIND'] == 'injection')]
avg_inj = fil_df_inj['BORE_WI_VOL'].fillna(method='ffill').rolling(50).mean()

fig2,ax2=plt.subplots(figsize=(16,6),sharex=True)

# Plot for the injecting well
ax2.plot(fil_df_inj['DATEPRD'],avg_inj, label=f"Injecting Well: {sel_well_inj}", color="green")
ax2.set_title(f"Water Injection for Well: {sel_well_inj}", fontsize=16,color='white')
ax2.set_xlabel("Injection Timeline", fontsize=12,color='white')
ax2.set_ylabel("Water Volume (units)", fontsize=12,color='white')
ax2.legend(fontsize=12, loc="upper left")
ax2.grid(visible=True, linestyle="--", alpha=0.6)

ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')

fig2.autofmt_xdate()
fig2.patch.set_alpha(0.1)
ax2.set_facecolor((0, 0, 0, 0.05)) 

st.pyplot(fig2)




st.write("")
st.write("")
st.write("")
st.markdown(
    """
    <style>
    .center-text{
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>

    <div class="center-text">
        Decline Curve Analysis
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# # # well decilne curve analysis
sel_well = st.selectbox("Select a Well", op_well)
fil_df_dca = df[(df['WELL_BORE_CODE'] == sel_well) & (df['FLOW_KIND'] == 'production')]

# st.write(fil_df_dca.describe())

df_d=fil_df_dca[['DATEPRD','BORE_OIL_VOL']]
df_d = df_d.set_index('DATEPRD')

sel_index = np.where(op_well == sel_well)[0][0]
specific_date = list(pd.to_datetime(['2014-04-25','2013-08-04','2008-06-02','2008-09-28','2014-01-17']))

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(df_d.index, df_d['BORE_OIL_VOL'], label='BORE_OIL_VOL')
# ax.set_xlabel('DATEPRD')
# ax.set_ylabel('BORE_OIL_VOL')
# ax.set_title('Oil Production Over Time')
# ax.axvline(x=specific_date[sel_index],linestyle='--',label=f"Start of production: {specific_date[sel_index].date()}")
# ax.legend()
# ax.grid(True)
# st.pyplot(fig)

df_mon = df_d.resample('ME').sum()

df_mon = df_mon.loc[specific_date[sel_index]:]

from scipy.optimize import curve_fit

df_mon['TIME_MONTHS'] = (df_mon.index.year-df_mon.index[0].year) * 12 + (df_mon.index.month - df_mon.index[0].month)
time = df_mon['TIME_MONTHS'].values
prod_rate = df_mon['BORE_OIL_VOL'].values

train_size = int(0.7 * len(df_mon))
train_time= df_mon['TIME_MONTHS'][:train_size].values
test_time= df_mon['TIME_MONTHS'][train_size:].values
train_prod_rate= df_mon['BORE_OIL_VOL'][:train_size].values
test_prod_rate= df_mon['BORE_OIL_VOL'][train_size:].values

def exponential_decline(t,qi,Di):
    return qi*np.exp(-Di*t)

def hyperbolic_decline(t,qi,Di,b):
    qfit = qi/(np.abs((1 + b * Di * t))**(1/b))
    return qfit

def harmonic_decline(t,qi,Di):
    return qi/(1+Di*t)


p_exp = [1000, 0.005]
p_hyp = [1000, 0.07, 0.005]
p_harm = [1000, 0.005]

para_exp, cov_exp = curve_fit(exponential_decline, train_time, train_prod_rate, p0=p_exp, maxfev=10000)
para_hyp, cov_hyp = curve_fit(hyperbolic_decline, train_time, train_prod_rate, p0=p_hyp, maxfev=10000)
para_harm, cov_harm = curve_fit(harmonic_decline, train_time, train_prod_rate, p0=p_harm, maxfev=10000)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

train_pred_exp = exponential_decline(train_time, para_exp[0], para_exp[1])
test_pred_exp = exponential_decline(test_time, para_exp[0], para_exp[1])

train_pred_hyp = hyperbolic_decline(train_time, para_hyp[0], para_hyp[1], para_hyp[2])
test_pred_hyp = hyperbolic_decline(test_time, para_hyp[0], para_hyp[1], para_hyp[2])

train_pred_harm = harmonic_decline(train_time, para_harm[0], para_harm[1])
test_pred_harm = harmonic_decline(test_time, para_harm[0], para_harm[1])

# Calculate MSE for training and test data
mse_train_exp = mse(train_prod_rate, train_pred_exp)
mse_test_exp = mse(test_prod_rate, test_pred_exp)

mse_train_hyp = mse(train_prod_rate, train_pred_hyp)
mse_test_hyp = mse(test_prod_rate, test_pred_hyp)

mse_train_harm = mse(train_prod_rate, train_pred_harm)
mse_test_harm = mse(test_prod_rate, test_pred_harm)

col1, col2, col3 = st.columns(3)

with col1:
    st.write("#### Exponential Decline")
    st.write("qi (initial rate):", round(para_exp[0], 2))
    st.write("Di (decline rate):", round(para_exp[1], 3))
    st.write(f"Train MSE: {mse_train_exp:.4f}")
    st.write(f"Test MSE: {mse_test_exp:.4f}")

with col2:
    st.write("#### Hyperbolic Decline")
    st.write("qi (initial rate):", round(para_hyp[0], 2))
    st.write("b (decline exponent):", round(para_hyp[1], 3))
    st.write("Di (decline rate):", round(para_hyp[2], 3))
    st.write(f"Train MSE: {mse_train_hyp:.4f}")
    st.write(f"Test MSE: {mse_test_hyp:.4f}")

with col3:
    st.write("#### Harmonic Decline")
    st.write("qi (initial rate):", round(para_harm[0], 2))
    st.write("Di (decline rate):", round(para_harm[1], 3))
    st.write(f"Train MSE: {mse_train_harm:.4f}")
    st.write(f"Test MSE: {mse_test_harm:.4f}")


fig, ax = plt.subplots(figsize=(14, 6))

ax.scatter(train_time,train_prod_rate,color='teal',label='train data',marker='^',s=25)
ax.plot(train_time,train_prod_rate,color='teal',label='train data')
ax.scatter(test_time,test_prod_rate,color='orange',label='test data',marker='^',s=25)
ax.plot(test_time,test_prod_rate,color='orange',label='test data')

# Exponential Decline curve
exp_curve_d = exponential_decline(time, *para_exp)
ax.plot(time, exp_curve_d, color='red', label='Exponential Decline')

# Hyperbolic Decline curve
hyp_curve_d = hyperbolic_decline(time, *para_hyp)
ax.plot(time, hyp_curve_d, color='blue', label='Hyperbolic Decline')

# Harmonic Decline curve
harm_curve_d = harmonic_decline(time, *para_harm)
ax.plot(time, harm_curve_d, color='green', label='Harmonic Decline')

ax.set_xlabel('Time (Months)')
ax.set_ylabel('Production Rate')
ax.set_title('Observed Data and Fitted Decline Curves')
ax.legend()
ax.grid()

st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
