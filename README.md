# Building-dynamic-scheduling-optimization
Dynamic scheduling optimization implementation in Pyomo, Python.

The U.S department of energy published in 2010, the building sector responsible for 73% of the total electricity consumption and 40% of greenhouse gas emissions.
Heating, ventilation and airconditioning(HVAC) systems generate 33% of the building energy consumption. Therefore, it is both economically and environmentally significant to reduce HVAC energy consumption. The HVAC system load depends on the season and it fluctuates throughout a day. For example, peak demand usually occurs during summer days due to air-conditioner cooling load, and winter nights due to heating load. It leads to challenging issure for Utilities to manage these fluctuating loads. The problem of load fluctuation is further increasing by incorporating renewable electricity into the grid. 

One strategy that many utilites have applied to manage these fluctuations is to adopt time-of-use pricing and thus financially incentivize customers to reduce their electiricity loads during peak hours. Thus shifting electricity from peak hours to off-peak hours promotes economic and environmental benefits.

In particular, the building energy systems have the ability to adjust the heating or cooling load over time using the thermal inertia of building structural elements by storing energy in the building envelop. Thereby, an effective approach would be HVAC  system load scheduling, which performs mathematical optimization to obtain the optimal operational plan for a time horizon in the order of two days. 

The main objective is to shift the demand from peak hours to non-peak hours and in addition studying the behavior of time discretization on dynamic optimization accuracy. The thermal comfort criteria should be followed always and for the office building case study the temperature limits within 21-24 C during occupied period and 18-27 C in unoccupied times. 

Building thermal model formulated from the literature (M. Zachar, P. Daoutidis / Journal of Process Control 74 (2019) 202–214) 
