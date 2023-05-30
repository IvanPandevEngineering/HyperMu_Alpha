# ChassisDyne
## Multibody, time-dependent vehicle simulations by Ivan Pandev

ChassisDyne is a chassis response calculator solving for tire load, lateral load distribution, damper force, and many other variables over time, using either real-world acceleration telemetry, or synthetic acceleration and road surface data as inputs. The vehicle model consists of 6 solid bodies: Front chassis and rear chassis (connected by a torsion spring), and 4 unsprung masses connected to the chassis by variations of springs, roll bars, dampers, etc. The full definition is found in `chassis_model.py.`

![alt text](https://github.com/IvanPandevEngineering/ChassisDyne_Alpha/blob/main/demo1.png)

The above screenshot represents a race car model, `Battle_Bimmer_28_Dec_2022`, responding to synthetic lateral and longitudinal inputs, with a bump affecting the inside tires on the right-handed turn. Demonstrated in the data is torsional chassis flex, suspension and tire harmonics, and much more. The screenshot below represents the same model responding to a replay of telemetry gathered during a real-world autocross race run, where chassis acceleration is known precisely, but surface conditions are assumed smooth.

![alt text](https://github.com/IvanPandevEngineering/ChassisDyne_Alpha/blob/main/demo2.png)
