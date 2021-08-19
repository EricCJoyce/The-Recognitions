Be careful drawing conclusions from this script. Things may happen to be true without any necessity.

Consider the following output:
```
SwitchStick is non-zero in every frame of Close Disconnect (MFB)
Disconnect_Closed is non-zero in every frame of Close Disconnect (MFB)
MainFeederBox_Closed is non-zero in every frame of Close Disconnect (MFB)
Regulator_Closed is non-zero in every frame of Close Door (Reg)
ArcHelmet is non-zero in every frame of Grab (Arcsuit)
Tritector is non-zero in every frame of Grab (Tritector)
SwitchStick is non-zero in every frame of Open Disconnect (MFB)
BackBreaker_Closed is non-zero in every frame of Open Disconnect (MFB)
Disconnect_Closed is non-zero in every frame of Open Disconnect (MFB)
MainFeederBox_Closed is non-zero in every frame of Open Disconnect (MFB)
MainFeederBox_Closed is non-zero in every frame of Open Door (BB)
MainFeederBox_Closed is non-zero in every frame of Open Door (MFB)
Regulator_Closed is non-zero in every frame of Open Door (Reg)
ControlPanel is non-zero in every frame of Push (Control Panel)
FrontBreaker is non-zero in every frame of Push (Control Panel)
Tritector is non-zero in every frame of Test (Tritector)
SwitchStick is non-zero in every frame of Ungrab (S.Stick)
Regulator_Closed is non-zero in every frame of Work (Reg)
```
The first line makes sense: one cannot close a disconnect without using the switch stick. The last line is misleading. If a user is working on a regulator, then the regulator must be open. The vector element for closed regulators is always non-zero because regulators appear in banks of three, and there always happens to be a closed regulator near whichever opened one the user is operating. We therefore do not want to make the presence of a closed regulator a condition for considering the action "Work(Regulator)".
