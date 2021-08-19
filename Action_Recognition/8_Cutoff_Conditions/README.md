# 8 - Increase system speed by identifying cutoff conditions

The script `survey_enactments.py` scans the given enactments and reports which object signals are reliably non-zero during which actions. You can use this information to write a "conditions" file, which is a human-readable text file describing certain conditions necessary in order to consider matches. For example, if the script determines that the control panel is always visible (has a non-zero element in the props sub-vector) during the action "Push(ControlPanel)", then we can write a condition that says, "The control panel must be visible to even consider matching the rolling buffer against Push(ControlPanel) database samples." The fewer comparisons the classifier has to make, the faster it will run.

## Inputs

One or more `*.enactment` files from the same detection source, that is from ground-truth or from a trained network.

## Outputs

This script prints its results to screen.

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

## Requirements
- Python
- NumPy
