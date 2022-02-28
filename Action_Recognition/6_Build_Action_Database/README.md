# 6 - Build a database

The deployed classifier will have an on-board database of action snippets. A real-time frame buffer will compare its contents against these representative samples and find a best match. Therefore, we want this database to be rich enough that a good match is likely, but not so full that it includes misleading samples or becomes too much to search.

The `Database` class in `database.py` was designed to help profile enactments and their derived snippets. We can compute a snippet's mean signal stength, sort labels accordinly, and drop the weakest snippets. This follows from the assumption that weak snippets are not distinct and likely to become "garbage-collector" matches.

We can also impose conditions on snippets, for instance, requiring that all snippets for actions involving the hands include a hand state transition. In other words, only admit to the database snippets for "Grab(Meter)" that include a hand transitioning from open to clutching. These conditions have been found to improve accuracy in our tests.

Other conditions, such as only admitting to the database snippets for "Open Disconnect(MainFeederBox)" that include an upturn from zero to anything greater than zero for the "Open-Disconnect" object are not consistently helpful. Consider that feeder box disconnects occur in sets of three; if one disconnect is already in the "Open" state, then requiring snippets to contain an upturn from zero to non-zero will ignore this valid snippet.

For these reasons it was difficult to create a single script for this repository that would work in any case. The thing for you to do is to work with the Database class to analyze your dataset in the Python interpreter. The following code is what we used for the results reported:

```
from database import *
db = Database(enactments=['BackBreaker1', 'Enactment1', 'Enactment2', 'Enactment3', 'Enactment4', 'Enactment5', 'Enactment6', 'Enactment7', 'Enactment9', 'Enactment10', 'MainFeederBox1', 'Regulator1', 'Regulator2'], verbose=True)

#  Some actions are just no good. We spotted several in Enactment7 while reviewing the video.

db.itemize_actions(['GT/Enactment7'])
db.drop_enactment_action('GT/Enactment7', 5)
db.drop_enactment_action('GT/Enactment7', 6)
db.drop_enactment_action('GT/Enactment7', 9)
db.drop_enactment_action('GT/Enactment7', 10)
db.drop_enactment_action('GT/Enactment7', 17)
db.drop_enactment_action('GT/Enactment7', 18)
db.drop_enactment_action('GT/Enactment7', 21)
db.drop_enactment_action('GT/Enactment7', 22)

#  Chop into snippets.

db.commit()

#  Drop all snippets from discarded labels.

db.drop_all('ConnectBlackProbe("SpareRegulator_PhaseA.BlackBay")')
db.drop_all('ConnectRedProbe("SpareRegulator_PhaseA.RedBay")')
db.drop_all('DisconnectBlackProbe("SpareRegulator_PhaseA.BlackBay")')
db.drop_all('DisconnectRedProbe("SpareRegulator_PhaseA.RedBay")')
db.drop_all('FlipSwitch("SpareRegulator_PhaseA.Switch")')
db.drop_all('Grab("Multimeter.BlackProbe")')
db.drop_all('Grab("Multimeter.RedProbe)')
db.drop_all('TurnDial("SpareRegulator_PhaseA.Dial")')
db.drop_all('Ungrab("Multimeter.BlackProbe")')
db.drop_all('Ungrab("Multimeter.RedProbe)')

#  Use shorter labels so the confusion matrix is easier to read.

db.relabel_from_file('relabels.txt')

#  Compute signal strengths.

db.compute_signal_strength()

#  Mitigate Grab-Release ambiguities by only admitting to the database subsets that include indicative hand-transitions.

keepers = db.lambda_identify( db.snippets('Grab (ArcSuit)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (ArcSuit)')
db.keep('Grab (ArcSuit)', keepers)
db.itemize('Grab (ArcSuit)')

keepers = db.lambda_identify( db.snippets('Grab (F.light)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (F.light)')
db.keep('Grab (F.light)', keepers)
db.itemize('Grab (F.light)')

keepers = db.lambda_identify( db.snippets('Grab (Meter)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Meter)')
db.keep('Grab (Meter)', keepers)
db.itemize('Grab (Meter)')

keepers = db.lambda_identify( db.snippets('Grab (Sw. Stick)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Sw. Stick)')
db.keep('Grab (Sw. Stick)', keepers)
db.itemize('Grab (Sw. Stick)')

keepers = db.lambda_identify( db.snippets('Grab (Tritector)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Tritector)')
db.keep('Grab (Tritector)', keepers)
db.itemize('Grab (Tritector)')

keepers = db.lambda_identify( db.snippets('Grab (Y. Tag)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Y. Tag)')
db.keep('Grab (Y. Tag)', keepers)
db.itemize('Grab (Y. Tag)')

keepers = db.lambda_identify( db.snippets('Release (ArcSuit)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (ArcSuit)')
db.keep('Release (ArcSuit)', keepers)
db.itemize('Release (ArcSuit)')

keepers = db.lambda_identify( db.snippets('Release (F.light)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (F.light)')
db.keep('Release (F.light)', keepers)
db.itemize('Release (F.light)')

keepers = db.lambda_identify( db.snippets('Release (Meter)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Meter)')
db.keep('Release (Meter)', keepers)
db.itemize('Release (Meter)')

keepers = db.lambda_identify( db.snippets('Release (Sw. Stick)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Sw. Stick)')
db.keep('Release (Sw. Stick)', keepers)
db.itemize('Release (Sw. Stick)')

keepers = db.lambda_identify( db.snippets('Release (Tritector)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Tritector)')
db.keep('Release (Tritector)', keepers)
db.itemize('Release (Tritector)')

keepers = db.lambda_identify( db.snippets('Release (Y. Tag)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Y. Tag)')
db.keep('Release (Y. Tag)', keepers)
db.itemize('Release (Y. Tag)')

#  Write to file.

db.output('10f-train.db')
```

## Inputs

One or more `*.enactment` files from the same detection source, that is from ground-truth or from a trained network.

## Outputs

A human-readable `*.db` database file. (For deployment, you may want to re-write this as a binary file.)
