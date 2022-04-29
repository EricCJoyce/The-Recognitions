# 6 - Build a database

The deployed classifier will have an on-board database of action snippets. A real-time frame buffer will compare its contents against these representative samples and find a best match. Therefore, we want this database to be rich enough that a good match is likely, but not so full that it includes misleading samples or becomes too much to search.

The `Database` class in `database.py` was designed to help profile enactments and their derived snippets. We can compute a snippet's mean signal stength, sort labels accordinly, and drop the weakest snippets. This follows from the assumption that weak snippets are not distinct and likely to become "garbage-collector" matches.

We can also impose conditions on snippets, for instance, requiring that all snippets for actions involving the hands include a hand state transition. In other words, only admit to the database snippets for "Grab(Meter)" that include a hand transitioning from open to clutching. These conditions have been found to improve accuracy in our tests.

Other conditions, such as only admitting to the database snippets for "Open Disconnect(MainFeederBox)" that include an upturn from zero to anything greater than zero for the "Open-Disconnect" object are not consistently helpful. Consider that feeder box disconnects occur in sets of three; if one disconnect is already in the "Open" state, then requiring snippets to contain an upturn from zero to non-zero will ignore this valid snippet.

For these reasons it was difficult to create a single script for this repository that would work in any case. The thing for you to do is to work with the Database class to analyze your dataset in the Python interpreter. The following code is what we used for the results reported:

```
from database import *
db = Database(stride=2, enactments=['BackBreaker1', 'Enactment1', 'Enactment2', 'Enactment3', 'Enactment4', 'Enactment5', 'Enactment6', 'Enactment7', 'Enactment9', 'Enactment10', 'MainFeederBox1', 'Regulator1', 'Regulator2'], verbose=True)

db.itemize_actions(['GT/Enactment7'])                               #  Some actions are just no good.
                                                                    #  We spotted several in Enactment7 while reviewing the video.
db.drop_enactment_action('GT/Enactment7', 0)
db.drop_enactment_action('GT/Enactment7', 1)
db.drop_enactment_action('GT/Enactment7', 2)
db.drop_enactment_action('GT/Enactment7', 3)
db.drop_enactment_action('GT/Enactment7', 4)
db.drop_enactment_action('GT/Enactment7', 5)
db.drop_enactment_action('GT/Enactment7', 6)
db.drop_enactment_action('GT/Enactment7', 7)
db.drop_enactment_action('GT/Enactment7', 8)
db.drop_enactment_action('GT/Enactment7', 9)
db.drop_enactment_action('GT/Enactment7', 10)
db.drop_enactment_action('GT/Enactment7', 11)
db.drop_enactment_action('GT/Enactment7', 12)
db.drop_enactment_action('GT/Enactment7', 13)
db.drop_enactment_action('GT/Enactment7', 14)
db.drop_enactment_action('GT/Enactment7', 15)
db.drop_enactment_action('GT/Enactment7', 18)
db.drop_enactment_action('GT/Enactment7', 19)
db.drop_enactment_action('GT/Enactment7', 20)
db.drop_enactment_action('GT/Enactment7', 21)
db.drop_enactment_action('GT/Enactment7', 22)
db.drop_enactment_action('GT/Enactment7', 23)
db.drop_enactment_action('GT/Enactment7', 24)
db.drop_enactment_action('GT/Enactment7', 25)
db.drop_enactment_action('GT/Enactment1', 21)                       #  Drop because hand is obscured.
db.drop_enactment_action('GT/Enactment1', 23)                       #  Drop because hand is obscured.
db.drop_enactment_action('GT/Enactment1', 24)                       #  Drop because hand is obscured.

db.commit()                                                         #  Chop into snippets.

db.drop_all('ConnectBlackProbe("SpareRegulator_PhaseA.BlackBay")')  #  Drop all snippets from discarded labels.
db.drop_all('ConnectRedProbe("SpareRegulator_PhaseA.RedBay")')
db.drop_all('DisconnectBlackProbe("SpareRegulator_PhaseA.BlackBay")')
db.drop_all('DisconnectRedProbe("SpareRegulator_PhaseA.RedBay")')
db.drop_all('FlipSwitch("SpareRegulator_PhaseA.Switch")')
db.drop_all('Grab("Multimeter.BlackProbe")')
db.drop_all('Grab("Multimeter.RedProbe)')
db.drop_all('TurnDial("SpareRegulator_PhaseA.Dial")')
db.drop_all('Ungrab("Multimeter.BlackProbe")')
db.drop_all('Ungrab("Multimeter.RedProbe)')

db.relabel_from_file('relabels.txt')                                #  Use shorter labels so the confusion matrix is easier to read.

db.drop_all('Open (Safety)')                                        #  Drop because it is under-represented and useless.

db.compute_signal_strength()                                        #  Compute signal strengths.
                                                                    #########################
db.output('10f-2s-GT-train.db')                                     #  Write a basline DB.  #
                                                                    #########################
                                                                    #  Mitigate Grab-Release ambiguities: admit only actual grabbings.
keepers = db.lambda_identify( db.snippets('Grab (ArcSuit)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (ArcSuit)')
db.keep('Grab (ArcSuit)', keepers)
                                                                    #  Mitigate Grab-Release ambiguities: admit only actual grabbings.
keepers = db.lambda_identify( db.snippets('Grab (F.light)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (F.light)')
db.keep('Grab (F.light)', keepers)
                                                                    #  Mitigate Grab-Release ambiguities: admit only actual grabbings.
keepers = db.lambda_identify( db.snippets('Grab (Meter)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Meter)')
db.keep('Grab (Meter)', keepers)
                                                                    #  Mitigate Grab-Release ambiguities: admit only actual grabbings.
keepers = db.lambda_identify( db.snippets('Grab (Sw. Stick)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Sw. Stick)')
db.keep('Grab (Sw. Stick)', keepers)
                                                                    #  Mitigate Grab-Release ambiguities: admit only actual grabbings.
keepers = db.lambda_identify( db.snippets('Grab (Tritector)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Tritector)')
db.keep('Grab (Tritector)', keepers)
                                                                    #  Mitigate Grab-Release ambiguities: admit only actual grabbings.
keepers = db.lambda_identify( db.snippets('Grab (Y. Tag)'), lambda seq: db.contains_hand_status_change_to(seq, 1) )
db.drop_all('Grab (Y. Tag)')
db.keep('Grab (Y. Tag)', keepers)
                                                                    #  Mitigate Release-Grab ambiguities: admit only actual releases.
keepers = db.lambda_identify( db.snippets('Release (ArcSuit)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (ArcSuit)')
db.keep('Release (ArcSuit)', keepers)
                                                                    #  Mitigate Release-Grab ambiguities: admit only actual releases.
keepers = db.lambda_identify( db.snippets('Release (F.light)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (F.light)')
db.keep('Release (F.light)', keepers)
                                                                    #  Mitigate Release-Grab ambiguities: admit only actual releases.
keepers = db.lambda_identify( db.snippets('Release (Meter)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Meter)')
db.keep('Release (Meter)', keepers)
                                                                    #  Mitigate Release-Grab ambiguities: admit only actual releases.
keepers = db.lambda_identify( db.snippets('Release (Sw. Stick)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Sw. Stick)')
db.keep('Release (Sw. Stick)', keepers)
                                                                    #  Mitigate Release-Grab ambiguities: admit only actual releases.
keepers = db.lambda_identify( db.snippets('Release (Tritector)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Tritector)')
db.keep('Release (Tritector)', keepers)
                                                                    #  Mitigate Release-Grab ambiguities: admit only actual releases.
keepers = db.lambda_identify( db.snippets('Release (Y. Tag)'), lambda seq: db.contains_hand_status_change_from(seq, 1) )
db.drop_all('Release (Y. Tag)')
db.keep('Release (Y. Tag)', keepers)
                                                                    #  Mitigate Close-Open ambiguities: admit only snippets containing a downturn in Open or Unknown props + snippets containing an upturn in Closed props.
keepers = db.lambda_identify( db.snippets('Close Dcnnct (MFB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Open') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Close Dcnnct (MFB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('Disconnect_Closed') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Close Dcnnct (MFB)')
db.keep('Close Dcnnct (MFB)', keepers)
                                                                    #  Mitigate Close-Open ambiguities: admit only snippets containing a downturn in Open or Unknown props + snippets containing an upturn in Closed props.
keepers = db.lambda_identify( db.snippets('Close (MFB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('MainFeederBox_Open') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('MainFeederBox_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Close (MFB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('MainFeederBox_Closed') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Close (MFB)')
db.keep('Close (MFB)', keepers)
                                                                    #  Mitigate Close-Open ambiguities: admit only snippets containing a downturn in Open or Unknown props + snippets containing an upturn in Closed props.
keepers = db.lambda_identify( db.snippets('Close (BB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('BackBreaker_Open') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('BackBreaker_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Close (BB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('BackBreaker_Closed') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Close (BB)')
db.keep('Close (BB)', keepers)
                                                                    #  Mitigate Close-Open ambiguities: admit only snippets containing a downturn in Open or Unknown props + snippets containing an upturn in Closed props.
keepers = db.lambda_identify( db.snippets('Close (Reg)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('Regulator_Open') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('Regulator_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Close (Reg)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('Regulator_Closed') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Close (Reg)')
db.keep('Close (Reg)', keepers)
                                                                    #  Mitigate Open-Close ambiguities: admit only snippets containing a downturn in Closed or Unknown props + snippets containing an upturn in Open props.
keepers = db.lambda_identify( db.snippets('Open Dcnnct (TFB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Closed') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Open Dcnnct (TFB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('Disconnect_Open') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Open Dcnnct (TFB)')
db.keep('Open Dcnnct (TFB)', keepers)
                                                                    #  Mitigate Open-Close ambiguities: admit only snippets containing a downturn in Closed or Unknown props + snippets containing an upturn in Open props.
keepers = db.lambda_identify( db.snippets('Open Dcnnct (MFB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Closed') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('Disconnect_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Open Dcnnct (MFB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('Disconnect_Open') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Open Dcnnct (MFB)')
db.keep('Open Dcnnct (MFB)', keepers)
                                                                    #  Mitigate Open-Close ambiguities: admit only snippets containing a downturn in Closed or Unknown props + snippets containing an upturn in Open props.
keepers = db.lambda_identify( db.snippets('Open (Reg)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('Regulator_Closed') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('Regulator_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Open (Reg)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('Regulator_Open') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Open (Reg)')
db.keep('Open (Reg)', keepers)
                                                                    #  Mitigate Open-Close ambiguities: admit only snippets containing a downturn in Closed or Unknown props + snippets containing an upturn in Open props.
keepers = db.lambda_identify( db.snippets('Open (TFB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('TransferFeederBox_Closed') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('TransferFeederBox_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Open (TFB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('TransferFeederBox_Open') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Open (TFB)')
db.keep('Open (TFB)', keepers)
                                                                    #  Mitigate Open-Close ambiguities: admit only snippets containing a downturn in Closed or Unknown props + snippets containing an upturn in Open props.
keepers = db.lambda_identify( db.snippets('Open (BB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('BackBreaker_Closed') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('BackBreaker_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Open (BB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('BackBreaker_Open') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Open (BB)')
db.keep('Open (BB)', keepers)
                                                                    #  Mitigate Open-Close ambiguities: admit only snippets containing a downturn in Closed or Unknown props + snippets containing an upturn in Open props.
keepers = db.lambda_identify( db.snippets('Open (MFB)'), (lambda seq: db.contains_downturn(seq, db.recognizable_objects.index('MainFeederBox_Closed') + 12) or db.contains_downturn(seq, db.recognizable_objects.index('MainFeederBox_Unknown') + 12)) )
keepers += db.lambda_identify( db.snippets('Open (MFB)'), lambda seq: db.contains_upturn(seq, db.recognizable_objects.index('MainFeederBox_Open') + 12) )
keepers = list(np.unique(keepers))
db.drop_all('Open (MFB)')
db.keep('Open (MFB)', keepers)
                                                                    #########################
db.output('10f-split2-stride2-GT-train-curated.db')                 #  Write a curated DB.  #
                                                                    #########################
```

## Inputs

One or more `*.enactment` files from the same detection source, that is from ground-truth or from a trained network.

## Outputs

A human-readable `*.db` database file(s). (For deployment, you may want to re-write this as a binary file.)
