Walk Detection and Step Counting on Unconstrained Smartphones - The Dataset

Dataset description

The provided dataset contains time-annotated sensor traces obtained from smartphones in typical, unconstrained use while walking. 27 participants were asked to walk a route at three different walk paces: starting with normal, followed by fast, and ending with slow. Each participant walked the same distance and changed her/his speed at markers installed on the path. The participant carried one or two phones placed at varying positions (in a front or back trouser pocket, in a backpack/handbag, or in a hand with or without simultaneous typing).

This dataset has been used for a fair, quantitative comparison of standard algorithms for walk detection and step counting. For more details about these experiments and the obtained results please refer to the full paper on the left.

Naming scheme

The name of each trace is of the following form:

        UID.EID_GENDER_AGE_HEIGHT_PLACEMENT.out or 
        UID.EID_GENDER_AGE_HEIGHT_PLACEMENT.dat

where the components of the name have the following meaning: participant id (UID), experiment id (EID), gender (GENDER), age range (AGE), height range in cm (HEIGHT) and phone placement (PLACEMENT).

Two different traces obtained from the same user that have the same experiment id denote the run in which the user was carrying more than one phone at the same time. Please note that the number of timestamps in such two traces might not be identical as the sensor logging application on the two phones may not have been started at the same time instant.

Content and format

Each trace in the database contains sensor readings from accelerometer, gyroscope and magnetometer. Each sensor sample is annotated with the timestamp when it was taken (as reported by Android). All three sensors were sampled at frequency of around 100 Hz.

Dataset also contains the information about the ground truth associated with each experiment (i.e. the number of steps taken and the order numbers of samples when the walk has begun and ended). This data has been obtained from the video sequences taken during experiments which have been omitted here for privacy reasons. Initial and trailing oscillations (outside the walking section of the trace) should be ignored.

Parsing code

The provided python scripts can be used for parsing the traces and extracting the magnitude of accelerometer readings.



Special thanks to the student Bono Xu who helped us recruit the participants and conduct the experiments.
