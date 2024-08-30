# INLP Assignment 3
## Name - Pranav Gupta
## Roll No. - 2021101095

### Execution of files:
For constructing SVD word vectors, run the following command - 
python3 ./svd.py

This will execute the Python Script and a Pre-Trained Word Embeddings Model will be saved in the current Working Directory.
Link for SVD trained Word Vectors: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/pranav_g_students_iiit_ac_in/EcyBf9CEQo5LodHU6n1Uvw8B_5M8yd84ZA2Jom4FNGp5EA?e=gUA8Ou

Link for SVD trained Word Vectors Indices: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/pranav_g_students_iiit_ac_in/ETsa4z4Ib5RPhQkQmeJoRREBQSINX_h0QlYtJ1qR9yUvVg?e=x1FXs1


Link for Skip-Gram trained Word Vectors: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/pranav_g_students_iiit_ac_in/EWB-XIBjPB9Dsbiu8ff8BUsBRMXHlJy8_UOGj5kIjdBS-Q?e=RICeop


For constructing skip gram word vectors, run the following command - 
python3 ./skip-gram.py

This will execute the Python Script and a Pre-Trained Word Embeddings Model will be saved in the current Working Directory.

For loading the Pre-Trained Model for Classification, use - torch.load(< name of pre-trained model file >).

For performin downstream classification task using RNN, execute the following commands for SVD and Skip-gram respectively:

SVD: python3 ./svd-classification.py

Skip-Gram: python3 ./skip-gram-classification.py


Link for Skip-Gram Pre-Trained Model - https://iiitaphyd-my.sharepoint.com/:u:/g/personal/pranav_g_students_iiit_ac_in/EXZQEHGt0uNBmT76wDftKiAB48cS3zhJJYDnya3sJ5UcRw?e=TqMgBc

Link for SVD Pre-Trained Model - https://iiitaphyd-my.sharepoint.com/:u:/g/personal/pranav_g_students_iiit_ac_in/ESvW68e7s7JIpvVzFO9Q9isBzBSkhWVLeplmA1r6rb-VTQ?e=NdJ4ww


The execution of these files will save a pre-trained model in form of .pt file extension will can be reused later to perform down-stream classification task for any dataset using the given command for loading of pre-trained model.


### Implementation Assumptions:
Everything is done according to the specifications of the Assignment PDF and clarifications given at the time of the Assignment. 
