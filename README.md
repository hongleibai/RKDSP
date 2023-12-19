# RKDSP
Data
Data composition: Similarity_Matrix_Drugs.txt in the folder Model/RKDSP/data/drug is the drug-drug similarity matrix based on chemical substructure. drug_drug_sim_dis.txt is the drug-drug similarity matrix based on associated diseases. mat_drug_se .txt is the drug-side effect association matrix. se_se_sim.txt is the side effect similarity matrix. The drug and side effect name files are saved in the Data folder and named drugname.txt and se.txt respectively.

Data sources：The dataset of drug-related side effect associations prediction is extracted from Luo et al [1], containing drug-side effect associations, drug-disease associations, and drug-drug similarities based on chemical substructure. A total of 80,164 pairs of associations involving 708 drugs and 4192 side effects were derived from the SIDER database. The original drug-disease associations are extracted from the Comparative Toxicogenomics database, which covers 199,214 pairs of associations for 708 drugs and 5,603 diseases.

[1]Luo Y, Zhao X, Zhou J, et al. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature Communications 2017;8(1):573.
Data preprocessing
We first used RKDSP/Process/drug_se_process.py to construct a drug-side effect heterogeneous graph based on the initial data. The heterogeneous graph is projected by different meta-paths to get semantic subgraphs containing different semantic information. The generated semantic subgraphs were saved in Model/RKDSP/data/drug folder.
Training process
Run Model/RKDSP/src/main.py to load the preprocessed semantic subgraphs and perform training and testing. Model details are saved in the file Model/RKDSP/src/model.py
Results
We applied the trained RKDSP model to obtain the top 30 candidate side effects for each drug. The results were saved as ST3.xls.
