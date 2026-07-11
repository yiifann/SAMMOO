## Updates 07-11 Yifan

1. Update configs/datasets.json

2. Update requirements.txt

3. Upload patient_data_od.xlsx and patient_data_os.xlsx

4. Update notebooks/PAPILA.ipynb   (recommended: Run this ipynb file in colab instead of in terminal)

   ​	-- Change path to your path (Only 1 path need to be changed at the beginning)

5. Upload result.txt (results about PAPULA, cmd and details can be found in it)

   



### PAPILA Datasets Pre-Processing

1. Download PAPILA Datasets: https://figshare.com/articles/dataset/PAPILA/14798004/1?file=28454352

2. Unzip PAPILA.zip and Rename this folder to PAPILA

3. Open PAPILA/ClinicalData
4. Delete orginal **patient_data_od.xlsx** and **patient_data_os.xlsx** 
5. Download new **patient_data_od.xlsx** and **patient_data_os.xlsx** from github and put them in the folder PAPILA/ClinicalData/



Updates 07-11 End -- Yifan




## Acknowledgement / Credits

This project is a modified version of the original work by Yongshuo Zong (https://github.com/ys-zong/MEDFAIR.git). 

- **Original Repository:** [MEDFAIR: Benchmarking Fairness for Medical Imaging](https://github.com/ys-zong/MEDFAIR.git)
- **Changes made:** We add one new bias mitigation algorithm -- SAMMOO

Thank you to the original author for their great work!
