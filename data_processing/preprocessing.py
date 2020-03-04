import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def data_bug_fix(bad_table):
    bad_table[np.abs(bad_table)==np.inf] = 0
    bad_table[np.isnan(bad_table)]=0   
    return bad_table

def default_preprocessing(x_table):
    x_table_lg10 = data_bug_fix(np.log(x_table))
    out_table = x_table_lg10 - np.min(x_table_lg10, axis=1).reshape([-1,1])
    return out_table

def get_poly_coeffs(input_vec, a_coef=9):
    from scipy import polyfit
    z1, resid, rank, singular, rcond = list(polyfit(np.arange(0,len(input_vec)),input_vec/np.sum(input_vec), a_coef, full=True))
    return(z1, resid)

def poly_extracter(x_table, poly_coef=9):
    x_table_coefs = np.array(list(map(lambda x: get_poly_coeffs(x/np.sum(x), a_coef=poly_coef), x_table)))
    out_table = np.array([list(i[0]) for i in x_table_coefs])
    return out_table


def dataset_from_files(experiments_folder, output_folder, sensor='S3', dataset='train'):

	# INPUTS
	#experiments_folder - path to folder with experiment dated folders, by default: './methane-propane/methane-propane-raw_data/' 
	    #following experiment folders at experiment_folder path should be exist:
	    # ['2018.12.12_20.14.18',
	    #  '2018.12.13_20.31.40',
	    #  '2018.12.14_20.53.45',
	    #  '2018.12.19_21.35.25',
	    #  '2018.12.20_21.36.43',
	    #  '2018.12.21_21.46.34',
	    #  '2019.01.11_15.51.25',
	    #  '2019.01.12_19.32.29',
	    #  '2019.01.14_19.58.00',
	    #  '2019.01.16_11.08.22',
	    #  '2019.01.18_15.19.45',
	    #  '2019.01.25_12.26.23',
	    #  '2019.01.27_12.47.08',
	    #  '2019.01.30_11.26.44',
	    #  '2019.01.31_20.25.57',
	    #  '2019.02.01_11.08.18',
	    #  '2019.02.20_22.18.39']
    #output_folder - path to folder with precessed files, by default: './methane-propane/training_data'
	#sensor - sensor string index, maybe 3 possible values: 'S1', 'S2', 'S3'
	#dataset - should data be randomly splitted to training and validating subsets. If not, "test" value should be selected.
	
	# OUTPUTS
	# there are 4 files types will be created: X, Xt Y, log 
		# X - file containing sensor responses, no colnames and indexes. rows - samples, columns - features; 
        # Xt - file containing temperature observations, no colnames and indexes. rows - samples, columns - features;
        # Y - file containing one-hot encoded classes, corresponding to each sample at X\Xt file, no colnames and indexes. rows - samples, columns - ['air', 'methane', 'propane']; 
        # log - file containing information about each sample in X\Xt\Y files:
        	#experiment - name of experimnet folder (day and time info)
        	#gas - gas type: 'air', 'methane', 'propane';
        	#concentration - gas concentation, ppm;
        	#file - file name of observation series in experiment folder where sample was initially stored; 
        	#total obs - total number of samples of selected observation series
        	#curr obs - the number of selected sample during observation series 
 

    experiment_list = os.listdir(experiments_folder)[:6]
    if dataset=='test': experiment_list = os.listdir(experiments_folder)[6:]

    glob_cnt = 0
    log_arr_train = []

    for exper_cnt in experiment_list[:6]:
        gas_list = np.sort(os.listdir(os.path.join(experiments_folder, exper_cnt)))

        for gas_cnt in gas_list:
            conc_list = np.sort(os.listdir(os.path.join(experiments_folder, exper_cnt, gas_cnt)))

            for conc_cnt in conc_list:

                file_list = np.sort(os.listdir(os.path.join(experiments_folder, exper_cnt, gas_cnt, conc_cnt)))
                file_list_sensor = [i for i in file_list  if i.split('_')[0]==sensor]

                for file_cnt in file_list_sensor:
                    raw_file = np.array(pd.read_csv(os.path.join(experiments_folder,exper_cnt,gas_cnt,conc_cnt,file_cnt), header=None))
                    R_data = raw_file[list(range(1,raw_file.shape[0],2))]
                    T_data = raw_file[list(range(0,raw_file.shape[0],2))]

                    y_data = np.zeros([R_data.shape[0], 3])
                    y_data[:,np.where(gas_cnt==gas_list)[0][0]]=1



                    if glob_cnt==0:
                        out_xr = R_data.copy()
                        out_xt = T_data.copy()
                        out_y = y_data.copy()
                        log_arr_train = np.hstack([np.repeat(np.array([exper_cnt, gas_cnt, conc_cnt, file_cnt, R_data.shape[0]]).reshape([1,-1]),R_data.shape[0], axis=0), np.arange(R_data.shape[0]).reshape(-1,1)])

                    else:
                        out_xr = np.vstack([out_xr, R_data.copy()])
                        out_xt = np.vstack([out_xt, T_data.copy()])
                        out_y = np.vstack([out_y,y_data.copy()])

                        log_arr_train = np.vstack([log_arr_train, np.hstack([np.repeat(np.array([exper_cnt, gas_cnt, conc_cnt, file_cnt,  R_data.shape[0]]).reshape([1,-1]),R_data.shape[0], axis=0), np.arange(R_data.shape[0]).reshape(-1,1)])])


                    glob_cnt+=1


    os.makedirs(output_folder, exist_ok=True)                

    if dataset=='train':
        X_train, X_val, y_train, y_val = train_test_split(out_xr, out_y, test_size=0.2, random_state=42)
        Xt_train, Xt_val, yt_train, yt_val = train_test_split(out_xt, out_y, test_size=0.2, random_state=42)

        log_arr_trn, log_arr_val, y_train2, y_val2 = train_test_split(log_arr_train,  out_y, test_size=0.2, random_state=42)

        np.savetxt(os.path.join(output_folder,sensor+'_X_train.csv'), X_train, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Xt_train.csv'), Xt_train, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_X_val.csv'), X_val, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Xt_val.csv'), Xt_val, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Y_train.csv'), y_train, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Y_val.csv'), y_val, delimiter=',')
        pd.DataFrame(log_arr_trn, columns=['experiment', 'gas','concentration', 'file', 'total obs', 'curr obs']).to_csv(os.path.join(output_folder,sensor+'_log_train.csv'))
        pd.DataFrame(log_arr_val, columns=['experiment', 'gas','concentration', 'file', 'total obs', 'curr obs']).to_csv(os.path.join(output_folder,sensor+'_log_val.csv'))


    if dataset == 'test':

        np.savetxt(os.path.join(output_folder,sensor+'_X_test.csv'), out_xr, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Xt_test.csv'), out_xt, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Y_test.csv'), out_y, delimiter=',')
        pd.DataFrame(log_arr_train, columns=['experiment', 'gas','concentration', 'file', 'total obs', 'curr obs']).to_csv(os.path.join(output_folder,sensor+'_log_test.csv'))                
         