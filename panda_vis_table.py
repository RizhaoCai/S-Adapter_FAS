import pandas
import os
import numpy
import sys


protocol = sys.argv[1]

if protocol == '1':
    task_dict = { 0:'BASE',
        1: 'ft_REPLAY', 2: 'ft_CASIA', 3: 'ft_MSU',
        4: 'ft_HKBU', 5: 'ft_OULU', 6: 'ft_CSMAD',
        7: 'ft_CasiaSurf', 8: 'ft_WFFD', 9:'ft_WMCA', 10:'ft_Casia3DMask'
                }

    test_dict = {0: 'TestOnBASE',
                 1: 'TestOnREPLAY', 2: 'TestOnCASIA', 3: 'TestOnMSU',
                 4: 'TestOnHKBU', 5: 'TestOnOULU', 6: 'TestOnCSMAD',
                 7: 'TestOnCasiaSurf', 8: 'TestOnWFFD', 9: 'TestOnWMCA', 10: 'TestOnCasia3DMASK',
                 11: 'TestOnROSE', 12: 'TestOnCeFA'
                 }
elif protocol == '2':
    task_dict = { 0:'BASE',
                  1: 'ft_Casia3DMask', 2:'ft_WMCA',3: 'ft_WFFD', 4: 'ft_CasiaSurf', 5: 'ft_CSMAD',6: 'ft_OULU',  7: 'ft_HKBU',
                  8: 'ft_MSU',9: 'ft_CASIA',   10: 'ft_REPLAY',
         }


    test_dict = {0: 'TestOnBASE',
               1: 'TestOnCasia3DMASK', 2: 'TestOnWMCA', 3: 'TestOnWFFD', 4: 'TestOnCasiaSurf', 5: 'TestOnCSMAD', 6: 'TestOnOULU',
               7: 'TestOnHKBU',  8: 'TestOnMSU', 9: 'TestOnCASIA', 10: 'TestOnREPLAY',
               11: 'TestOnROSE', 12: 'TestOnCeFA'
               }







base_dir = protocol = sys.argv[2]# '/home/rizhao/projects/DCL_FAS/output/bc_ewc/no_ewc/adaptive_cdc_3classes/layer_attn/protocol2/'
sub_folder = 'test'
results = {

}

def get_results_from_csv(csv_path):
    df=pandas.read_csv(csv_path)

    result_dict = {'AUC': df['AUC'][0]}
    return result_dict

results = -1.0*numpy.ones([13, 11])

print(base_dir)
for i in range(0, 11):
    task_name = task_dict[i]
    task_dir = os.path.join('task_{}_{}'.format(i, task_name))


    for j in range(0, 13):
        test_name = test_dict[j]
        test_csv_name = os.path.join(base_dir,task_dir, sub_folder, test_name, 'test_frame_metrics.csv')
        #print(test_csv_name)
        if os.path.exists(test_csv_name):
            print(test_csv_name)
            result_dict=get_results_from_csv(test_csv_name)

            auc = result_dict['AUC']
            results[j,i] = auc

task_names = ['0_BASE']+[str(i)+'_'+task_dict[i] for i in range(1,11)]

x=pandas.DataFrame(results, index=test_dict.keys(), columns=task_names)
print(x)
x.to_csv(os.path.join(base_dir, 'overall_results.csv'))