import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import utils
"""

python3 visual.py --csv_file <dir+filename.csv>  --cand_num <num> --cast_name <cast_name>

ex:

python3 visual.py --csv_file ./IMDb/sample_submission.csv  --cand_num 5 --cast_name tt1840309_nm0000171

"""




def main(opts):
    #---------------------------------------------- #
    # Visualize the ranked result from csv file     #
    # --------------------------------------------- #
    csv_result = pd.read_csv(opts.csv_file)
    cast_name = opts.cast_name
    movie_name = cast_name.split('_')[0]
    
    cast_dir = os.path.join(opts.test_dir+movie_name+'/cast/'+cast_name+'.jpg')
    cast_img = Image.open(cast_dir)
    cand_dir = os.path.join(opts.test_dir+movie_name+'/candidates/')
  
    cast_img=np.array(cast_img)
    cand_num =int(opts.cand_num)

    target = csv_result[csv_result['Id']==cast_name]
    cands = target['Rank'].tolist()[0].split(' ')
    plt.figure(figsize=(20,10))
    plt.subplot(1, cand_num+1, 1)
    plt.imshow(cast_img)
    plt.axis('off')
    for i in range(cand_num):
        cands_img = Image.open(os.path.join(cand_dir+cands[i]+'.jpg'))
        plt.subplot(1, cand_num+1, i+2)
        plt.imshow(cands_img)
        plt.axis('off')
    
    plt.savefig("demo.jpg")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cast_name',default = 'tt1840309_nm0000171', type=str)
    parser.add_argument('--csv_file', default='./IMDb/sample_submission.csv',type=str, help="If 'csv_file' contains value, read csv_file.csv")
    parser.add_argument('--test_dir', default='./IMDb/test/', type=str, help='./test_data')
    parser.add_argument('--cand_num', default=5, help='numbers of demo')
    opts = parser.parse_args()

    utils.details(opts)
    main(opts)