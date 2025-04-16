#This make point by point comparision of compressed quanitities with raw values
import numpy as np
from readers import HaloReader,SnapshotReader
from correlations import PowerSpectrum
from time import time
import gc

import h5py

import socket
print(socket.gethostname())

from scipy.spatial import cKDTree
import time
import os
import pickle


def ltime(date=True):
    tm=time.localtime()
    datestr=str(tm.tm_mday)+'/'+str(tm.tm_mon)+'/'+str(tm.tm_year)
    tmstr=str(tm.tm_hour)+':'+str(tm.tm_min)+':'+str(tm.tm_sec)
    if(date):
        tmstr=datestr+' '+tmstr
    return tmstr


#This function is used for index matching between compress and raw quantities
def find_locations_kdtree(arr1, arr2):
    # Reshape arrays to make them 2D as required by KDTree
    arr1_2d = np.array(arr1).reshape(-1, 1)
    arr2_2d = np.array(arr2).reshape(-1, 1)

    # Create KDTree from arr2
    tree = cKDTree(arr2_2d)

    # Find nearest neighbors and distances
    distances, indices = tree.query(arr1_2d, k=1)

    matched_mask=distances<1e-4
    indices[~matched_mask]=-1


    #print('matched/un-matched:',np.sum(matched_mask),np.sum(~matched_mask))

    return indices,matched_mask


def compare_quant(indices,mask,sr_o,sr_comp,quant_list=['ids'],subsamples_comp=[1],
                  raw_nfile=[0],res_dic={},Lbox=None):
    
    if('pos' in quant_list):
        assert Lbox is not None

    for qq,quant in enumerate(quant_list):
        quant_o = sr_o.read_block(quant,raw_nfile=raw_nfile).T
        quant_comp = sr_comp.read_block(quant,subsamples=subsamples_comp,
                                        raw_nfile=raw_nfile)

        nsh=quant_o.shape
        if(len(nsh)==1):
            ncol=1
        else:
            ncol=quant_o.shape[1]
        
        diff=quant_o[mask]-quant_comp[indices[mask]]
        if(quant=='pos'): #apply the pbc rules
            itmp=diff>(0.5*Lbox)
            diff[itmp]=diff[itmp]-Lbox
            itmp=diff<(-0.5*Lbox)
            diff[itmp]=Lbox-diff[itmp]

        min_o=quant_o[mask].min(axis=0)
        max_o=quant_o[mask].max(axis=0)
        min_comp=quant_comp[indices[mask]].min(axis=0)
        max_comp=quant_comp[indices[mask]].max(axis=0)
        

        if(quant not in res_dic.keys()):
            sum_quant={'sum':np.zeros(ncol),'sum2':np.zeros(ncol),'count':0,'ncol':ncol,
                       'diff_min':1000+np.zeros(ncol),'diff_max':-1000+np.zeros(ncol),
                        'raw_min':1000+np.zeros(ncol),'raw_max':-1000+np.zeros(ncol),
                        'comp_min':1000+np.zeros(ncol),'comp_max':-1000+np.zeros(ncol),
                        'hist':{'xmin':-10,'xmax':10,'nx':1000},
                       'hist_scale':{'pos':100,'vel':1.0,'potential':0.01,'ids':1},
                       'units':{'pos':r'Mpc/h','vel':'km/s','potential':'no','ids':'number'}
                    }
            sum_quant['hist']['xbin_edge']=np.linspace(sum_quant['hist']['xmin'],
                                              sum_quant['hist']['xmax'],
                                              sum_quant['hist']['nx']+1)
            sum_quant['hist']['xbin_mid']=0.5*(sum_quant['hist']['xbin_edge'][1:]+sum_quant['hist']['xbin_edge'][:-1])
            sum_quant['hist']['hcount']=np.zeros((sum_quant['hist']['nx'],ncol))
            res_dic[quant]=sum_quant

        #calculate sum and sum2    
        mult_factor=res_dic[quant]['hist_scale'][quant]
        res_dic[quant]['sum']=res_dic[quant]['sum'] + diff.sum(axis=0)
        res_dic[quant]['sum2']=res_dic[quant]['sum2'] + np.sum(np.power(diff,2),axis=0)
        res_dic[quant]['count'] = res_dic[quant]['count']+diff.shape[0]
        
        #calculate the minimum and maximum
        #res_dic[quant]['min_max']['diff_min']=min(res_dic[quant]['min_max']['diff_min'],diff.min(axis=0))
        #res_dic[quant]['min_max']['diff_max']=max(res_dic[quant]['min_max']['diff_max'],diff.max(axis=0))
        #res_dic[quant]['min_max']['raw_min']=min(res_dic[quant]['min_max']['raw_min'],quant_o.min(axis=0))
        
        
        if(ncol==1):
            hcount,hbin = np.histogram(diff*mult_factor,bins=res_dic[quant]['hist']['xbin_edge'])
            res_dic[quant]['hist']['hcount'][:,0]=res_dic[quant]['hist']['hcount'][:,0]+hcount
            #calculate the minimum and maximum
            res_dic[quant]['diff_min']=min(res_dic[quant]['diff_min'],diff.min())
            res_dic[quant]['diff_max']=max(res_dic[quant]['diff_max'],diff.max())

            res_dic[quant]['raw_min']=min(res_dic[quant]['raw_min'],min_o)
            res_dic[quant]['raw_max']=max(res_dic[quant]['raw_max'],max_o)

            res_dic[quant]['comp_min']=min(res_dic[quant]['comp_min'],min_comp)
            res_dic[quant]['comp_max']=max(res_dic[quant]['comp_max'],max_comp)
        else:
            for ii in range(0,ncol):
                hcount,hbin = np.histogram(diff[:,ii]*mult_factor,bins=res_dic[quant]['hist']['xbin_edge'])
                res_dic[quant]['hist']['hcount'][:,ii]=res_dic[quant]['hist']['hcount'][:,ii]+hcount
                #calculate the minimum and maximum
                res_dic[quant]['diff_min'][ii]=min(res_dic[quant]['diff_min'][ii],diff[:,ii].min())
                res_dic[quant]['diff_max'][ii]=max(res_dic[quant]['diff_max'][ii],diff[:,ii].max())
                
                res_dic[quant]['raw_min'][ii]=min(res_dic[quant]['raw_min'][ii],min_o[ii])
                res_dic[quant]['raw_max'][ii]=max(res_dic[quant]['raw_max'][ii],max_o[ii])
                                
                res_dic[quant]['comp_min'][ii]=min(res_dic[quant]['comp_min'][ii],min_comp[ii])
                res_dic[quant]['comp_max'][ii]=max(res_dic[quant]['comp_max'][ii],max_comp[ii])
    
    return res_dic


def plot_histogram_stats(fig,axarr,pl,data_dict,quant='pos'):
    """
    Plot histograms from the data dictionary and display statistics
    
    Parameters:
    data_dict: Dictionary containing histogram data and statistics
    """
    # Extract relevant data
    hist_data = data_dict['hist']
    bin_mids = hist_data['xbin_mid']
    counts = hist_data['hcount']
    
    # Calculate statistics for each column
    means = data_dict['sum'] / data_dict['count']
    # Calculate variance using sum2 and sum
    variances = (data_dict['sum2'] / data_dict['count']) - (means ** 2)
    std_dev=np.sqrt(variances) 
    nsigma=5;
    
    # Colors for different columns
    colors = ['blue', 'red', 'green']
    column_names = [f'Column {i+1}' for i in range(data_dict['ncol'])]
    
    stats_text=''
    # Plot histograms
    for i in range(data_dict['ncol']):
        ax=axarr[i]
        ax.plot(bin_mids/data_dict['hist_scale'][quant], counts[:, i], color=colors[i], label=column_names[i])
        #ax.fill_between(bin_mids, counts[:, i], alpha=0.3, color=colors[i])
        ax.set_title(f'{column_names[i]} {quant}')
        ax.set_xlabel(r'$ %s $'%(data_dict['units'][quant]))
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        ax.set_yscale('log')
        
        # Add statistical information as text
        stats_text=stats_text+'\n'+f'{i}: {means[i]:.4f} \pm {np.sqrt(variances[i]):.4f}'
        ax.set_xlim([-nsigma*std_dev[i],nsigma*std_dev[i]]) 

    stats_text=stats_text+'\n'+f'Npart: {data_dict["count"]}'
    ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    
    pl.tight_layout()

def get_matched_id(sr_o,sr_comp,subsamples=[1,3],raw_nfile=[0]):
    tkey='ids'
    id_raw=sr_o.read_block(tkey,raw_nfile=raw_nfile)
    res_dic={}
    for ss in subsamples:
        id_comp=sr_comp.read_block(tkey,subsamples=[ss],raw_nfile=raw_nfile)
        indices,matched_mask = find_locations_kdtree(id_raw, id_comp)
        print(ltime(),'subsample:',ss,'Number of id matched: ',matched_mask.sum(),indices.size)
        res_dic=compare_quant(indices,matched_mask,sr_o,sr_comp,
                      quant_list=['pos','vel'],subsamples_comp=[ss],
                              raw_nfile=raw_nfile,
                              res_dic=res_dic,Lbox=sr_o.Lbox)

    
    return res_dic



def main():
    #sim_stem = 'sinhagad/default256'
    #on pawna
    #sim_stem='sinhagad/default128'
    #on pegasus testing sinhagad
    #sim_stem='/scratch/csaee/sinhagad/data/sims/var_Om_m256/'
    if(False):#Sinhagad
        sim_name='Sinhagad'
        sim_stem='/mnt/home/project/chpc2501005/shadab/test_sinhagad/'
        snap = 200
        real = 1
        grid = 256
        downsample=0
        Npmin = 30
        Seed = 42
    else:
        sim_name='Sahyadri'
        #sahyadri sim on pegasus
        sim_stem='/mnt/home/project/chpc2501005/data/sims/sahyadri/default2048/'
        snap = 72
        real = 1
        grid = 2048
        downsample=0
        Npmin = 30
        Seed = 42
    # The compressed file will be split in several percentage subsample
    # One if free to choose this setting with following requirement
    # a) The sum of the subsamples must add to 100
    # b) The elements must be unique that is same subsample cannot be created twice because the file neme convention
    

    subsamples=[1,3,5,9,10,12,15,20,25]
    #subsamples=[1]#,3,5,9,10,12,15,20,25]
    quant_write=['positions','ids','velocities','potentials']

    print(ltime(),' Comparing the compression results with raw value')
    print(f'sim_stem: {sim_stem}')
    print(f'snap: {snap}')

    # The directory in the output directory to store compressed output
    compressed_fileroot='compressed'

    # This call reads the original snapshot and write the compressed file
    #%memit sr.compress_snapshot( subsamples,quant_write=quant_write,optimized=True)

    # By setting use_compress=False we are asking to read from original file for snapshot
    sr_o = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap,read_header=True,use_compressed=False)
    # By setting use_compress=True we are asking to read from compressed file
    sr_comp = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap,read_header=True,use_compressed=True)

    print(ltime(),'Created Objects for reading compressed and raw files')
    
    #res_dic=compare_quant(indices,matched_mask,sr_o,sr_comp,
    #                      quant_list=['pos'],subsamples_comp=[1],res_dic={})

   
    global_dic={}
    dic_froot=f'Histograms/compare_{sim_name}_{snap}'
    for ss,sub_this in enumerate(subsamples):
        dic_fname=f'{dic_froot}_sub{sub_this}.pkl'
        if(os.path.isfile(dic_fname)):

            # Loading with pickle
            with open(dic_fname, 'rb') as f:  # Note the 'rb' mode for binary reading
                loaded_data = pickle.load(f)
            print(ltime(),f'Loaded dictionary {dic_fname}')
            global_dic[sub_this]=loaded_data
        else:
            print(ltime(),'Calling the get_matched_id function with subsample',sub_this)
            res_dic=get_matched_id(sr_o,sr_comp,subsamples=[sub_this],raw_nfile=[0])
            #save result dictionary to a file

            # Saving with pickle
            with open(dic_fname, 'wb') as f:  # Note the 'wb' mode for binary writing
                pickle.dump(res_dic, f)
            print(ltime(),f'Written dictionary {dic_fname}')
            global_dic[sub_this]=res_dic


    if(False):
        import pylab as pl
        import matplotlib.pyplot as plt
        import matplotlib.colors as pltcol
        from matplotlib.gridspec import GridSpec
        # Using the provided data directly
        # Create the plot
        res_dic={}
        for ss,sub_this in enumerate(global_dic.keys()):
            # Create figure with GridSpec for better layout
            figwidth=15;figheight=5
            ncol=3;nrow=2
            fig,axarr=pl.subplots(nrow,ncol,sharex=False,sharey=False,
                          figsize=(figwidth,nrow*figheight))
            axarr=axarr.reshape(axarr.size)

            plot_histogram_stats(fig,axarr[:3],pl,global_dic[sub_this]['pos'],quant='pos')
            plot_histogram_stats(fig,axarr[3:],pl,global_dic[sub_this]['vel'],quant='vel')
            
            plot_name=f'plots/{sim_name}_snap{snap}_compare_compression_sub{sub_this}.png'
            pl.savefig(plot_name)
            print(ltime(),'saved: ',plot_name)
            #pl.show()
            if(res_dic=={}):
                res_dic=global_dic[sub_this].copy()
                print(res_dic.keys())
            else:
                for qq,quant in enumerate(['pos','vel']):
                    res_dic[quant]['hist']['hcount'] += global_dic[sub_this][quant]['hist']['hcount']
                    res_dic[quant]['count'] += global_dic[sub_this][quant]['count']
                    res_dic[quant]['sum'] += global_dic[sub_this][quant]['sum']
                    res_dic[quant]['sum2'] += global_dic[sub_this][quant]['sum2']

        # Create the final figure summed over individual_histograms
        figwidth=15;figheight=5
        ncol=3;nrow=2
        fig,axarr=pl.subplots(nrow,ncol,sharex=False,sharey=False,
                      figsize=(figwidth,nrow*figheight))
        axarr=axarr.reshape(axarr.size)

        plot_histogram_stats(fig,axarr[:3],pl,res_dic['pos'],quant='pos')
        plot_histogram_stats(fig,axarr[3:],pl,res_dic['vel'],quant='vel')
            
        plot_name=f'plots/{sim_name}_snap{snap}_compare_compression_subAll.png'
        pl.savefig(plot_name)
        print(ltime(),'saved: ',plot_name)

    print('done')


main()
