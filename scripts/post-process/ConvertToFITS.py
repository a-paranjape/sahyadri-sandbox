# Author: Shadab Alam, Feb 2024
import os
import numpy as np
import fitsio as F
import time
import pandas as pd

def parse_header(header_line,file_type='trees'):
    '''This parse the header to figure out the column names'''

    #make sure the line starte with #
    assert(header_line[0]=='#')
    
    #To store the column map
    col_dic={}
    #remove hash and split
    tspl=header_line[1:].split()
    for tt,tmp in enumerate(tspl):
        if(file_type=='trees'):
            tmp_spl=tmp.split('(')[0]
            col_dic[tmp_spl]=tt   
        elif(file_type=='vahc'):
            tmp_spl=tmp
            col_dic[tmp]=tt
        
        #print(tt,tmp_spl,col_dic[tmp_spl])

    return col_dic

def get_header_dic(file_name):
    '''extract the line which has header and file_type and call the parse header'''
    
    head_row={'vahc':2,'trees':1}
    
    file_type=file_name.split('.')[-1]
    with open(file_name,'r') as fin:
        for ii in range(0,head_row[file_type]):
            tline=fin.readline()
            #print(ii,tline)

    col_dic=parse_header(tline,file_type=file_type)

    return col_dic, file_type


def out_column_list(col_dic,output_type='basic'):
    '''gives the list of columns to write given the file type'''

    basic_cols=['x','y','z','vx','vy','vz','Mvir','Mvir_all','M200b','M200c','M500c','M2500c',
                'rvir','rs','vrms','id','pid','T/|U|']

    if(output_type=='basic'):#only one of include or exclude should be filled
        out_column={'include':basic_cols}
    elif(output_type=='extended'):
        out_column={'exclude':basic_cols}
    else:
        out_column='All'

    col_write={}

    use_this=False
    for tt,tkey in enumerate(col_dic.keys()):
        if(out_column=='All'):
            use_this=True
        elif('exclude' in out_column.keys()):
            if(tkey in out_column['exclude']):
                use_this=False
            else:
                use_this=True
        elif('include' in out_column.keys()):
            if(tkey in out_column['include']):
                use_this=True
            else:
                use_this=False
        else:
            use_this=False

        #print(tt,tkey)
        if(use_this):
            #print(tt,tkey,use_this,out_column=='All','exclude' in out_column.keys(),tkey not in out_column['exclude'])
            col_write[tkey]=col_dic[tkey]

    return col_write




def Write2FITS(dicIN,datain=np.array([]),outfile='',colTokey={}):
    '''This takes the input data and write the relevant properties to FITS file'''
      
    #datatye to write to fits
    outtype=[]
    for kk,tkey in enumerate(colTokey.keys()):
        if(tkey in ['id','pid','upid']):
            outtype.append((tkey,'>i8'))
        else:
            outtype.append((tkey,'>f4'))
    #print(outtype)    
    
    rowsize=datain.shape[0]
    if('rowsize' not in dicIN):
        dicIN['rowsize']=int(datain.shape[0]*2)
        dicIN['dataout']=np.zeros(dicIN['rowsize'],dtype=outtype)

        
    if(rowsize>dicIN['rowsize']):
        dicIN['rowsize']=rowsize
        dicIN['dataout']=None
        dicIN['dataout']=np.zeros(rowsize,dtype=outtype)


    #transfer datain to data out
    for kk in colTokey.keys():
        dicIN['dataout'][kk][:rowsize]=datain[:,colTokey[kk]]

    #If fits file haven't been opened yet the open the fits file
    
    #dictionary to store file handles
    if('fhandle' not in dicIN.keys()):
        dicIN['fname']=outfile
        dicIN['fhandle']=F.FITS(outfile,'rw')
        dicIN['fhandle'].write(dicIN['dataout'][:rowsize])
    else:
        dicIN['fhandle'][-1].append(dicIN['dataout'][:rowsize])

    return dicIN


def convert_fits(indir='',rootin='',outdir=None,list_output_type=['basic','extended','vahc'],
                nlim=1000000):    
    '''converts the ascii halos to fits file'''

    if(outdir is None):
        outdir=indir

    #extension for different output_type
    ext_out={'basic':'trees','extended':'trees','vahc':'vahc'}
    
    for oo,out_type in enumerate(list_output_type):
        source_file='%s%s.%s'%(indir,rootin,ext_out[out_type])
        outfile='%s%s_%s.fits.gz'%(outdir,rootin,out_type)

        if(not os.path.isfile(source_file)):
            print('### File not found:%s'%(source_file))
            continue
        
        if(os.path.isfile(outfile)):
            print('### %s exists:%s'%(out_type,outfile))
            continue

        ts1=time.localtime()
        print(time.asctime(ts1),'\n\tworking on: ',source_file)
        print('\toutput file: ',outfile)


        #figure out the column mapping
        col_dic,file_type=get_header_dic(source_file)
        col_write=out_column_list(col_dic,output_type=out_type)
        
        Fdic_out={}

        #open the text file with pandas iterator
        tb=pd.read_csv(source_file,delim_whitespace=True,comment='#',header=None,
              dtype=float,chunksize=nlim)
        count=0
        for chunk in tb:
            if(count%5000==0):
                ts2=time.localtime()
                print(count,time.asctime(ts2))
            count=count+1
            
            Fdic_out=Write2FITS(Fdic_out,datain=chunk.values,outfile=outfile,colTokey=col_write)
            #if(count>10):
            #    break
    
        #break
        #cleanup by closing the fits file and writing checksum
        ts2=time.localtime()
        print('**** closed: ',Fdic_out['fname'],time.asctime(ts2))
        Fdic_out['fhandle'][-1].write_checksum()
        Fdic_out['fhandle'].close() 

    return 0


def fits_to_record(fhandle,index,list_sel_col=None):
    '''converts a fits file to record array kepping only the index
    You can keep only selected fields by giving a list as list_sel_col
    If the columns needed is spread over mutiple files then fhandle can be a list of fits handels'''

    if(type(fhandle)!=type([])):
        fhandle=[fhandle]


    #combined list of columns available in all fits file and organize the datatype
    list_col_dic={}
    dtype_dic={}
    for ii in range(0,len(fhandle)):
        list_col_dic[ii]=fhandle[ii][1][0].dtype.names
        dtype_dic[ii]=fhandle[ii][1][0].dtype

        if(ii==0):
            list_col=list_col_dic[ii]
        else:
            list_col=list_col+list_col_dic[ii]


    if(list_sel_col is None):
        list_sel_col=list_col

    dtype_rec=[]
    for ii in range(0,len(fhandle)):
        for jj in range(0,len(dtype_dic[ii])):
            if(list_col_dic[ii][jj] in list_sel_col):
                 dtype_rec.append((list_col_dic[ii][jj],dtype_dic[ii][jj]))

    #declare the record array to hold the variables
    rec_array=np.recarray((index.size),dtype=dtype_rec)
    col_missing=np.ones(len(list_sel_col),dtype=bool)
    for cc,cname in enumerate(list_sel_col):
        for ii in range(0,len(fhandle)):
            if(cname in list_col_dic[ii]):
                rec_array[cname]=fhandle[ii][1][cname][index]
                col_missing[cc]=False
                break

    if(np.sum(col_missing)>0):
        print('*** Warning *** \n Following field were requested but not found in the files')
        for ii in range(0,col_missing.size):
            if(col_missing[ii]):
                print(ii,list_sel_col[ii])

    return rec_array
