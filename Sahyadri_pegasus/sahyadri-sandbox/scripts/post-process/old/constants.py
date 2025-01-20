#!/mnt/csoft/tools/anaconda2/bin/python2

rhoc = 2.775366e11 # critical density in (Msun/h)/(Mpc/h)^3; value from PDG2012
TINY = 1e-15
NOTSOTINY = 1e-8
PI = 3.14159265359
full_sky = 4*PI*(180./PI)**2 # square degrees in the full sky.. ~= 41253
H0inv = 9.784619421 # 1/H0 in Gyr/h. 
speed_of_light = 2.99792458e5 # c in km/s
c_by_H0 = 0.01*speed_of_light # present Hubble radius in Mpc/h
Mpc_per_km = 3.2408e-20 # Mpc per km
yr_per_s  = 3.171e-8    # years per second


trees_dict = {'scm1024':True,'su1024/delta0.0':False,'scmL1024':False,'bdm_cdm1024':True,
              'su512/delta0.0':False,'bdm_zs1e5f0.51024':True,'wscm0.4keV1024':True}
nfile_dict = {'scm1024':8,'su1024/delta0.0':8,'scmL1024':8,'bdm_cdm1024':8,'su512/delta0.0':1,
              'bdm_zs1e5f0.51024':8,'wscm0.4keV1024':8}

yang_gal = [('galid','i'),('nyuid','i'),('ra','f'),('dec','f'),('z','f'),
            ('mr','f'),('mrlim','f'),('comp','f'),
            ('Mrpet','f'),('grpet','f'),('Mrmod','f'),('grmod','f'),('source','i')]
yang_grpgal = [('galid','i'),('nyuid','i'),('grpid','i'),('bcg','i'),('mmg','i')]
yang_grpmem = [('galid','i'),('nyuid','i'),('dr7photoid','i')]
yang_grpprop = [('grpid','i'),('ra','f'),('dec','f'),('z','f'),
                ('grplgL','f'),('grpmass','f'),('grplgMhLum','f'),('grplgMhMass','f'),
                ('dmeanLum','f'),('dmeanMass','f'),('fedge','f'),
                ('id1','i'),('id2','i')]
mock = [('galid','i'),('haloid','i'),('lgm','f'),
        ('x','f'),('y','f'),('z','f'),('vx','f'),('vy','f'),('vz','f'),
        ('gr','f'),('ur','f'),('Mr','f'),('lgmstar','f'),('lgmHI','f'),
        ('haloAlpha','f'),('haloAlpha2Mpch','f'),
        ('haloDelta','f'),('haloDelta2Mpch','f'),
        ('haloPsi1','f'),('haloPsi2','f'),('haloPsi3','f'),
        ('haloPsi12Mpch','f'),('haloPsi22Mpch','f'),('haloPsi32Mpch','f'),('b1','f'),
        ('Nsat','i'),('isred','i'),('iscen','i')]

halo_data_type = [('ID','i8'),('descID','i8'),('mbnd_vir','f'),('vmax','f'),('vrms','f'),
                  ('rvir','f'),('rs','f'),('np','i'),('x','f'),('y','f'),('z','f'),
                  ('vx','f'),('vy','f'),('vz','f'),('Jx','f'),('Jy','f'),('Jz','f'),
                  ('spin','f'),('rs_klypin','f'),
                  ('mvir','f'),('m200b','f'),('m200c','f'),('mCustom2','f'),('mCustom','f'),
                  ('Xoff','f'),('Voff','f'),('spin_bullock','f'),
                  ('b_to_a','f'),('c_to_a','f'),('Ax','f'),('Ay','f'),('Az','f'),
                  ('b_to_a_500c','f'),('c_to_a_500c','f'),('Ax_500c','f'),('Ay_500c','f'),('Az_500c','f'),
                  ('TbyU','f'),('Mpe_Behroozi','f'),('Mpe_Diemer','f'),('halfmassradius','f'),('pid','i')] 

halo_data_trees = [('Scale','f'),('ID','i8'),('descScale','f'),('descID','i8'),('numProg','i'),('pid','i'),
                   ('upid','i'),('descpid','i'),('phantom','i'),('sam_mvir','f'),('mbnd_vir','f'),
                   ('rvir','f'),('rs','f'),('vrms','f'),('mmp','i'),('ScaleLastMM','f'),('vmax','f'),
                   ('x','f'),('y','f'),('z','f'),('vx','f'),('vy','f'),('vz','f'),
                   ('Jx','f'),('Jy','f'),('Jz','f'),('spin','f'),
                   ('BreadthFirstID','i8'),('DepthFirstID','i8'),('TreeRootID','i8'),('OrigHaloID','i8'),
                   ('snapnum','i'),('NextCoprogDepthFirstID','i8'),('LastProgDepthFirstID','i8'),
                   ('rs_klypin','f'),('mvir','f'),('m200b','f'),('m200c','f'),('mCustom2','f'),('mCustom','f'),
                   ('Xoff','f'),('Voff','f'),('spin_bullock','f'),
                   ('b_to_a','f'),('c_to_a','f'),('Ax','f'),('Ay','f'),('Az','f'),
                   ('b_to_a_500c','f'),('c_to_a_500c','f'),('Ax_500c','f'),('Ay_500c','f'),('Az_500c','f'),
                   ('TbyU','f'),('Mpe_Behroozi','f'),('Mpe_Diemer','f'),('halfmassradius','f'),
                   ('macc','f'),('mpeak','f'),('vacc','f'),('vpeak','f'),('halfmassscale','f'),
                   ('accrate_inst','f'),('accrate_100Myr','f'),
                   ('accrate_1tdyn','f'),('accrate_2tdyn','f'),('accrate_mpeak','f'),
                   ('mpeakscale','f'),('accscale','f'),
                   ('firstaccscale','f'),('firstaccmvir','f'),('firstaccvmax','f'),('vmaxAtmpeak','f')] 

# scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) desc_pid(7) phantom(8) sam_mvir(9) mvir(10) 
# rvir(11) rs(12) vrms(13) mmp?(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21) vz(22) 
# Jx(23) Jy(24) Jz(25) Spin(26) Breadth_first_ID(27) Depth_first_ID(28) Tree_root_ID(29) Orig_halo_ID(30) 
# Snap_num(31) Next_coprogenitor_depthfirst_ID(32) Last_progenitor_depthfirst_ID(33) Rs_Klypin(34) 
# Mvir_all(35) M200b(36) M200c(37) M500c(38) M2500c(39) Xoff(40) Voff(41) Spin_Bullock(42) b_to_a(43) c_to_a(44) 
# A[x](45) A[y](46) A[z](47) b_to_a(500c)(48) c_to_a(500c)(49) A[x](500c)(50) A[y](500c)(51) A[z](500c)(52) 
# T/|U|(53) M_pe_Behroozi(54) M_pe_Diemer(55) Halfmass_Radius(56) 
# Macc(57) Mpeak(58) Vacc(59) Vpeak(60) Halfmass_Scale(61) Acc_Rate_Inst(62) Acc_Rate_100Myr(63) Acc_Rate_1*Tdyn(64) 
# Acc_Rate_2*Tdyn(65) Acc_Rate_Mpeak(66) Mpeak_Scale(67) Acc_Scale(68) First_Acc_Scale(69) First_Acc_Mvir(70) 
# First_Acc_Vmax(71) Vmax@Mpeak(72)

# expect lam1 <= lam2 <= lam3
halo_data_vac = [('ID','i8'),
                 ('lam1_R2R200b','f'),('lam2_R2R200b','f'),('lam3_R2R200b','f'),
                 ('lam1_R4R200b','f'),('lam2_R4R200b','f'),('lam3_R4R200b','f'),
                 ('lam1_R6R200b','f'),('lam2_R6R200b','f'),('lam3_R6R200b','f'),
                 ('lam1_R8R200b','f'),('lam2_R8R200b','f'),('lam3_R8R200b','f'),
                 ('lam1_R2Mpch','f'),('lam2_R2Mpch','f'),('lam3_R2Mpch','f'),
                 ('lam1_R3Mpch','f'),('lam2_R3Mpch','f'),('lam3_R3Mpch','f'),
                 ('lam1_R5Mpch','f'),('lam2_R5Mpch','f'),('lam3_R5Mpch','f'),
                 ('lamH1_R3Mpch','f'),('lamH2_R3Mpch','f'),('lamH3_R3Mpch','f'),
                 ('lamH1_R5Mpch','f'),('lamH2_R5Mpch','f'),('lamH3_R5Mpch','f'),
                 ('b1','f'),('b1wtd','f')]

scale_strings = ['R2R200b','R4R200b','R6R200b','R8R200b','R2Mpch','R3Mpch','R5Mpch']


# dictionaries for use with pandas
dict_yang_gal = {'galid':int,'nyuid':int,'ra':float,'dec':float,'z':float,
                 'mr':float,'mrlim':float,'comp':float,
                 'Mrpet':float,'grpet':float,'Mrmod':float,'grmod':float,'source':int}
dict_yang_grpgal = {'galid':int,'nyuid':int,'grpid':int,'bcg':int,'mmg':int}
dict_yang_grpmem = {'galid':int,'nyuid':int,'dr7photoid':int}
dict_yang_grpprop = {'grpid':int,'ra':float,'dec':float,'z':float,
                     'grplgL':float,'grpmass':float,'grplgMhLum':float,'grplgMhMass':float,
                     'dmeanLum':float,'dmeanMass':float,'fedge':float,
                     'id1':int,'id2':int}
dict_mock = {'galid':int,'haloid':int,'lgm':float,
             'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
             'gr':float,'ur':float,'Mr':float,'lgmstar':float,'lgmHI':float,
             'haloAlpha':float,'haloAlpha2Mpch':float,
             'haloDelta':float,'haloDelta2Mpch':float,
             'haloPsi1':float,'haloPsi2':float,'haloPsi3':float,
             'haloPsi12Mpch':float,'haloPsi22Mpch':float,'haloPsi32Mpch':float,'b1':float,
             'Nsat':int,'isred':int,'iscen':int}


dict_halo_data_type = {'ID':long,'descID':long,
                       'mbnd_vir':float,'vmax':float,'vrms':float,'rvir':float,'rs':float,'np':int,
                       'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
                       'Jx':float,'Jy':float,'Jz':float,'spin':float,'rs_klypin':float,
                       'mvir':float,'m200b':float,'m200c':float,'mCustom2':float,'mCustom':float,
                       'Xoff':float,'Voff':float,'spin_bullock':float,'b_to_a':float,'c_to_a':float,
                       'Ax':float,'Ay':float,'Az':float,'b_to_a_500c':float,'c_to_a_500c':float,
                       'Ax_500c':float,'Ay_500c':float,'Az_500c':float,
                       'TbyU':float,'Mpe_Behroozi':float,'Mpe_Diemer':float,'halfmassradius':float,
                       'pid':int} 

dict_halo_data_trees = {'Scale':float,'ID':long,'descScale':float,'descID':long,'numProg':int,
                        'pid':int,'upid':int,'descpid':int,'phantom':int,'sam_mvir':float,
                        'mbnd_vir':float,'rvir':float,'rs':float,'vrms':float,'mmp':int,
                        'ScaleLastMM':float,'vmax':float,
                        'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
                        'Jx':float,'Jy':float,'Jz':float,'spin':float,
                        'BreadthFirstID':long,'DepthFirstID':long,'TreeRootID':long,'OrigHaloID':long,
                        'snapnum':int,'NextCoprogDepthFirstID':long,'LastProgDepthFirstID':long,
                        'rs_klypin':float,
                        'mvir':float,'m200b':float,'m200c':float,'mCustom2':float,'mCustom':float,
                        'Xoff':float,'Voff':float,'spin_bullock':float,
                        'b_to_a':float,'c_to_a':float,'Ax':float,'Ay':float,'Az':float,
                        'b_to_a_500c':float,'c_to_a_500c':float,'Ax_500c':float,'Ay_500c':float,'Az_500c':float,
                        'TbyU':float,'Mpe_Behroozi':float,'Mpe_Diemer':float,'halfmassradius':float,
                        'macc':float,'mpeak':float,'vacc':float,'vpeak':float,'halfmassscale':float,
                        'accrate_inst':float,'accrate_100Myr':float,'accrate_1tdyn':float,
                        'accrate_2tdyn':float,'accrate_mpeak':float,'mpeakscale':float,
                        'accscale':float,'firstaccscale':float,'firstaccmvir':float,
                        'firstaccvmax':float,'vmaxAtmpeak':float} 

# expect lam1 <= lam2 <= lam3
dict_halo_data_vac = {'ID':long,
                      'lam1_R2R200b':float,'lam2_R2R200b':float,'lam3_R2R200b':float,
                      'lam1_R4R200b':float,'lam2_R4R200b':float,'lam3_R4R200b':float,
                      'lam1_R6R200b':float,'lam2_R6R200b':float,'lam3_R6R200b':float,
                      'lam1_R8R200b':float,'lam2_R8R200b':float,'lam3_R8R200b':float,
                      'lam1_R2Mpch':float,'lam2_R2Mpch':float,'lam3_R2Mpch':float,
                      'lam1_R3Mpch':float,'lam2_R3Mpch':float,'lam3_R3Mpch':float,
                      'lam1_R5Mpch':float,'lam2_R5Mpch':float,'lam3_R5Mpch':float,
                      'lamH1_R3Mpch':float,'lamH2_R3Mpch':float,'lamH3_R3Mpch':float,
                      'lamH1_R5Mpch':float,'lamH2_R5Mpch':float,'lamH3_R5Mpch':float,
                      'b1':float,'b1wtd':float}


# name lists for use with pandas
names_yang_gal = ['galid','nyuid','ra','dec','z',
                  'mr','mrlim','comp',
                  'Mrpet','grpet','Mrmod','grmod','source']
names_yang_grpgal = ['galid','nyuid','grpid','bcg','mmg']
names_yang_grpmem = ['galid','nyuid','dr7photoid']
names_yang_grpprop = ['grpid','ra','dec','z',
                      'grplgL','grpmass','grplgMhLum','grplgMhMass',
                      'dmeanLum','dmeanMass','fedge',
                      'id1','id2']
names_mock = ['galid','haloid','lgm','x','y','z','vx','vy','vz',
             'gr','ur','Mr','lgmstar','lgmHI',
              'haloAlpha','haloAlpha2Mpch',
              'haloDelta','haloDelta2Mpch',
              'haloPsi1','haloPsi2','haloPsi3',
              'haloPsi12Mpch','haloPsi22Mpch','haloPsi32Mpch','b1',
              'Nsat','isred','iscen']

names_halo_data_type = ['ID','descID',
                       'mbnd_vir','vmax','vrms','rvir','rs','np',
                       'x','y','z','vx','vy','vz',
                       'Jx','Jy','Jz','spin','rs_klypin',
                       'mvir','m200b','m200c','mCustom2','mCustom',
                       'Xoff','Voff','spin_bullock','b_to_a','c_to_a',
                       'Ax','Ay','Az','b_to_a_500c','c_to_a_500c',
                       'Ax_500c','Ay_500c','Az_500c',
                       'TbyU','Mpe_Behroozi','Mpe_Diemer','halfmassradius',
                       'pid'] 

names_halo_data_trees = ['Scale','ID','descScale','descID','numProg',
                        'pid','upid','descpid','phantom','sam_mvir',
                        'mbnd_vir','rvir','rs','vrms','mmp',
                        'ScaleLastMM','vmax',
                        'x','y','z','vx','vy','vz',
                        'Jx','Jy','Jz','spin',
                        'BreadthFirstID','DepthFirstID','TreeRootID','OrigHaloID',
                        'snapnum','NextCoprogDepthFirstID','LastProgDepthFirstID',
                        'rs_klypin',
                        'mvir','m200b','m200c','mCustom2','mCustom',
                        'Xoff','Voff','spin_bullock',
                        'b_to_a','c_to_a','Ax','Ay','Az',
                        'b_to_a_500c','c_to_a_500c','Ax_500c','Ay_500c','Az_500c',
                        'TbyU','Mpe_Behroozi','Mpe_Diemer','halfmassradius',
                        'macc','mpeak','vacc','vpeak','halfmassscale',
                        'accrate_inst','accrate_100Myr','accrate_1tdyn',
                        'accrate_2tdyn','accrate_mpeak','mpeakscale',
                        'accscale','firstaccscale','firstaccmvir',
                        'firstaccvmax','vmaxAtmpeak'] 

names_halo_data_vac = ['ID',
                      'lam1_R2R200b','lam2_R2R200b','lam3_R2R200b',
                      'lam1_R4R200b','lam2_R4R200b','lam3_R4R200b',
                      'lam1_R6R200b','lam2_R6R200b','lam3_R6R200b',
                      'lam1_R8R200b','lam2_R8R200b','lam3_R8R200b',
                      'lam1_R2Mpch','lam2_R2Mpch','lam3_R2Mpch',
                      'lam1_R3Mpch','lam2_R3Mpch','lam3_R3Mpch',
                      'lam1_R5Mpch','lam2_R5Mpch','lam3_R5Mpch',
                      'lamH1_R3Mpch','lamH2_R3Mpch','lamH3_R3Mpch',
                      'lamH1_R5Mpch','lamH2_R5Mpch','lamH3_R5Mpch',
                      'b1','b1wtd']
