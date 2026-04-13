from __future__ import print_function
print( 'IMPORTING PACKAGES...')
from numpy import *
from math import *
from pyraf import iraf
from pyraf.iraf import noao, digiphot, daophot
import astropy.io.fits as pyfits
from astropy.io import ascii
import os, shutil,sys
import random

#iraf.noao(_doprint=0)
#iraf.obsutil(_doprint=0)
#iraf.daophot(_doprint=0)

print( 'STARTING...')
pydir = os.getcwd()
print( " ")

#####################################################################################
################### READ THE INPUT FILE AND DEFINE USEFUL STUFF #####################
#####################################################################################

infile1 = 'legus_clusters_extraction.input'
# verify that file exists
if os.path.exists(pydir+'/'+infile1) == False:
    print( "File ",infile1," could not be found in ",pydir)
    sys.exit("quitting now...")
text1 = loadtxt(infile1, unpack=True, skiprows=0, comments="#", dtype='str')
# read the file
# ---- --- -- - distance:
galdist=text1[1].strip()
galdist=float(galdist)*1e6
# ---- --- -- - ci value chosen for the cut:
ci=text1[7].strip()
# ---- --- -- - aperture for photometry:
useraperture=text1[5].strip()

infile2 ='legus_clusters_comptest.input'
# verify that file exists
if os.path.exists(pydir+'/'+infile2) == False:
    print( "File ",infile2," could not be found in ",pydir)
    sys.exit("quitting now...")
label2,text2 = loadtxt(infile2, unpack=True, skiprows=0, comments="#", dtype='str')
# read the file

# ---- --- -- - Generate the process flags (whether or not to proceed with a step)- -- --- ----
#flagstep1: do the source creation?:
if 'DO_CREATION' in label2:
  flagstep1=text2[where(label2=='DO_CREATION')]
  flagstep1=str(flagstep1[0])
  if flagstep1 not in ('yes', 'no', 'YES', 'NO'):
    sys.exit("Wrong input in line labeled DO_CREATION")
else:
  flagstep1='yes'

#flagstep3: do the source extraction?:
if 'DO_EXTRACTION' in label2:
  flagstep2=text2[where(label2=='DO_EXTRACTION')]
  flagstep2=str(flagstep2[0])
  if flagstep2 not in ('yes', 'no', 'YES', 'NO'):
    sys.exit("Wrong input in line labeled DO_EXTRACTION")
else:
  flagstep3='yes'

#flagstep3: do the source extraction?:
if 'DO_PHOTOMETRY' in label2:
  flagstep3=text2[where(label2=='DO_PHOTOMETRY')]
  flagstep3=str(flagstep3[0])
  if flagstep3 not in ('yes', 'no', 'YES', 'NO'):
    sys.exit("Wrong input in line labeled DO_PHOTOMETRY")
else:
    flagstep3b='yes'


#flagstep4: do the source recovery and comparison?:
if 'DO_RECOVERY' in label2:
  flagstep4=text2[where(label2=='DO_RECOVERY')]
  flagstep4=str(flagstep4[0])
  if flagstep4 not in ('yes', 'no', 'YES', 'NO'):
    sys.exit("Wrong input in line labeled DO_RECOVERY")
else:
  flagstep4='yes'

#flagstep5: do the source recovery and comparison?:
if 'DO_PLOTTING' in label2:
  flagstep5=text2[where(label2=='DO_PLOTTING')]
  flagstep5=str(flagstep5[0])
  if flagstep5 not in ('yes', 'no', 'YES', 'NO'):
    sys.exit("Wrong input in line labeled DO_PLOTTING")
else:
  flagstep5='yes'

# ---- --- -- - scientific frame name: - -- --- ----
sciframe = text2[where(label2=='FRAME_NAME')]
sciframe=str(sciframe[0])
sciframepath = pydir+'/'+sciframe

# ---- --- -- - filter name: - -- --- ----
fname=text2[where(label2=='FILTER')]
fname=fname[0].split(',')
fname=list(map(str,fname))
#fname=fromiter(fname, dtype=str)

# ---- --- -- - psf model frame name: - -- --- ----
psffile=text2[where(label2=='PSF_FILE')]
psffile=str(psffile[0])
psffilepath=pydir+'/'+psffile

# ---- --- -- - number of sources per sim frame: - -- --- ----
nums_perframe=text2[where(label2=='CLUSTERS_PER_FRAME')]
nums_perframe=int(nums_perframe[0])

# ---- --- -- - number of sources per Reff value: - -- --- ----
nums_perradius=text2[where(label2=='FRAME_NUMBER')]
nums_perradius=int(nums_perradius[0])

# ---- --- -- - Effective radii for which I want to simulate sources: - -- --- ----
reffs=text2[where(label2=='R_EFF')]
reffs=reffs[0].split(',')
reffs=map(float,reffs)
reffs=fromiter(reffs, dtype=float)
a=argsort(reffs)
reffs=reffs[a]

# convert from Reff to baolab FWHM
pixscale=0.04       # 1 px = 0.04''
baoFHWM=zeros(size(reffs))
for i in range (0,len(baoFHWM)): baoFHWM[i]=(reffs[i]/galdist)*(180./pi)*(3600./pixscale)/1.13

# ---- --- -- - Magnitude range for the simulated sources - -- --- ----
# maglim=text2[where(label2=='MAG_RANGE')]
# maglim=maglim[0].split(',')
# if len(maglim) != 2 : sys.exit('The magnitude range assignment is incorrect, check it in the input file! \nquitting now...')
# maglim=map(float,maglim)
# maglim=asarray(maglim)
# a=argsort(maglim)
# maglim=maglim[a]

maglim = text2[where(label2=='MAG_RANGE')][0].split(',')
if len(maglim) != 2:
    sys.exit('The magnitude range assignment is incorrect, check it in the input file! \nquitting now...')
maglim = asarray(list(map(float, maglim)), dtype=float)
a = argsort(maglim)
maglim = maglim[a]

# ---- --- -- - parameters for finding the region where to simulate the sources - -- --- ----
par=text2[where(label2=='REGION')]
par=par[0].split(',')
par=list(map(float,par))
if len(par) != 5:
    if (par[0] != 0 or len(par) != 1):
        sys.exit("I need 5 numbers to define a region (or number 0 for not defining it)\nCheck the line labeled REGION!")
else:
    xcenter=par[0]
    ycenter=par[1]
    xsize=par[2]
    ysize=par[3]
    deg=par[4]

# ---- --- -- - aperture correction file: - -- --- ----
apcorrfile=text2[where(label2=='APCORR')]
apcorrfile=str(apcorrfile[0])

# ---- --- -- - aperture correction file: - -- --- ----
merr_cut=text2[where(label2=='MERR_CUT')]
merr_cut=float(merr_cut[0])

# ---- --- -- - BAOlab zeropoint value: - -- --- ----
baozpoint=1e15
# ---- --- -- - BAOlab path:
baopath=text2[where(label2=='BAO_PATH')]
baopath=str(baopath[0])

# ---- --- -- - scientific frame dimensions: - -- --- ----
hd = pyfits.getheader(sciframe)
xaxis = hd['NAXIS1']
yaxis = hd['NAXIS2']
# coordinates for test source:
xcss=int(xaxis/2)
ycss=int(yaxis/2)
# ---- --- -- - parameters for finding the region where to simulate the sources - -- --- ----
if (len(par) == 1 and par[0]==0):
    xcenter=xcss
    ycenter=ycss
    xsize=xaxis
    ysize=yaxis
    deg=0.

# ---- --- -- - finding the zeropoint and exptime - -- --- ----
zpfile=text2[where(label2=='ZPOINT')]
zpfile=str(zpfile[0])
instrument,filter,zpoint = loadtxt(zpfile, unpack=True, skiprows=0,  usecols=(0,1,2), dtype='str')
match = (instrument == fname[0]) & (filter == fname[1])
zp=zpoint[match]
if size(zp)==0: sys.exit("Wrong instrument/filter names! Check the input file! \nQuitting...")
zp = float(zp[0])
expt = hd['EXPTIME']

print ('zp: ')
print (zp)
print ('expt: ')
print (expt)

#----------------------------------------------------------------------------------------------------------------------------------
#STEP 1: SOURCE CREATION
#----------------------------------------------------------------------------------------------------------------------------------
if (flagstep1=='yes' or flagstep1=='YES'):
  
# ---- --- -- - create directories - -- --- ----
  if os.path.exists(pydir+'/baolab') == False:
    os.makedirs(pydir+'/baolab')
  else :
    os.system('rm '+pydir+'/baolab/* ')
  if os.path.exists(pydir+'/synthetic_frames') == False:
    os.makedirs(pydir+'/synthetic_frames')
  else:
    os.system('rm '+pydir+'/synthetic_frames/* ')

  #########################################################################################
  ######### I  GENERATE 2 TEST SOURCES FOR A VISUAL CHECK OF THE SIMULATED SOURCES ########
  #########################################################################################
  path = pydir + '/baolab'
  os.chdir(path)
  factor_zp=10**(-0.4*zp)
  factor_def=baozpoint*factor_zp

  f=open('mk_cmppsf.bl','w')
  f.write('# I GENERATE THE COMPOSITE PSFs \n\n')
  for z in range(0,len(baoFHWM)):
   f.write('mkcmppsf cmppsf_'+str(reffs[z])+'pc.fits MKCMPPSF.PSFTYPE=USER MKCMPPSF.OBJTYPE=EFF15 MKCMPPSF.FWHMOBJX='+str(baoFHWM[z])+' MKCMPPSF.FWHMOBJY='+str(baoFHWM[z])+' MKCMPPSF.RADIUS=100. MKCMPPSF.BITPIX=-32 MKCMPPSF.PSFFILE='+psffilepath+' \n\n')
  f.write('quit \n')
  f.close()

  g1=open('testcoord1.txt','w')
  g1.write('50 50 '+str(maglim[0]))
  g1.close()
  g2=open('testcoord2.txt','w')
  g2.write('50 50 '+str(maglim[1]))
  g2.close()

  testimg1='test_source_mag'+str(maglim[0])+'_reff'+str(reffs[0])+'.fits'
  testimg2='test_source_mag'+str(maglim[1])+'_reff'+str(reffs[-1])+'.fits'
  f=open('mk_test.bl','w')
  f.write('# I GENERATE THE TEST SOURCE IN ORDER TO GET THE REAL ZEROPOINT I NEED \n\n')
  f.write('mksynth testcoord1.txt '+testimg1+' MKSYNTH.REZX=100 MKSYNTH.REZY=100 MKSYNTH.RANDOM=NO MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT='+str(baozpoint)+' MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE=cmppsf_'+str(reffs[0])+'pc.fits \n \n')
  f.write('mksynth testcoord2.txt '+testimg2+' MKSYNTH.REZX=100 MKSYNTH.REZY=100 MKSYNTH.RANDOM=NO MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT='+str(baozpoint)+' MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE=cmppsf_'+str(reffs[-1])+'pc.fits \n \n')
  f.write('quit \n')
  f.close()

  print( 'Running baolab to create the composite PSFs...')
  os.system(baopath+'bl < mk_cmppsf.bl > bao.txt')
  print( 'Running baolab to create the test frames...')
  os.system(baopath+'bl < mk_test.bl > bao.txt')


  iraf.imarith(operand1 = testimg1, op = '/', operand2 = factor_def, result = testimg1)
  iraf.imarith(operand1 = testimg2, op = '/', operand2 = factor_def, result = testimg2)
  print( '')

  #########################################################################################
  ######### I  GENERATE THE COMPOSITE THE SYNTHETIC SOURCES FOR COMPLETENESS TEST #########
  #########################################################################################
  theta=deg*pi/180.
  border=100.
  print( 'I am writing the baolab file to generate the frames with synthetic sources...')
  for z in range(0,len(baoFHWM)):
   f=open('mk_frames_'+str(reffs[z])+'pc.bl','w')
   f.write('# I GENERATE THE TEST SOURCE IN ORDER TO GET THE REAL ZEROPOINT I NEED \n \n')
   for i in range(int(nums_perradius)):
    namesource=''+str(reffs[z])+'pc_sources_'+str(i)+'_temp.fits'
    namecoord=''+str(reffs[z])+'pc_sources_'+str(i)+'.txt'
    file=open(namecoord,'w')
    for j in range(int(nums_perframe)):
     xr=random.uniform(-xsize/2.+border,xsize/2.-border)
     yr=random.uniform(-ysize/2.+border,ysize/2.-border)
     x_rot=xcenter+xr*cos(theta)-yr*sin(theta)
     y_rot=ycenter+xr*sin(theta)+yr*cos(theta)
     file.write(str(x_rot)+' '+str(y_rot)+' '+str(random.uniform(maglim[0],maglim[1]))+'\n')
    file.close()
    f.write('mksynth '+namecoord+' '+namesource+' MKSYNTH.REZX='+str(xaxis)+' MKSYNTH.REZY='+str(yaxis)+' MKSYNTH.RANDOM=NO MKSYNTH.NSTARS=100 MKSYNTH.MAGFAINT=0. MKSYNTH.MAGBRIGHT=0. MKSYNTH.MAGSTEP=0 MKSYNTH.RDNOISE=0 MKSYNTH.BACKGR=0 MKSYNTH.ZPOINT='+str(baozpoint)+' MKSYNTH.EPADU=1.0 MKSYNTH.BIAS=0 MKSYNTH.PHOTNOISE=YES MKSYNTH.PSFTYPE=USER MKSYNTH.PSFFILE=cmppsf_'+str(reffs[z])+'pc.fits \n \n')
   f.write('quit \n')
   f.close()

  print( 'I am running baolab to generate the frames with synthetic sources... \nthis could take a while... BE PATIENT!')
  for z in range(0,len(baoFHWM)):
   print( str(reffs[z])+' pc sources...')
   os.system(baopath+"bl < mk_frames_"+str(reffs[z])+"pc.bl > bao.txt")


  ##########################################################################################
  ########### I  DIVIDE THE FRAMES FOR THE CORRECT FACTOR AND ADD THE BACKGROUND ###########
  ##########################################################################################
  print( '')
  print( 'I am adding the newly generated synthetic frames to the background...')
  print( 'The resulting frames of this operation will be the ones to use for completeness test!')
  print( 'But this operation can take several minutes! BE PATIENT (again!!!)...')

  os.system('ls *pc_sources_*_temp.fits > list_temp.txt')
  listimg=genfromtxt('list_temp.txt',dtype='str')
  if size(listimg)==1:
   listimg=append(listimg,'string')
   listimg=listimg[where(listimg[1]=='string')]
  for i in range(0,len(listimg)):
   print( 'working on image '+str(i+1)+' out of '+str(len(listimg)))
   name_final=listimg[i][0:-10]+'.fits'
   temp=listimg[i][0:-10]+'_temp2.fits'
   iraf.imarith(operand1=listimg[i],   op ='/',   operand2=factor_def,   result=temp)
   iraf.imarith(operand1=temp,   op='+',   operand2=sciframepath,   result=name_final)

   os.system('rm '+temp+' '+listimg[i])
  os.system('rm list_temp.txt')

  ######################################################################################################
  ################ MAKE SOME ORDER BEFORE STARTING WITH EXTRACTION AND PHOTOMETRY !!!!! ################
  ######################################################################################################

  # move synthetic frames
  os.system('mv *pc_sources_* '+pydir+'/synthetic_frames/')
  # remove useless stuff
  os.system('rm testcoord?.txt')
# listing the fits file I want to analyze
  path = pydir+'/synthetic_frames/'
  os.chdir(path)
  os.system('ls *.fits > list_frames.ls')
  os.system('ls *.txt > list_coords.ls')
  os.chdir(pydir)
  print('\n Source creation is completed! \n')
else:
  print('\n Source creation skipped!\n')


#----------------------------------------------------------------------------------------------------------------------------------
#STEP 2: SOURCE EXTRACTION
#----------------------------------------------------------------------------------------------------------------------------------
if flagstep2=='yes' or flagstep2=='YES':

  print( 'Preparing for s_extraction ...')
  # check and move the config files for SExtractor
  if os.path.exists(pydir+'/R2_wl_aa.config') == False: sys.exit("cannot find the file 'R2_wl_aa.config' in the main dir \nquitting now...")
  if os.path.exists(pydir+'/output.param') == False: sys.exit("cannot find the file 'output.param' in the main dir \nquitting now...")
  if os.path.exists(pydir+'/default.nnw') == False: sys.exit("cannot find the file 'default.nnw' in the main dir \nquitting now...")
  
  if os.path.exists(pydir+'/s_extraction') == False:
      os.makedirs(pydir+'/s_extraction')
  
  os.system('scp '+pydir+'/R2_wl_aa.config '+pydir+'/s_extraction/.')
  os.system('scp '+pydir+'/output.param '+pydir+'/s_extraction/.')
  os.system('scp '+pydir+'/default.nnw '+pydir+'/s_extraction/.')

  #######################################=- SEXTRACTOR -=#################################
  # move to s_extraction directory
  path = pydir + '/s_extraction'
  os.chdir(path)

  source =  pydir + '/synthetic_frames/list_frames.ls'
  destination = path+'/list_frames.ls'
  shutil.copyfile(source, destination)

  framename=loadtxt('list_frames.ls',dtype='str')

  if size(framename)==1:
    framename=append(framename,'string')
    framename=framename[where(framename[1]=='string')]

  for z in range(0,len(framename)):
   framepath=pydir+'/synthetic_frames/'+framename[z]
   print( 'Executing SExtractor on image : ', framename[z])
   # verify that file exists
   if os.path.exists(framepath) == False:
    print( "File "+framename[z]+" could not be found in "+pydir+'/synthetic_frames/')
    sys.exit("quitting now...")
   #####   here we run sextractor
   command = 'sex '+framepath+'  -c R2_wl_aa.config'
   #print( command)
   os.system(command)
   #####   write the file 'catalog_ds9_sextractor.reg'
   #Original command:
   #xx, yy ,fwhmdeg,source_class,mag_best= loadtxt('R2_wl_dpop_detarea.cat', unpack=True, skiprows=5,  dtype='str')
   #mag_best, xx, yy,fwhmdeg, source_class = loadtxt('sextractorUband9_mod.txt', unpack=True, skiprows=2,  dtype='str')
   xx, yy ,fwhmdeg,source_class,mag_best = loadtxt('sextractorVband.txt', unpack=True, skiprows=5,  dtype='str')
   outputfile = 'cat_ds9_sextr_'+framename[z]+'.reg'
   file = open(outputfile, "w")
   file.write('global color=blue width=5 font="helvetica 15 normal roman" highlite=1 \n')
   file.write('image\n')
   for i in range(len(xx)):
    c2 =  str(xx[i])
    c3 =  str(yy[i])
    newline = 'circle('+ c2 + ',' + c3 +  ',7) \n'
    file.write(newline)
   file.close()

  # --- -- - - write the sex_*.coo file - after sextraction, before photometry
   outputfile = 'sex_'+framename[z]+'.coo'
   file = open(outputfile, "w")
   for i in range(len(xx)):
    c2 =  str(xx[i])
    c3 =  str(yy[i])
    newline = c2+' '+c3+'\n'
    file.write(newline)
   file.close()

  print('\n Source extraction is completed! \n')
else:
  print('\n Source extraction skipped!\n')

#----------------------------------------------------------------------------------------------------------------------------------
#STEP 3: PHOTOMETRY
#----------------------------------------------------------------------------------------------------------------------------------
if flagstep3=='yes' or flagstep3=='YES':

  # creating photometry and CI folders
  if os.path.exists(pydir+'/photometry') == False: os.makedirs(pydir+'/photometry')
  if os.path.exists(pydir+'/CI') == False: os.makedirs(pydir+'/CI')

  framename=loadtxt(pydir+'/synthetic_frames/list_frames.ls',dtype='str')
  if size(framename)==1:
    framename=append(framename,'string')
    framename=framename[where(framename[1]=='string')]

  coordname=loadtxt(pydir+'/synthetic_frames/list_coords.ls',dtype='str')
  if size(coordname)==1:
    coordname=append(coordname,'string')
    coordname=coordname[where(coordname[1]=='string')]

  #######################################=- PHOTOMETRY -=#################################
  path = pydir+'/photometry'
  os.chdir(path)

  for z in range(0,size(framename)):
   framepath=pydir+'/synthetic_frames/'+framename[z]
   if os.path.exists(pydir+'/s_extraction/sex_'+framename[z]+'.coo') == False:
     print( "file "+'/s_extraction/sex_'+framename[z]+'.coo'+" not found: photometry will be run on SYNTHETIC SOURCES ONLY!")
     coords=pydir+'/synthetic_frames/'+coordname[z]
   else: coords=pydir+'/s_extraction/sex_'+framename[z]+'.coo'
   
   iraf.unlearn("datapars")
   iraf.datapars.scale=1.0
   iraf.datapars.fwhmpsf = 2.0
   iraf.datapars.sigma     = 0.01
   iraf.datapars.readnoise = 5.0
   iraf.datapars.epadu     = expt    ##  exptime
   iraf.datapars.itime      = 1.
   iraf.unlearn("centerpars")
   iraf.centerpars.calgorithm = 'centroid' ###centering will be done here for the whole dataset
   iraf.centerpars.cbox=1
   iraf.centerpars.cmaxiter=3
   iraf.centerpars.maxshift=1
   iraf.unlearn("fitskypars")
   iraf.fitskypars.salgori  = 'mode'
   iraf.fitskypars.annulus  = 7.
   iraf.fitskypars.dannulu  = 1.
   iraf.unlearn("photpars")
   if (float(useraperture) == 3. or float(useraperture) == 1.): apertures = "1.0,3.0"
   else: apertures = "1.0,3.0,"+useraperture
   iraf.photpars.apertures = apertures
   iraf.photpars.zmag = zp
   iraf.unlearn("phot")
   iraf.phot.image =  framepath
   iraf.phot.coords = coords
   iraf.phot.output = pydir + '/photometry/mag_'+framename[z]+'.mag'
   iraf.phot.interactive = "no"
   iraf.phot.verbose = "no"
   iraf.phot.verify = "no"
   print( 'starting photometric analysis on frame '+framename[z]+'.. \t using apertures: '+apertures)
   iraf.phot(framepath)
   # txdump detarea_f555w_3px.mag XCENTER,YCENTER,MAG,MERR,MSKY,ID mode=h > detarea_f555w_3px_short.mag
   cmd = 'grep "*" mag_'+framename[z]+'.mag > detarea_f555w_short.mag'
   os.system(cmd)
   #cmd = "sed 's/INDEF/99.999/g' -i detarea_f555w_short.mag"
   #os.system(cmd)
   cmd = "/usr/bin/sed 's/INDEF/99.999/g' detarea_f555w_short.mag > kk.tmp"
   os.system(cmd)
   source =   'kk.tmp'
   destination = 'mag_'+framename[z]+'.txt'
   shutil.copyfile(source, destination)

  #######################################=- CONCENTRATION INDEX CUT -=#################################
  path = pydir+'/photometry'
  os.chdir(path)

  if os.path.exists(pydir+'/'+apcorrfile) == False:
    print( "I cannot find the 'aperture correction' file")
    sys.exit("quitting now...")
  filter,ap,aperr=loadtxt(pydir+'/'+apcorrfile,unpack=True,usecols=(0,1,2),dtype='str')
  a=where(filter == fname[1])
  apcorr=ap[a]
  apcorrerr=aperr[a]
  apcorr=float(apcorr[0])
  apcorrerr=float(apcorrerr[0])
  #path = pydir + '/photometry'
  #os.chdir(path)
  for z in range(0,len(framename)):
   # here we use CI_cut_tab.pro
   aper,a,a,a, mag,merr= loadtxt('mag_'+framename[z]+'.txt', unpack=True, skiprows=0,  usecols=(0,1,2,3,4,5), dtype='str')
   # select stars
   id = where(aper == '1.00')
   mag_1 = mag[id]
   id = where(aper == '3.00')
   mag_3 = mag[id]
   
   useraperture=float(useraperture)
   useraperture="{0:.2f}".format(useraperture)
   if (float(useraperture)== 3.): id = where(aper == '3.00')
   elif (float(useraperture)== 1.): id = where(aper == '1.00')
   else: id = where(aper == str(useraperture))
   mag_4 = mag[id]
   merr_4 = merr[id]
   mag_1 = map(float,mag_1)
   mag_3 = map(float,mag_3)
   mag_4 = map(float,mag_4)
   merr_4 = map(float,merr_4)
   mag_3 = asarray(list(mag_3))
   mag_1 = asarray(list(mag_1))
   mag_4 = asarray(list(mag_4))
   merr_4 = asarray(list(merr_4))
   values = subtract(mag_1,mag_3)
   #print( values)
   iraf.unlearn("txdump")
   iraf.txdump.textfiles = 'mag_'+framename[z]+'.mag'
   iraf.txdump.fields = "XCENTER,YCENTER"
   iraf.txdump.expr = "yes"
   iraf.txdump(Stdout='temp'+framename[z]+'.mag',textfiles = 'mag_'+framename[z]+'.mag',fields = "XCENTER,YCENTER",expr = "yes")
   #xc, yc = loadtxt(pydir + '/s_extraction/R2_wl_dpop_detarea.cat', unpack=True, skiprows=5,  usecols=(0,1), dtype='str')
   #os.system('cp temp.mag coo_'+framename[z]+'.coo')
   xc, yc = loadtxt('temp'+framename[z]+'.mag', unpack=True, skiprows=0,  usecols=(0,1), dtype='str')

  # --- -- - - write the coo_*.coo file - after photometry, before ci cut
   f = open('coo_'+framename[z]+'.coo', 'w')
   for k in range(len(xc)):
    mag_4[k]=mag_4[k]+apcorr
    merr_4[k]=sqrt(merr_4[k]**2+apcorrerr**2)
    if merr_cut==0:
      f.write(xc[k]+'  '+yc[k]+' '+str(mag_4[k])+' '+str(merr_4[k])+' '+str(values[k])+'\n')
    else:
      if merr_4[k] <= merr_cut: f.write(xc[k]+'  '+yc[k]+' '+str(mag_4[k])+' '+str(merr_4[k])+' '+str(values[k])+'\n')
   f.close()

  # --- -- - - write the ci_cut_*.coo file - after photometry, after ci cut
   ci= float(ci)
   id = where((values >= ci))
   xc= xc[id]
   yc= yc[id]
   mag_4=mag_4[id]
   merr_4=merr_4[id]
   ci_values=values[id]
   f = open('ci_cut_'+framename[z]+'.coo', 'w')
   mag_4 = array(list(map(float,mag_4)))
   merr_4 = array(list(map(float,merr_4)))
   for k in range(len(xc)):
     if merr_cut==0:
       f.write(xc[k]+'  '+yc[k]+' '+str(mag_4[k])+' '+str(merr_4[k])+' '+str(ci_values[k])+'\n')
     else:
       if merr_4[k] <= merr_cut: f.write(xc[k]+'  '+yc[k]+' '+str(mag_4[k])+' '+str(merr_4[k])+' '+str(ci_values[k])+'\n')
   f.close()
   os.system('mv ci_cut_'+framename[z]+'.coo '+pydir+'/CI/ci_cut_'+framename[z]+'.coo')
   if os.path.exists('ci_cut_'+framename[z]+'.coo') == True:
    os.system('rm ci_cut_'+framename[z]+'.coo')

  print('\n Photometry has been completed! \n')
else:
  print('\n Photometry skipped! \n')
 

#----------------------------------------------------------------------------------------------------------------------------------
#STEP 4: RECOVERY/ COMPARISON
#----------------------------------------------------------------------------------------------------------------------------------
import itertools

#----------------------------------------------------------------------------------------------
#STARTUP - CHECK INPUTS AND INITIATE FOLDERS
#----------------------------------------------------------------------------------------------
os.chdir(pydir)

infile='legus_clusters_comptest.input'
label,inputs = genfromtxt(infile, unpack=True, skip_header=0, comments="#", dtype='str')

framenumber=nums_perradius

minmag=maglim[0]
maxmag=maglim[1]

binsize=float(fromiter(inputs[where(label=='BINSIZE')],dtype=float))
tolerance=float(fromiter(inputs[where(label=='TOLERANCE')],dtype=float))

xcol=0            #specifies which column in the data contains the x-coordinates 
ycol=1            #specifies which column in the data contains the x-coordinates 


if flagstep4=='yes' or flagstep4=='YES':
  #---CREATE DATA FOLDER---
  if os.path.exists(pydir+'/recovery') == False:
    os.makedirs(pydir+'/recovery')

  #----------------------------------------------------------------------------------------------
  #SECTION 1: RECOVER DATA
  #----------------------------------------------------------------------------------------------


  #loop over all source sizes
  for pc in reffs:
    if type(pc)=='int':
      pc=round(pc+0.1-0.1,1)


    #report state to user
    print('---------------------------------------------')
    print('Starting recovery on {}pc sources: \n'.format(pc))
    
    #loop over both states of Concentration index cutting. 0=no CI cut, 1=CI cut
    for CIcut in [0,1]:
      #initiate temporary storage vectors for summation (Section 2)
      tot_sim_mag = []
      rec_out_mag=[]
      rec_out_mag_err=[]
      rec_sim_mag=[]
      tot_per_cent=[]
      ci_values=[]

      #Loop over each frame 
      for frame in arange(framenumber):
        #report progress to user
        if CIcut==0:
          sys.stdout.write('\r')
          comp_frac=round(((frame+1)/float(framenumber))*100.,1)
          sys.stdout.write('Recovering from data before CI cut: {}% complete'.format(comp_frac))
          sys.stdout.flush()
        else:
          sys.stdout.write('\r')
          comp_frac=round(((frame+1)/float(framenumber))*100.,1)
          sys.stdout.write('Recovering from data after CI cut: {}% complete'.format(comp_frac))
          sys.stdout.flush()



        #Select what catalogs to use:
        simcatalog = 'synthetic_frames/{}pc_sources_{}.txt'.format(pc,frame)
        if CIcut==1:
            reccatalog = 'CI/ci_cut_{}pc_sources_{}.fits.coo'.format(pc,frame)
        elif CIcut==0:
            reccatalog = 'photometry/coo_{}pc_sources_{}.fits.coo'.format(pc,frame)

        #Load data
        simdata=loadtxt(simcatalog, comments='#')        #load the 'fake-source'catalog
        recdata=loadtxt(reccatalog,comments='#')         #load the recovered catalog


        #Order the data
        simX=simdata[:,xcol]                    #Take out x and y coordinates for each dataset    
        simY=simdata[:,ycol]

        recX=recdata[:,xcol]
        recY=recdata[:,ycol]

        #initialize vector of zeros for storing results
        checksum=zeros(len(simX))                  
        sim_magnitudes=[]
        rec_magnitudes=[]
        rec_magnitudes_err=[]
        ci=[]

        #Match the coordinates and store corresponding simulated and recovered magnitudes
        for a in list(arange(len(simX))):
          b=0
          while b < len(recX):
            if sqrt((recX[b]-simX[a])**2+(recY[b]-simY[a])**2)<=tolerance:
              checksum[a]=1                   
              sim_magnitudes.append(simdata[a,2])
              #check if recovered magnitudes exist in file, if so save them.
              if len(recdata[1,:])>3:           
                rec_magnitudes.append(recdata[b,2])
                rec_magnitudes_err.append(recdata[b,3])
                ci.append(recdata[b,4])
              b=len(recX)+1
            else:
              b=b+1

        #Format results
        percent_recovered=100*round(sum(checksum)/len(simX),2)    #sum the checksumvector and divide by total amount of seeded sources
        tot_per_cent.append(percent_recovered)

        rec_out_mag.append(rec_magnitudes)
        rec_out_mag_err.append(rec_magnitudes_err)
        rec_sim_mag.append(sim_magnitudes)
        ci_values.append(ci)


        data=loadtxt(simcatalog, comments='#')               #load the 'fake-source'catalog
        tot_sim_mag.append(data[:,2])

      #---SAVE DATA TO FILE ---

      if CIcut==1:
        outputfile = 'recovered_magnitudes_{}pc_CIcut.txt'.format(pc)
      elif CIcut==0:
        outputfile = 'recovered_magnitudes_{}pc.txt'.format(pc) 

      rec_out_mag=array(list(itertools.chain(*rec_out_mag)))
      rec_out_mag_err=array(list(itertools.chain(*rec_out_mag_err)))
      rec_sim_mag=array(list(itertools.chain(*rec_sim_mag)))
      ci_values=array(list(itertools.chain(*ci_values)))

      file = open(outputfile, "w")
      file.write('#simulated mag\t recovered mag\t expected error\t CI-value\n')
      for a in arange(len(rec_out_mag)):
        b=str(rec_sim_mag[a])
        c=str(rec_out_mag[a])
        d=str(rec_out_mag_err[a])
        e=str(ci_values[a])
        newline = '{} {}  {}  {}\n'.format(b,c,d,e)
        file.write(newline)
      file.close()
      os.rename(outputfile, pydir+'/recovery/'+outputfile)


      #----------------------------------------------------------------------------------------------
      #SECTION 2: BINNING THE DATA - CALCULATING RECOVERY RATES
      #----------------------------------------------------------------------------------------------
      if CIcut==1:
        magnitude_catalog='recovery/recovered_magnitudes_{}pc_CIcut.txt'.format(pc)
      else:
        magnitude_catalog='recovery/recovered_magnitudes_{}pc.txt'.format(pc)

      mag_data=genfromtxt(magnitude_catalog)

      
      if size(mag_data)>4:
        differences=mag_data[:,0]-mag_data[:,1]
        avg_difference=sum(differences)/len(mag_data[:,0])
        rec_out_mag=rec_out_mag+avg_difference

      tot_sim_mag=array(list(itertools.chain(*tot_sim_mag)))

      bin_on_sim_bins,binedges=histogram(rec_sim_mag,bins=arange(minmag,maxmag+binsize,binsize))
      bin_on_sim_bins=array(list(map(float,bin_on_sim_bins)))

      

      bin_on_rec_bins,binedges=histogram(rec_out_mag,bins=arange(minmag,maxmag+binsize,binsize))
      bin_on_rec_bins=array(list(map(float,bin_on_rec_bins)))


      actual_bins,binedges=histogram(tot_sim_mag,bins=arange(minmag,maxmag+binsize,binsize))

      if CIcut==1:
        bin_on_sim_CI=bin_on_sim_bins/actual_bins*100.
        bin_on_rec_CI=bin_on_rec_bins/actual_bins*100.
      else:
        bin_on_sim=bin_on_sim_bins/actual_bins*100.
        bin_on_rec=bin_on_rec_bins/actual_bins*100.

      percent_output=sum(tot_per_cent)/len(tot_per_cent)
      print('')
      print('Total percent recovered:',round(percent_output,2))
      print('')


    #---SAVE THE BINNED DATA TO FILE---
    outputfile='rec_per_mag_{}pc.txt'.format(pc)
    file = open(outputfile, "w")
    file.write('#rec. before CI cut\t  rec. after CI cut \t rec. before CI cut\t  rec. after CI cut \n')
    file.write('#binned on sim mag \t binned on sim mag \t binned on rec mag \t binned on rec mag \n')
    for a in arange(len(bin_on_sim)):
      b=str(round(bin_on_sim[a],1))
      c=str(round(bin_on_sim_CI[a],1))
      d=str(round(bin_on_rec[a],1))
      e=str(round(bin_on_rec_CI[a],1))

      newline=b+'\t'+c+'\t'+d+'\t'+e+'\n'
      file.write(newline)
    file.close()
    os.rename(outputfile, pydir+'/recovery/'+outputfile)

  #calculate total recovery rates
  data=zeros(len(actual_bins))
  CI_data=zeros(len(actual_bins))
  for pc in reffs:
    if type(pc)=='int':
      pc=round(pc+0.1-0.1,1)

    percentage_catalog1='recovery/rec_per_mag_{}pc.txt'.format(pc) 
    rec_data=loadtxt(percentage_catalog1, comments='#')
    data=data+rec_data[:,0]
    CI_data=CI_data+rec_data[:,1]
  data=data/size(reffs)
  CI_data=CI_data/size(reffs)

  outputfile='total_rec_per_mag.txt'
  file = open(outputfile, "w")
  file.write('#Column 1: rec. before CI cut\t Column 2: rec. after CI cut \n')
  file.write('#binned on sim mag \n')
  for a in arange(len(bin_on_sim)):
    b=str(round(data[a],1))
    c=str(round(CI_data[a],1))

    newline=b+'\t'+c+'\n'
    file.write(newline)
  os.rename(outputfile, pydir+'/recovery/'+outputfile)



  #-----------------------------------------------------------------------------------------
  #SECTION 3 - CALCULATING THE 50 AND 90 % COMPLETENESS LIMITS
  #-----------------------------------------------------------------------------------------

  outputfile='completeness_limits.txt'
  file = open(outputfile, "w")
  file.write('#90% lim\t 90% lim after CI cut \t 50% lim \t 50% lim after CI cut \n')

  #Calculate total comp limits
  lim90=float('NaN')
  lim90_CI=float('NaN')
  lim50=float('NaN')
  lim50_CI=float('NaN')
  for CI in [0,1]:
    magnitudes=arange(minmag,maxmag,binsize)+0.5*binsize
    if CI==0:
      data=data
    else:
      data=CI_data

    for a in arange(len(data)-1):
      if data[a]>=90 and data[a+1]<=90:
        k=(data[a+1]-data[a])/(magnitudes[a+1]-magnitudes[a])
        m=data[a]-k*magnitudes[a]     
        if CI==0:
          lim90=(90-m)/k
        elif CI==1:
          lim90_CI=(90-m)/k
      if data[a]>=50 and data[a+1]<=50:
        k=(data[a+1]-data[a])/(magnitudes[a+1]-magnitudes[a])
        m=data[a]-k*magnitudes[a]
        if CI==0:
          lim50=(50-m)/k
        elif CI==1:
          lim50_CI=(50-m)/k
  file = open(outputfile, "a")
  a=str(lim90)
  b=str(lim90_CI)
  c=str(lim50)
  d=str(lim50_CI)
  e='#Total recovery'.format(pc)
  newline=a+'\t'+b+'\t'+c+'\t'+d+'\t'+e+'\n'
  file.write(newline)
  file.close()

  #calculate completeness limits for each size
  for pc in reffs:
    if type(pc)=='int':
      pc=round(pc+0.1-0.1,1)

    lim90=float('NaN')
    lim90_CI=float('NaN')
    lim50=float('NaN')
    lim50_CI=float('NaN')

    for CI in [0,1]:
      inputfile='recovery/rec_per_mag_{}pc.txt'.format(pc)
      inputdata=loadtxt(inputfile,comments='#',usecols=(0,1))
      magnitudes=arange(minmag,maxmag,binsize)+0.5*binsize
      if CI==0:
        data=inputdata[:,0]
      else:
        data=inputdata[:,1]

      for a in arange(len(data)-1):
        if data[a]>=90 and data[a+1]<=90:
          k=(data[a+1]-data[a])/(magnitudes[a+1]-magnitudes[a])
          m=data[a]-k*magnitudes[a]     
          if CI==0:
            lim90=(90-m)/k
          elif CI==1:
            lim90_CI=(90-m)/k
        if data[a]>=50 and data[a+1]<=50:
          k=(data[a+1]-data[a])/(magnitudes[a+1]-magnitudes[a])
          m=data[a]-k*magnitudes[a]
          if CI==0:
            lim50=(50-m)/k
          elif CI==1:
            lim50_CI=(50-m)/k

    #Save Completeness limits to file
    file = open(outputfile, "a")
    a=str(lim90)
    b=str(lim90_CI)
    c=str(lim50)
    d=str(lim50_CI)
    e='#{}pc Sources'.format(pc)

    newline=a+'\t'+b+'\t'+c+'\t'+d+'\t'+e+'\n'
    file.write(newline)
    file.close()
  os.rename(outputfile, pydir+'/recovery/'+outputfile)
else:
  print('\n Recovery process has been skipped! \n')
#-----------------------------------------------------------------------------------------
#SECTION 5 - PLOTTING THE RESULTS
#-----------------------------------------------------------------------------------------
#First - check whether to plot or not

if (flagstep5=='yes' or flagstep5=='YES'):
  import matplotlib.pyplot as plt
  magnitude_bins=arange(minmag,maxmag,binsize)

  datafile='recovery/total_rec_per_mag.txt'
  tot_data,tot_CI_data=loadtxt(datafile,unpack=True)
  limit_file='recovery/completeness_limits.txt'
  limit_data=loadtxt(limit_file, comments='#')
  limit_90_CI=limit_data[:,1]
  limit_50_CI=limit_data[:,3]
  limit_90=limit_data[:,0]
  limit_50=limit_data[:,2]

  fig,ax=plt.subplots(1,2,figsize=(9,5))

  #plot total recovery before CI-cut:
  ca=ax[0]
  ca.plot(magnitude_bins+binsize/2,tot_data,color='grey',linewidth=2)
  ca.plot(magnitude_bins+binsize/2,tot_data,'o',color='grey',markeredgecolor='#4D4D4D')
  ca.set_title('Total recovery')
  ca.set_ylabel('Completeness [%]')
  ca.set_xlabel('Mag')
  ca.set_ylim((0,105))
  ca.set_xlim(minmag,maxmag)

  #Plot comp-limits:
  ca.plot([20,limit_90[0]],[90,90],color='#60BD68',linewidth=2,linestyle='--')
  ca.text(20.07,91,r'90%',fontweight='bold',color='#60BD68')
  ca.plot([limit_90[0],limit_90[0]],[0,90],color='#60BD68',linewidth=2,linestyle='--')

  ca.plot([20,limit_50[0]],[50,50],color='#FAA43A',linewidth=2,linestyle='--')
  ca.text(20.07,51,r'50%',fontweight='bold',color='#FAA43A')
  ca.plot([limit_50[0],limit_50[0]],[0,50],color='#FAA43A',linewidth=2,linestyle='--')


  #plot total recovery after CI-cut
  ca=ax[1]
  ca.plot(magnitude_bins+binsize/2,tot_CI_data,color='grey',linewidth=2)
  ca.plot(magnitude_bins+binsize/2,tot_CI_data,'o',color='grey',markeredgecolor='#4D4D4D')
  ca.set_title('Total recovery after CIcut')
  ca.set_ylabel('Completeness [%]')
  ca.set_xlabel('Mag')
  ca.set_ylim((0,105))
  ca.set_xlim(minmag,maxmag)
  #Plot comp-limits:
  ca.plot([20,limit_90_CI[0]],[90,90],color='#60BD68',linewidth=2,linestyle='--')
  if limit_90_CI[0]==limit_90_CI[0]:
    ca.text(20.07,91,r'90%',fontweight='bold',color='#60BD68')
  ca.plot([limit_90_CI[0],limit_90_CI[0]],[0,90],color='#60BD68',linewidth=2,linestyle='--')

  ca.plot([20,limit_50_CI[0]],[50,50],color='#FAA43A',linewidth=2,linestyle='--')
  if limit_50_CI[0]==limit_50_CI[0]:
    ca.text(20.07,51,r'50%',fontweight='bold',color='#FAA43A')
  ca.plot([limit_50_CI[0],limit_50_CI[0]],[0,50],color='#FAA43A',linewidth=2,linestyle='--')


  plt.tight_layout()

  plt.savefig('Total_completeness.pdf',bbox_inches='tight')
  os.rename('Total_completeness.pdf', pydir+'/recovery/'+'Total_completeness.pdf')



  #Make one plot for the non CI-cut data and one for the CI-cut data and repeat over all sizes.
  if size(reffs)==1:
    xplotsize=5
    yplotsize=5
  else:
    xplotsize=16
    yplotsize=10

  fig,ax=plt.subplots(2,size(reffs),sharex=True,sharey=True,figsize=(xplotsize,yplotsize))

  n=0

  for a in reffs:
    
    pc=round(a+0.1-0.1,1)
    percentage_catalog1='recovery/rec_per_mag_{}pc.txt'.format(pc)  
    rec_data=loadtxt(percentage_catalog1, comments='#')


    data=rec_data[:,0]
    CI_data=rec_data[:,1]


    limit_file='recovery/completeness_limits.txt'
    limit_data=loadtxt(limit_file, comments='#')

    lim90=limit_data[:,0]
    lim90_CI=limit_data[:,1]
    lim50=limit_data[:,2]
    lim50_CI=limit_data[:,3]

    if size(reffs)==1:
      ca=ax[0]
    else: 
      ca=ax[0,n]
    #ca.bar(magnitude_bins,data,binsize,color='grey',linewidth=0.5,edgecolor='#4D4D4D')


    ca.plot(magnitude_bins+binsize/2,data,color='grey',linewidth=2)
    ca.plot(magnitude_bins+binsize/2,data,'o',color='grey',markeredgecolor='#4D4D4D')

    ca.plot([20,limit_90[n+1]],[90,90],color='#60BD68',linewidth=2,linestyle='--')
    if limit_90[n+1]==limit_90[n+1]:
      ca.text(20.07,91,r'90%',fontweight='bold',color='#60BD68')
    ca.plot([limit_90[n+1],limit_90[n+1]],[0,90],color='#60BD68',linewidth=2,linestyle='--')

    ca.plot([20,limit_50[n+1]],[50,50],color='#FAA43A',linewidth=2,linestyle='--')
    if limit_50[n+1]==limit_50[n+1]:
      ca.text(20.07,51,r'50%',fontweight='bold',color='#FAA43A')
    ca.plot([limit_50[n+1],limit_50[n+1]],[0,50],color='#FAA43A',linewidth=2,linestyle='--')



    ca.text(24.5,95,'{} pc'.format(a),fontweight='bold')
    ca.set_ylim((0,105))
    ca.set_xlim((minmag,maxmag))

    if size(reffs)==1:
      ca=ax[1]
    else: 
      ca=ax[1,n]
    #ca.bar(magnitude_bins,CI_data,binsize,color='grey',linewidth=0.5,edgecolor='#4D4D4D')
    ca.plot(magnitude_bins+binsize/2,CI_data,color='grey',linewidth=2)
    ca.plot(magnitude_bins+binsize/2,CI_data,'o',color='grey',markeredgecolor='#4D4D4D')
    ca.plot([20,limit_90_CI[n+1]],[90,90],color='#60BD68',linewidth=2,linestyle='--')
    if limit_90_CI[n+1]==limit_90_CI[n+1]:
      ca.text(20.07,91,r'90%',fontweight='bold',color='#60BD68')

    ca.plot([limit_90_CI[n+1],limit_90_CI[n+1]],[0,90],color='#60BD68',linewidth=2,linestyle='--')

    ca.plot([20,limit_50_CI[n+1]],[50,50],color='#FAA43A',linewidth=2,linestyle='--')
    if limit_50_CI[n+1]==limit_50_CI[n+1]:
      ca.text(20.07,51,r'50%',fontweight='bold',color='#FAA43A')
    ca.plot([limit_50_CI[n+1],limit_50_CI[n+1]],[0,50],color='#FAA43A',linewidth=2,linestyle='--')


    ca.text(25,93,'{} pc \n CI cut'.format(a),fontweight='bold', horizontalalignment='center',verticalalignment='center',)
    ca.set_ylim((0,105))
    ca.set_xlim((minmag,maxmag))
    
    if size(reffs)==1:
      ax[0].tick_params(labelsize=10)
      ax[1].tick_params(labelsize=10)
    else:
      ax[0,n].tick_params(labelsize=10)
      ax[1,n].tick_params(labelsize=10)
    n=n+1
  
  if size(reffs)==1:
    yyl=ax[1].set_ylabel(r'Completeness [%]', fontweight='bold')
    yyl.set_position((yyl.get_position()[0],1)) # This says use the top of the bottom axis as the reference point.
    yyl.set_verticalalignment('center')

    plt.xlabel(r'Magnitudes', fontweight='bold')
  else:
    yyl=ax[1,0].set_ylabel(r'Completeness [%]', fontweight='bold')
    yyl.set_position((yyl.get_position()[0],1)) # This says use the top of the bottom axis as the reference point.
    yyl.set_verticalalignment('center')

    if size(reffs) % 2 == 0:
      xxl=ax[1,round(size(reffs)/2-0.01)].set_xlabel(r'Magnitudes', fontweight='bold')
      xxl.set_position((xxl.get_position()[1],1)) # This says use the top of the bottom axis as the reference point.
    else:
      xxl=ax[1,round(size(reffs)/2-0.01)].set_xlabel(r'Magnitudes', fontweight='bold')
    xxl.set_horizontalalignment('center')

  #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  fig.subplots_adjust(wspace=0.1)
  fig.subplots_adjust(hspace=0.1)

  plt.suptitle('Completeness levels before and after CI cut',fontweight='bold',fontsize='14')

  plt.savefig('Completenessfig.pdf',bbox_inches='tight')
  os.rename('Completenessfig.pdf', pydir+'/recovery/'+'Completenessfig.pdf')



  if size(reffs)==1:
    xplotsize=6
    yplotsize=5
  else:
    xplotsize=15
    yplotsize=5
  fig3,ax3=plt.subplots(1,size(reffs),figsize=(xplotsize,yplotsize), sharex=True,sharey=True)

  n=0
  for a in reffs:
    
    pc=round(a+0.1-0.1,1)
    mag_cat='recovery/recovered_magnitudes_{}pc.txt'.format(pc)  
    sim_mag,rec_mag,mag_err,CI=loadtxt(mag_cat, comments='#',unpack=True)

    if size(reffs)==1:
      ca=ax3
      ax3.tick_params(labelsize=10)
    else: 
      ca=ax3[n]
      ax3[n].tick_params(labelsize=10)

    ca.plot(sim_mag,mag_err,'o',color='grey',markeredgecolor='#4D4D4D',markersize=6)
   
    
    if size(reffs)==1:
      yyl=ax3.set_ylabel(r'Magnitude errors [mag]', fontweight='bold')
      

      plt.xlabel(r'Magnitudes', fontweight='bold')
    else:
      yyl=ax3[0].set_ylabel(r'Magnitude errors [mag]', fontweight='bold')

      if size(reffs) % 2 == 0:
        xxl=ax3[round(size(reffs)/2-0.01)].set_xlabel(r'Magnitudes', fontweight='bold')
        xxl.set_position((xxl.get_position()[1],0)) # This says use the top of the bottom axis as the reference point.
      else:
        xxl=ax3[round(size(reffs)/2-0.01)].set_xlabel(r'Magnitudes', fontweight='bold')
      xxl.set_horizontalalignment('center')

    n=n+1
  
  plt.tight_layout()
  fig3.subplots_adjust(wspace=0.1)
  fig3.subplots_adjust(hspace=0.1)
  fig3.subplots_adjust(top=0.88)

  plt.suptitle('Simulated magnitude vs Magnitude errors',fontweight='bold',fontsize='14',y=0.95)
  plt.savefig('Mag_err_plot.pdf',bbox_inches='tight')
  os.rename('Mag_err_plot.pdf', pydir+'/recovery/'+'Mag_err_plot.pdf')

  plt.show()
  print('All requested operations have been performed!')


else:
  print('All requested operations have been performed!')
