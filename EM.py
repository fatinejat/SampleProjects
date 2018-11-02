
# Own implementations
import OwnITK as oitk
import NameAssign as na
import OwnPlotClass as opc
import Distributions as distr
import Normalisation as norm

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
#from SimpleITK.SimpleITK import sitkBox, sitkBall
import sh
import os
import pdb
import time

''' 
Global variables within this module 
'''
# Variable specifying if segmentations of some modalities or to be a 
# subpart from segmentations of other modalities => specified to brain
# MR scans in this code
inclusion_app = True
# Print out gaussian parameters within each iteration
verbose = False
# Some value indicating how far the tumor distribution is different 
# from the others
threshold = 0

maxIteration = 15

flat_prior = 0.5

def ems_segment(hyperDataFiles, hypoDataFiles, maskFiles, atlasFiles, 
		tumor_lower_co = None, tumor_upper_co = None,
                time_index = None,tumor_center = None, show = True):
    '''     
    Segments a brain image in tumor tissue and in brain tissues (white
    matter, grey matter and cerebrospinal fluid).
    
    Parameters
    ----------
    - hyperDataFiles: list of strings
        path to Images in which tumor appears hyperintensive
    - hypoDataFiles: list of strings
        paths to images in which tumor appears hypointensive
    - maskFiles: list of strings
        paths to binary mask(s) of one or more of the images in hyper- and
        hypoDataFiles. Defaults to intersection of registered
        atlas in atlasFiles and image data in hyper- and hypoDataFiles
    - atlasFiles: list of 3 strings
        the 1st, 2nd and 3rd string reference atlas files of grey matter, 
        white matter and cerebrospinal fluid *respectively*.
    - time_index: integer [optional]
        integer denoting a time index, used for paths to which results are written 
        (only integrated in filename so that multiple image series can be stored with
        different filenames)
    - tumor_center: tuple of three integers [optional]
        index of tumor_center, for visualisation purposes used only
    - show: boolean [optional]
        Generate and visualise images during runtime or not
    
    .. NOTE:: All files should contain preprocessed images, in the sense that,
        all images should have the same resolution, the same isotropic pixel
        dimensions and they should be in the same reference space. 
        Accepted file extensions are: *TIFF, JPEG, PNG, BMP, DICOM, GIPL, 
        Bio-Rad, LSM, Nifti, Analyze, SDT/SPR (Stimulate), Nrrd or VTK images*.
        
    Returns
    -------
    All of the returned itk images are also written to files in a 
    subdirectory of the current directory: ::./results/ .
    - tumorImages: list of 3D itk images
        Soft tumor segmentations for each supplied file following the order of
        the concatenation of hyperDataFiles and hypoDataFiles. 
        Written to ::./results/<filename>_tumor_t<time_index>.nii ,
        where filename is the original filename extracted from the supplied 
        filenames (hypo- and hyperDataFiles)
    - gmImage: 3D itk image
        Soft grey matter segmentation. Written to ::./results/gm.nii .
    - wmImage: 3D itk image
        Soft white matter segmentation. Written to ::./results/wm.nii . 
    - csfImage: 3D itk image
        Soft cerebrospinal fluid segmentation. Written to ::./results/csf.nii .
    '''

    print('')
    print('Simple EMS has started')
    print('no biasfield correction and no MRF spatial regularisator')
    print('Input data is supposed to be preprocessed')
    print('Images, masks and atlas should be in the same reference space')
    print('No tumor will be allowed in CSF voxels')
    print('')
    
    start_time = time.time()

    # Concatenate datafiles and store which ones have hyperintense tumors.
    if hyperDataFiles is None:
        hyperDataFiles = []
    if hypoDataFiles is None:
        hypoDataFiles = []
    if isinstance(hyperDataFiles,str):
        hyperDataFiles = [hyperDataFiles]
    if isinstance(hypoDataFiles,str):
        hypoDataFiles = [hypoDataFiles]
    
    dataFiles = np.concatenate((hyperDataFiles,hypoDataFiles),axis=0)
    modalities = getModalityStrings(dataFiles)
        
    nrOfChannels = len(dataFiles) 
    hyper = np.zeros(shape=(nrOfChannels,))
    hyper[0:len(hyperDataFiles)] = 1
    
    # Check data dimensionalities.
    maxDim = 0
    for channel in range(nrOfChannels):
        noNeed, tmpDIM, tmpSpaces = oitk.getItkImageData(dataFiles[channel])
        dataType = noNeed.dtype
        if maxDim < np.prod(tmpDIM):
            if channel is not 0:
                ask_continue_datasize('data',channel+1)
            DIM = tmpDIM
            maxDim = np.prod(DIM)
            dataImagePrototype = oitk.getItkImage(dataFiles[channel])
            dataSpaces = tmpSpaces
        print('Data size of channel '+str(channel + 1)+' is: '+str(tmpDIM)) 
        
    # Read brain mask(s).
    if maskFiles != None:
        playing = np.ones(shape=(np.prod(DIM)),dtype='?')
        if isinstance(maskFiles,str):
            maskFiles = [maskFiles]
        else:
            print('You supplied '+str(len(maskFiles))+' masks')
            print('The intersection will be taken as the true mask')
        for fileNb in range(len(maskFiles)):
            mask, DIMmask = oitk.getItkImageData(maskFiles[fileNb])[0:-1]        
            if DIMmask!=DIM:
                ask_continue_datasize('mask',fileNb+1)
            playing = np.logical_and(playing, np.ravel(mask))
    
    # Read atlas.
    classGM = 0; classWM = 1; classCSF = 2;
    nrOfClasses = len(atlasFiles)
    atlasTmp = np.zeros(shape=(nrOfClasses,np.prod(DIM)))
    for classNr in range(nrOfClasses):
        atlasRaw, DIMatlas, noNeed = oitk.getItkImageData(atlasFiles[classNr])
        if DIMatlas!=DIM:
            ask_continue_datasize('atlas',fileNb+1)
        atlasTmp[classNr,:] = np.ravel(atlasRaw)      
    atlasTmp = norm.window(atlasTmp,0,1)  
    
    # Define brain voxels in case mask is not given.
    sumAtlas = np.sum(atlasTmp, axis=0)
    if maskFiles == None:   
        playing = np.ones(shape=(np.prod(DIM),),dtype='?')
        playing[sumAtlas>0.5] = True
        playing[sumAtlas<=0.5] = False
    else: #throw out brain voxels with nonzero atlas probabilities
        playing = np.logical_and(sumAtlas>0,playing)           
    nrOfBrainVoxels = np.count_nonzero(playing)

    # Normalise and mask the atlas.
    atlas = np.zeros(shape=(nrOfClasses,nrOfBrainVoxels))
    for classNr in range(nrOfClasses):
        atlasTmp[classNr,playing] = atlasTmp[classNr,playing]/sumAtlas[playing]
        atlas[classNr,:] = atlasTmp[classNr,playing]    
        
    # Read, normalise and mask the data.
    data = np.zeros(shape=(nrOfChannels,nrOfBrainVoxels),dtype = dataType)
    dataNotNorm = np.zeros(shape=(nrOfChannels,np.prod(DIM)),dtype = dataType)
    for channel in range(nrOfChannels):
        image, dataDIM, dontNeed = oitk.getItkImageData(dataFiles[channel])
        flatImage = np.ravel(image)  
        dataNotNorm[channel,:] = flatImage
        data[channel,:] = flatImage[playing]
        
    inclusion = getInclusion(dataFiles)    
        
    # Specify how many gaussians are used to model each tissue class
    nc = np.array([1, 1, 2]) # Number of Gaussians per tissue class
    lkp = [] # Class to which a certain gaussian belongs
    for i in range(len(nc)):
        lkp.extend(np.ones(nc[i],dtype='int')*(i)) 
    lkp = np.array(lkp)
    nrOfGaussians = len(lkp)

    # Read out tumor mask
    print 'original image dimensions: '+str(DIM)
    if tumor_lower_co is None:
        tlc = np.zeros(3,dtype='int')
    else:
        tlc = tumor_lower_co
    if tumor_upper_co is None:
        tuc = DIM - np.ones(3,dtype='int')
    else:
        tuc = tumor_upper_co
    
    DIMbox = (tuc - tlc) + np.ones(3,dtype='int')
    print 'tumor box dimensions: '+str(DIMbox)
    tumorMask = np.zeros(DIM)
    tumorMask[tlc[0]:tuc[0]+1,tlc[1]:tuc[1]+1,tlc[2]:tuc[2]+1] = 1
    tumorMask = np.ravel(tumorMask)[playing]
        
    # Initialise brain tumor atlas
    tumorAtlas = np.zeros(shape=(2,nrOfBrainVoxels))
    tumorBit = 1; noTumorBit = 0;
    tumorAtlas[noTumorBit,:] = tumorMask * flat_prior
    tumorAtlas[tumorBit,:] = 1 - tumorAtlas[noTumorBit,:] 
    
    # Initialise tumor classification.
    tumorClassification = np.zeros(shape=(nrOfChannels,nrOfBrainVoxels))
    for channel in range(nrOfChannels):
        tumorClassification[channel,:] = tumorAtlas[tumorBit,:]
    
    # Specifiy tumor classes: tumor is allowed only in WM and GM, not in CSF.
    tumorClasses = np.zeros((nrOfClasses,), dtype ='?')
    tumorClasses[classWM] = 1; tumorClasses[classGM] = 1;
    noTumorClasses = 1 - tumorClasses;
    
    # Specify tumor presence configurations along the channels.
    nrTumorConf = 2**nrOfChannels
    tumorConf = np.zeros(shape=(nrTumorConf,nrOfChannels),dtype='?')
    for conf in range(nrTumorConf): 
        tumorConf[conf,:] = [bool(int(i)) for i in 
            list(format(conf,'0'+str(nrOfChannels)+'b'))]
    tumorTransitionMatrix = tumorConf 
    # throw out unrealistic tumor transitions
    if inclusion: 
        tumorTransitionMatrix = inclusion_restrictions(tumorTransitionMatrix,modalities)
    nrTumorTrans = tumorTransitionMatrix.shape[0]
    
    # Calculate matrix with all tumor configurations over all tissue classes.   
    nrTissueConf = np.dot(nc,noTumorClasses) + np.dot(nc,tumorClasses)*nrTumorTrans
    classes = np.zeros(shape=(nrTissueConf))
    gaussians = np.zeros(shape=(nrTissueConf))
    tumorTransitions = np.zeros(shape=(nrTissueConf,nrOfChannels),dtype='?')
    ind = 0
    for gauss in range(nrOfGaussians):
        classNr = lkp[gauss]
        if tumorClasses[classNr] == noTumorBit:
            classes[ind] = classNr
            gaussians[ind] = gauss
            tumorTransitions[ind,:] = 0
            ind = ind +1;
        if tumorClasses[classNr] == tumorBit:    
            classes[ind:ind+nrTumorTrans] = classNr
            gaussians[ind:ind+nrTumorTrans] = gauss
            tumorTransitions[ind:ind+nrTumorTrans,:] = tumorTransitionMatrix
            ind = ind + nrTumorTrans
    nrTissueConf = len(classes)           

    mask = playing.astype('int')
  
    # Initialisations for intensity guassians.
    normalMeans = np.zeros(shape=(nrOfGaussians, nrOfChannels))
    normalVars = np.zeros(shape=(nrOfGaussians, nrOfChannels, nrOfChannels))
    tumorMeans = np.zeros(shape=(nrOfChannels,))
    tumorVars = np.zeros(shape=(nrOfChannels, nrOfChannels))
    tissueConfVars = np.zeros(shape=(nrTissueConf, nrOfChannels, nrOfChannels))  
    tissueConfMeans = np.zeros(shape=(nrTissueConf, nrOfChannels))  
  
    # Initialise healthy tissue classification.
    normalClassification = np.zeros(shape=(nrOfClasses,nrOfBrainVoxels))
    gaussianClassification = np.zeros(shape=(nrOfGaussians,nrOfBrainVoxels))
    atlasPrior = np.zeros(shape=(nrOfGaussians,nrOfBrainVoxels))
    for gauss in range(nrOfGaussians):
        classNr = lkp[gauss]
        normalClassification[classNr,:] = atlas[classNr,:] 
        atlasPrior[gauss,:] = atlas[classNr,:]/nc[classNr]
        gaussianClassification[gauss,:] = atlasPrior[gauss,:]
 
    # Initialisations for iterations.
    converged = 0
    iteration = 0
    
    # Numerical declarations.
    global EPS, TINY
    EPS = np.finfo(np.double).eps; TINY = np.finfo(np.double).tiny;
    
    # Log-likelihood initialisations.
    logLLHArray = np.zeros((maxIteration,))
    logLLH = 1/EPS
    relativeLogLLH = 1
    
    # Create directory to store results.
    dir_results = os.path.dirname(dataFiles[0])+'/ems_results/'
    make_dir(dir_results)
    if inclusion:
        dir_results = os.path.join(dir_results,'with_inclusion/')
        make_dir(dir_results)
    nii_dir = dir_results + 'images/'
    make_dir(nii_dir)
    fig_dir = dir_results + 'figures/'
    make_dir(fig_dir)

    # Visualisation.
    if show:
        opInst = opc.OwnPlot(visual_center=tumor_center,visual_axis=0,spacing=dataSpaces[::-1],
                 file_extension='.png',fig_dir=fig_dir)
        opInst.show3Dplot(to_matrix(mask, DIM),None,'Mask')
        opInst.show5Dplot(to_matrix(atlas, DIM, playing),None,'Input Atlas')
        opInst.show5Dplot(to_matrix(dataNotNorm,DIM),None,'Input Data')
        for channel in range(nrOfChannels):
            dataSlice = dataNotNorm[channel,:].reshape(DIM)
            opInst.show2Dplot(dataSlice,None,'') 
    # Start iterating
    while converged==False: 
        
        iteration = iteration + 1
        previousLogLLH = logLLH
        
        print('---------------')
        print('Iteration ' + str(iteration))
       # if verbose:
          #  memory = float( awk(ps('u','-p',os.getpid()),
          #                         '{sum=sum+$6}; END {print sum/1024}') )
        #    print('Memory used: ' + str(memory) + ' MB')
       # print('')
  #
        # Estimate Gaussian mixture parameters mean and covariance
        print('Estimating mixture parameters')
        print('')

        # 1. Calculate normal means
        new_normalWeights = np.zeros([nrOfGaussians,nrOfChannels,nrOfBrainVoxels])
        for gauss in range(nrOfGaussians):
            new_normalWeights[gauss,:,:] = np.tile(gaussianClassification[gauss,:],
                (nrOfChannels,1)) * (1-tumorClassification)
            
            for channel in range(nrOfChannels):
                new_normalWeights[gauss,channel,:] = norm.rescale(new_normalWeights[gauss,channel,:])
            
            # If weights all become zero, reset to weights of previous iteration
            if iteration > 1:
                for channel in range(nrOfChannels):
                    if np.sum(new_normalWeights[gauss,channel,:]) == 0:
                        new_normalWeights[gauss,channel,:] = \
                                np.copy(normalWeights[gauss,channel,:])
                
            normalMeans[gauss,:] = distr.getMean(data,new_normalWeights[gauss,:,:])  
            normalVars[gauss,:,:], nonsymm = distr.getVariance(data, 
                                new_normalWeights[gauss,:,:], normalMeans[gauss,:],
                                verbose=verbose)
            if nonsymm:
                pdb.set_trace()
                
        normalWeights = new_normalWeights
              
        # Split gaussians within the same tissue class
        # gaussians would remain identical after initialisation if we do not push 
        # them apart artificially                  
        if iteration<=3:
            print('Splitting clusters of normal tissue classes')
            if verbose: 
                print('Old means: '+str(normalMeans))
            normalMeans = splitClassMeans(normalMeans,lkp)       
                    
            if verbose:                          
                print('Normal means of iteration '+str(iteration)+': ')
                print(str(normalMeans))
                print('')
                           
        # 2. Calculate tumor means
        tumorWeights = np.zeros([nrOfChannels,nrOfBrainVoxels])
        for channel in range(nrOfChannels):
            # Initialise weights with previous tumor classification
            weights = tumorClassification[channel,:]
            
            # Do a binary threshold of the previous tumor classification
            weights[weights<np.mean(weights[weights>0])] = 0.1
            weights[weights>=np.mean(weights[weights>0])] = 0.9
            
            # do not weight with non-presence of normal tissue, 
            # normalCLassification is overlapping with tumorClassification

            # Throw out pixels which are far from hyper- or hypo-intensive
            channelNormalMeans = normalMeans[:,channel]
            meanGM = np.average(channelNormalMeans[lkp==classGM])
            meanWM = np.average(channelNormalMeans[lkp==classWM])

            # Weights should be inversily related to the WM probabilities
            WMprob = np.zeros((nrOfBrainVoxels,))
            for gauss in np.nditer(np.where(lkp==classWM)):
                gaussWMprob = distr.gaussianEval_multiVar(data[channel,:], normalMeans[gauss,
                    channel], normalVars[gauss,channel, channel],False)*gaussianClassification[gauss,:]
                WMprob = np.amax([WMprob, gaussWMprob],axis=0)                                
            notWMprob = 1 - WMprob
            notWMprob = norm.window(notWMprob,0,1)
            weights = weights * notWMprob 
            
            df = threshold*np.absolute(meanGM - meanWM) # distance between WM & GM means
            if(hyper[channel]==1):
#                 print('WM Mean for '+modalities[channel]+': '+str(meanWM))
#                 print('GM Mean for '+modalities[channel]+': '+str(meanGM))
                weights[data[channel,:] <= (meanWM + df)] = 0       
            else:
                weights[data[channel,:] >= (meanGM)] = 0  
            
            try:
                if(np.sum(weights) == 0):                    
                    raise ZeroDivisionError('Tumor weights are all zero')
            except ZeroDivisionError:
                print('No tumor pixels present to update tumor gaussian')
                print('Problem in channel '+modalities[channel])
                print('Maybe no tumor is present in the input images?')
                print('If tumor is present, try loosening the tumor threshold') 
                print('Check #df in code')
                raise
                
            tumorWeights[channel,:] = norm.rescale(weights)
            
        if np.count_nonzero(tumorWeights<0) is not 0:
            print('Warning: Negative weights present for gaussian update')
            pdb.set_trace()        
        tumorMeans = distr.getMean(data,tumorWeights)
        tumorVars, nonsymm = distr.getVariance(data,tumorWeights,tumorMeans) 

        if nonsymm:
            pdb.set_trace()
        
        if verbose:
            print('Tumor means of iteration '+str(iteration)+': ')
            print(str(tumorMeans))
            print('')
        
        # 3. Calculate covariance matrices for each tissue configuration map            
        for tissueConf in range(nrTissueConf):
            gauss = gaussians[tissueConf]
            tumorArray = tumorTransitions[tissueConf,:]
            tumorInd = tumorArray==tumorBit
            normInd = tumorArray==noTumorBit

            currentMeans = np.zeros((nrOfChannels,))
            currentMeans[tumorInd] = tumorMeans[tumorInd]   
            
            if np.all(tumorArray==tumorBit):
                tissueConfMeans[tissueConf,:] = currentMeans 
                tissueConfVars[tissueConf,:,:] = tumorVars

            else:
                currentMeans[normInd] = normalMeans[gauss,normInd]           
                tissueConfMeans[tissueConf,:] = currentMeans                 
                if np.all(tumorArray==noTumorBit):
                    tissueConfVars[tissueConf,:,:] = normalVars[gauss,:,:]                        
                else:
                    currentWeights = np.zeros((nrOfChannels,nrOfBrainVoxels))
                    currentWeights[normInd,:] = normalWeights[gauss,normInd,:]           
                    currentWeights[tumorInd,:] = tumorWeights[tumorInd,:]  
                    tissueConfVars[tissueConf,:,:],nonsymm = \
                                distr.getVariance(data,currentWeights,currentMeans)
                    if nonsymm:
                        pdb.set_trace()
                    
        print('Estimating tissue classification')
        print('')
        # Calculate classification
        tissueClassification = np.zeros(shape=(nrTissueConf,nrOfBrainVoxels))        
        LikeliProduct = np.zeros(shape=(nrOfBrainVoxels,))
        for tissueConf in range(nrTissueConf):
            
            classNr = classes[tissueConf]
            gauss = gaussians[tissueConf]
            tumorArray = tumorTransitions[tissueConf,:]
            
            LikeliProduct[:] = atlasPrior[gauss,:]
            
            tumorAdaptedData = np.copy(data)
            for channel in range(nrOfChannels):
                # if no tumor in channel
                if(tumorArray[channel]==noTumorBit):
                    LikeliProduct = LikeliProduct * tumorAtlas[noTumorBit,:]
                # if tumor in channel    
                if(tumorArray[channel]==tumorBit):    
                    LikeliProduct = LikeliProduct * tumorAtlas[tumorBit,:]
                    # Hyperintense channels are modeled by a sigmoid function
                    # Therefore, data values are temporally adjusted so that very high intensities
                    # are at least as likely to belong to the tumor as the gaussian mean intensity
                    if hyper[channel]:
                        data_tmp = tumorAdaptedData[channel,:]
                        data_tmp = data_tmp - np.sqrt(tumorVars[channel,channel])
                        data_tmp[data_tmp<0] = 0
                        data_tmp[data_tmp>tumorMeans[channel]] = tumorMeans[channel]  
                        tumorAdaptedData[channel,:] = data_tmp     
            
            
            justEval = distr.gaussianEval_multiVar(tumorAdaptedData, tissueConfMeans[tissueConf,:], 
                                    tissueConfVars[tissueConf,:,:], True)
            LikeliProduct = LikeliProduct * justEval
                
            tissueClassification[tissueConf,:] = np.amax([LikeliProduct,
                        np.zeros_like(LikeliProduct)],axis=0)  
        
        # Calculate likelihood and normalise classification
        likelihood = np.fmax(np.sum(tissueClassification,axis=0),TINY)
        tissueClassification = tissueClassification / \
            np.tile(likelihood,(nrTissueConf,1))
        logLLH =  np.sum( np.log( likelihood ) )
        logLLHArray[iteration-1] = logLLH
        relativeLogLLH = np.absolute( (previousLogLLH - logLLH) / logLLH )        
        #converged = (relativeLogLLH<0.0001) | (iteration==maxIteration) 
        converged = iteration == maxIteration
        
        print('=> logLikelihood = ' + str(logLLH))
        print('=> relative change in loglikelihood = ' + str(relativeLogLLH))
        print('')
      
        # Calculating normal tissue classification and tumor classification.      
        print('Calculating normal tissue and tumor classification')
        print('')
        normalClassification = np.zeros(shape=(nrOfClasses,nrOfBrainVoxels))
        gaussianClassification = np.zeros(shape=(nrOfGaussians,nrOfBrainVoxels))
        tumorClassification = np.zeros(shape=(nrOfChannels,nrOfBrainVoxels))        
        
        for tissueConf in range(nrTissueConf):
            configurationProb = tissueClassification[tissueConf,:]
            tumorArray = tumorTransitions[tissueConf,:]
            
            tumorChannels = np.where(tumorArray == tumorBit)
            tumorClassification[tumorChannels,:] += configurationProb*tumorMask
                
            classNr = classes[tissueConf]
            gauss = gaussians[tissueConf]
            normalClassification[classNr,:] += configurationProb
            gaussianClassification[gauss,:] += configurationProb 
			
	    tumorAtlas[tumorBit,:] = np.sum(tumorClassification,axis=0) / \
            	nrOfChannels 
            tumorAtlas[noTumorBit,:] = 1 - tumorAtlas[tumorBit,:]
            
        tumorClassification = norm.window(tumorClassification,0,1)
        normalClassification = norm.window(normalClassification,0,1)
        gaussianClassification = norm.window(gaussianClassification,0,1)
        
        oitk.writeItkImage(oitk.makeItkImage(to_matrix(tumorClassification[1,:],
                DIM, playing),dataImagePrototype),
                '/home/alberts/Downloads/tumor_'+\
                modalities[1]+'_'+str(iteration)+'.nii')
                    
        # Visualisation   
        if converged:   
            opInst.setVisualCenter(tumor_center)       
            opInst.show4Dplot(to_matrix(tumorClassification, DIM, playing),
                None,'Tumor Segmentation Per Channel Iter '+str(iteration))     
            opInst.show5Dplot(to_matrix(normalClassification, DIM, playing),
                None,'Tissue segmentations Iter '+str(iteration))
            for channel in range(nrOfChannels):
                dataSlice = to_matrix(dataNotNorm[channel,:], DIM)
                tumorSlice = to_matrix(tumorClassification[channel,:],DIM,playing)
                opInst.show2Dplot(dataSlice,tumorSlice,
                                  'Tumor Segmentation channel '+str(channel)) 
        
    
    # Write tissue segmentations    
    if time_index is not None:
        time_string = '_t'+str(time_index)
    else:
        time_string = ''
        
    for tissueClass in range(nrOfClasses):
                
        segmentTissueClass = to_matrix(normalClassification[tissueClass,:], DIM, playing)
        image = oitk.makeItkImage(segmentTissueClass,dataImagePrototype) 

        if tissueClass==classGM:
            gmImage = image
            oitk.writeItkImage(gmImage,nii_dir+'gm'+time_string+'.nii')
        if tissueClass==classWM:
            wmImage = image
            oitk.writeItkImage(wmImage,nii_dir+'wm'+time_string+'.nii')
        if tissueClass==classCSF:
            csfImage = image
            oitk.writeItkImage(image,nii_dir+'csf'+time_string+'.nii') 
    
    # Write channel-specific tumor segmentations.
    tumorChannels = np.zeros(shape=(nrOfChannels,DIM[0],DIM[1],DIM[2]))
    result_files = na.getEMsegmFiles(dataFiles, time_index, inclusion)
    tumorImages = []
    for channel in range(nrOfChannels):
        tumorChannelTmp = to_matrix(tumorClassification[channel,:], DIM, playing)
        tumorImage = oitk.makeItkImage(tumorChannelTmp,dataImagePrototype)
        oitk.writeItkImage(tumorImage,result_files[channel]) 
        print 'Tumor nii image written to '+result_files[channel]
        print 'Tumor volume '+str(np.sum(tumorClassification[channel,:]))
        tumorImages.append(tumorImage)
   
    # Plot log likelihood curve along iterations.
    logLLHArray = logLLHArray[0:iteration]   
    if show: 
        opInst.plotGraph(np.arange(iteration)+1,logLLHArray,'Log likelihood')

    # Print running time.    
    elapsed_time = time.time() - start_time    
    print('Convergence reached in '+str(elapsed_time)+' seconds')
    print('')
    
    plt.close('all')
        
    return tumorImages,wmImage,gmImage,csfImage

#############################################  


def ask_continue_datasize(fileType,index):
    ''' Warn user about nonuniform image dimensions, throw error.'''
    
    print('Unexpected image dimensions in '+fileType+' file '+str(index))
    print('Check registration during preprocessing')
    print('-> all files should contain images of same dimensions')
    raise RuntimeError('Algorithm aborted')
    return    
    
def make_dir(new_dir):
    ''' Make a new directory if it doesn't exist already'''
    
    dirPresent = False
    while dirPresent is not True:
        try:
            os.mkdir(new_dir)
        except OSError: #this directory already exists
            dirPresent = True    
    
def to_matrix(flatArray,DIM,playing=None):
    ''' Reshape flat array back into a matrix with dimensions DIM,
    if playing is given, these represent the indices in the matrix 
    which should be filled out with the values in flatArray, all other
    indices in the matrix are zero.'''
    
    if len(flatArray.shape) == 1:
        if playing is not None:
            flatArrayFull = np.zeros((np.prod(DIM)),)
            flatArrayFull[playing] = flatArray
        else:
            flatArrayFull = flatArray
            
        matrix = flatArrayFull.reshape(DIM)
        
    elif len(flatArray.shape) == 2:
        flatArrays = flatArray
        matrix = np.zeros((flatArray.shape[0],DIM[0],DIM[1],DIM[2]))
        
        for i in range(flatArray.shape[0]):
            flatArray = flatArrays[i]
            matrix[i,:,:,:] = to_matrix(flatArray,DIM,playing)
    
    return matrix

def inclusion_restrictions(matrix,modalities):
    
    print 'FATI: Adapt this piece of code to the modalities you are working with'
    all_modalities = ['t1','t1c','t2','flair']
    indices = []
    for i in range(len(all_modalities)):
        this_mod = all_modalities[i]
        this_ind = np.where(np.asarray(modalities) == this_mod)[0]
        if len(this_ind) == 1:
            this_ind = this_ind[0]
        else:
            this_ind = None
        indices.append(this_ind)
    
    combinationCount = 0
    for i in range(matrix.shape[0]):
        combination = np.zeros([1,matrix.shape[1]],dtype='?')
        combination[0,:] = matrix[i,:]
        log = 0
        if indices[0] is not None and indices[3] is not None:
            log = log + (combination[0][indices[0]] > combination[0][indices[3]])
        if indices[1] is not None and indices[3] is not None:
            log = log + (combination[0][indices[1]] > combination[0][indices[3]])
        if indices[2] is not None and indices[3] is not None:
            log = log + (combination[0][indices[2]] > combination[0][indices[3]])
        if log == 0:
            combinationCount = combinationCount + 1; 
            if combinationCount == 1:
                restricted_matrix = combination
            else:
                restricted_matrix = np.concatenate((restricted_matrix,combination))
                
    return restricted_matrix     

def splitClassMeans(means,classPerGaussian):
    ''' Split means within the same channel and class when they have the same 
    mean value.
    
     Parameters
    ----------
    - means: 2D array             
        Means per gaussian (1st axis) and per channel (2nd axis)
    - classPerGaussian: 1D array  
        Indices for each gaussian to which tissue class it belongs.'''       
        
    changes = [1.005, 0.995, 1.007, 0.993, 1.01, 0.99]
    nrOfChannels = means.shape[1]
    nrOfPriors = len(np.unique(classPerGaussian))
    for channel in range(nrOfChannels):
        channelMeans = means[:,channel]
        for classNr in range(nrOfPriors):
            classMeans = channelMeans[classPerGaussian == classNr]
            uniqueMeans = np.unique(classMeans)                
            nonuniques = (len(classMeans)!=len(uniqueMeans))
            while nonuniques:
                for mean in uniqueMeans:
                    if np.count_nonzero(classMeans == mean)>1:
                        ind = (classMeans == mean)
                        classMeans[ind] = [x * mean for x in changes[0:np.count_nonzero(ind)]] 
                uniqueMeans = np.unique(classMeans)                
                nonuniques = (len(classMeans)!=len(uniqueMeans))
            channelMeans[classPerGaussian==classNr] = classMeans
        means[:,channel] = channelMeans
    
    return means

def getInclusion(dataFiles):
    
    if inclusion_app == False:
        print('Inclusion will NOT hold')
        return False
      
    inclusion = True  
    modalities = getModalityStrings(dataFiles)
    if np.any(np.asarray(modalities) == '?'):
        print('Inclusion will NOT hold')
        inclusion = False
        
    return inclusion

def getModalityStrings(dataFiles):
    
    print 'Adapt this piece of code to the modalities you are working with'
    modalities = []
    for dataFile in dataFiles:
        modalities.append('?')
        
    return modalities

def morphological_filter(flatArray,DIM,playing,dataImagePrototype,
                        morphological_operation,radius=None):
    
    if len(flatArray.shape) == 2:
        filtered_flatArray = np.zeros_like(flatArray)
        itk_filtered = [None]*flatArray.shape[0]
        for channel in range(flatArray.shape[0]):
            filtered_flatArray[channel,:], itk_filtered[channel] = \
                morphological_filter(flatArray[channel],DIM,playing,
                    dataImagePrototype,morphological_operation,radius)
    else:
        matrix = to_matrix(flatArray,DIM, playing)
        itk_image = oitk.makeItkImage(matrix, dataImagePrototype)
       
        itk_filtered = oitk.morphological_itk(itk_image, 
                                         morphological_operation,radius)
        
        filtered_flatArray = np.ravel(sitk.GetArrayFromImage(itk_filtered))[playing]
    
    return filtered_flatArray, itk_filtered

###############################################################################################################
       
