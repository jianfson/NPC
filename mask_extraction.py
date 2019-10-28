from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import random
import dicom
from scipy import ndimage
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

#Some helper functions

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))


def interplote(points):
    added = []
    for i in range(len(points)-1):
        dist = np.linalg.norm(np.array(points[i+1]) - np.array(points[i]))
        if dist > 1.4:
            pair = [points[i], points[i+1]]

            if np.abs(points[i][0]-points[i+1][0]) > np.abs(points[i][1]-points[i+1][1]):

                min_idx = np.argmin([points[i][0],points[i+1][0]])
                xx = np.linspace(start=pair[min_idx][0], stop=pair[1-min_idx][0], num=pair[1-min_idx][0]-pair[min_idx][0]+2, dtype='int32')
                interp = np.interp(xx, [pair[min_idx][0],pair[1-min_idx][0]], [pair[min_idx][1],pair[1-min_idx][1]])
                for dummy in zip(xx, interp):
                    added.append([int(dummy[0]),int(dummy[1])])
                
            else:
                min_idx = np.argmin([points[i][1],points[i+1][1]])
                yy = np.linspace(start=pair[min_idx][1], stop=pair[1-min_idx][1], num=pair[1-min_idx][1]-pair[min_idx][1]+2, dtype='int32')
                interp = np.interp(yy, [pair[min_idx][1],pair[1-min_idx][1]], [pair[min_idx][0],pair[1-min_idx][0]])
                for dummy in zip(interp,yy):
                    added.append([int(dummy[0]),int(dummy[1])])
                

    return [list(x) for x in set(tuple(x) for x in added+points)]

def load_scan(input_path, output_path):
    files = os.listdir(input_path)
    case_num = len(files)
    #print(case_num)
    height = 512
    width = 512
    for epoch in range(case_num):
        #random.shuffle(files)
        folder = files[epoch]
        path = input_path + folder
        slices = []
        for s in os.listdir(path):
            suf = s[0:2] # =>文件名,文件后缀
            if suf == 'CT':
                slices.append(dicom.read_file(path + '/' + s))
            elif suf == 'RS':
#                continue
                #slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
                #if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
                #    sec_num = 2;
                #    while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
                #        sec_num = sec_num+1;
                #    slice_num = int(len(slices) / sec_num)
                #    slices.sort(key = lambda x:float(x.InstanceNumber))
                #    slices = slices[0:slice_num]
                #    slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
                #try:
                #    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
                #except:
                #    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
                #for single in slices:
                #    single.SliceThickness = slice_thickness
                #first_ID_pixels, spacing = get_pixels_hu(slices)
                #print(first_ID_pixels.shape)
                print(path + '/' + s)
                location = dicom.read_file(path + '/' + s)
                #print(location.dir())
                #print(location.ROIContourSequence[1].dir())
                #print(location.RTROIObservationsSequence)
                #numberOfContours = len(location.ROIContourSequence);
                #print(numberOfContours)
                #print(location.StructureSetROISequence[1].dir())
                for index in range(len(location.StructureSetROISequence)):
                    #print(location.StructureSetROISequence[index - 1].ROIName[0:3])
                    if location.StructureSetROISequence[index].ROIName[0:3] == 'GTV':
                        #print(location.StructureSetROISequence[index - 1].ROINumber)
                        imgs = []
                        masks = []
                        predict = []
                        masks_contour = []
                        predict_contour = []
                        for index_contour in range(len(location.ROIContourSequence[index].ContourSequence)):
                        #for index_contour in range(1):
                            rfContent = location.ROIContourSequence[index].ContourSequence[index_contour]
                            sliceID = []
                            sliceID.append(dicom.read_file(path + '/CT.' + rfContent.ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm'))
                            dcmOrigin = sliceID[0].ImagePositionPatient
                            dcmSpacing = sliceID[0].PixelSpacing
                            numberOfPoints = rfContent.NumberOfContourPoints
                            #print(first_ID_pixels.shape)
                            conData = np.zeros((numberOfPoints,3))
                            pointData = np.zeros((numberOfPoints,2))
                            predictData = np.zeros((numberOfPoints,2))
                            test_aa = []
                            test_bb = []
                            for jj in range(len(conData)):
                                ii = (jj)*3
                                conData[jj][0] = rfContent.ContourData[ii]
                                conData[jj][1] = rfContent.ContourData[ii+1]
                                conData[jj][2] = rfContent.ContourData[ii+2]
                                #test_aa.append([round( (conData[jj][0] - dcmOrigin[0])/dcmSpacing[0] ),round( (conData[jj][1] - dcmOrigin[1])/dcmSpacing[1] )])
                                pointData[jj][0] = round( (conData[jj][0] - dcmOrigin[0])/dcmSpacing[0] )
                                pointData[jj][1] = round( (conData[jj][1] - dcmOrigin[1])/dcmSpacing[1] )
                                predictData[jj][0] = round( (conData[jj][0] - dcmOrigin[0])/dcmSpacing[0] ) + random.randint(5, 15)
                                predictData[jj][1] = round( (conData[jj][1] - dcmOrigin[1])/dcmSpacing[1] ) + random.randint(-5, 5)
                                test_aa.append([pointData[jj][0],pointData[jj][1]])
                                test_bb.append([predictData[jj][0],predictData[jj][1]])
                            #z_voxel = int(round((conData[0][2] - dcmOrigin[2]) / slices[0].SliceThickness))
                            test_aa.append(test_aa[0])
                            temp_contour = interplote(test_aa)
                            temp_contour = np.rint(temp_contour)

                            mask = np.zeros([width,height])
                            #mask[z_voxel,temp_contour[:,1],temp_contour[:,0]] = 1
                            for i in range(len(temp_contour)):
                                mask[int(temp_contour[i][1])][int(temp_contour[i][0])] = 1 # mind the dimension matching
                            seg = ndimage.binary_fill_holes(mask[:,:]) # fill the inside of the contour

                            test_bb.append(test_bb[0])
                            predict_temp_contour = interplote(test_bb)
                            predict_temp_contour = np.rint(predict_temp_contour)

                            predict_mask = np.zeros([width,height])
                            #mask[z_voxel,temp_contour[:,1],temp_contour[:,0]] = 1
                            for i in range(len(predict_temp_contour)):
                                predict_mask[int(predict_temp_contour[i][1])][int(predict_temp_contour[i][0])] = 1 # mind the dimension matching
                            predict_seg = ndimage.binary_fill_holes(predict_mask[:,:]) # fill the inside of the contour
                            #pointData[jj+1][0] = pointData[0][0]
                            #pointData[jj+1][1] = pointData[0][1]
                            #print([x[0] for x in conData])
                            # return sliceID
                            #x = np.zeros(1,512)
                            #y = x
                            #for i in range(512):
                            #    x[i-1] = i
                            #    y[i-1] = list(range(1,512))
                            #in = inpolygon(x,y,pointData(:,1)', pointData(:,2)');
                            #mask = interplote(width, height, [x[0] for x in pointData], [x[1] for x in pointData])
                            #pointData.append(pointData[0])
                            #temp_contour = interplote(pointData)
                            #temp_contour = np.array(temp_contour)
                            #mask[temp_contour[:,0],temp_contour[:,1]] = 1 # mind the dimension matching
                            masks.append(seg) # fill the inside of the contour
                            predict.append(predict_seg) 

                            first_ID_pixels, spacing = get_pixels_hu(sliceID)
                            imgs.append(first_ID_pixels[0])
                            masks_contour.append(pointData)
                            predict_contour.append(predictData)
                            # Show some slice in the middle
                            #plt.imshow(first_ID_pixels[0], cmap=plt.cm.gray)
                            #plt.plot([x[0] for x in pointData], [x[1] for x in pointData], color='red', linewidth=1)
                            #plt.show()
                        np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (epoch, index)),imgs)
                        np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (epoch, index)),masks)
                        np.save(os.path.join(output_path,"predict_%04d_%04d.npy" % (epoch, index)),predict)
                        np.save(os.path.join(output_path,"maskContour_%04d_%04d.npy" % (epoch, index)),masks_contour)
                        np.save(os.path.join(output_path,"predictContour_%04d_%04d.npy" % (epoch, index)),predict_contour)

    #return case_num

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)

############
#
# Getting list of image files
#luna_path = "/media/jiangxin/data/LUNA2016/"
#luna_subset_path = luna_path+"subset8/"
#output_path = "/media/jiangxin/data/LUNA2016/test/"
#file_list=glob(luna_subset_path+"*.mhd")


#####################
#
# Helper function to get rows in data frame associated 
# with each file
#def get_filename(file_list, case):
#    for f in file_list:
#        if case in f:
#            return(f)
#
# The locations of the nodes
#df_node = pd.read_csv(luna_path+"annotations.csv")
#df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
#df_node = df_node.dropna()



#####
#
# Looping over the image files
#
#for fcount, img_file in enumerate(tqdm(file_list)):
#    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
#    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
#        # load the data once
#        itk_img = sitk.ReadImage(img_file) 
#        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
#        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
#        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
#        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
#        # go through all nodes (why just the biggest?)
#        for node_idx, cur_row in mini_df.iterrows():       
#            node_x = cur_row["coordX"]
#            node_y = cur_row["coordY"]
#            node_z = cur_row["coordZ"]
#            diam = cur_row["diameter_mm"]
#            # just keep 3 slices
#            imgs = np.ndarray([3,height,width],dtype=np.float32)
#            masks = np.ndarray([3,height,width],dtype=np.uint8)
#            center = np.array([node_x, node_y, node_z])   # nodule center
#            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
#            for i, i_z in enumerate(np.arange(int(v_center[2])-1,
#                             int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
#                mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
#                                 width, height, spacing, origin)
#                masks[i] = mask
#                imgs[i] = img_array[i_z]
#            np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
#            np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)



input_path = "/media/jiangxin/data/npc/source/"
output_path = "/media/jiangxin/data/npc/training/"
xc = np.array([219.5,284.8,340.8,363.5,342.2,308.8,236.8,214.2])
yc = np.array([284.8,220.8,203.5,252.8,328.8,386.2,382.2,328.8])
#test_aa = []
#for point in range(8):
#    test_aa.append([xc[point],yc[point]])
#test_aa.append(test_aa[0])
#temp_contour = interplote(test_aa)
#temp_contour = np.rint(temp_contour)
#print(temp_contour)
#mask = np.zeros([512,512])
#print(mask.shape)
#for i in range(len(temp_contour)):
    #print(temp_contour[i][0])
#    mask[int(temp_contour[i][1])][int(temp_contour[i][0])] = 1 # mind the dimension matching
#seg = ndimage.binary_fill_holes(mask[:,:]) # fill the inside of the contour
#mask = interplote(512, 512, xc, yc)
#print(mask)
#fig,ax = plt.subplots(2,2,figsize=[8,8])
#ax[0,0].imshow(np.zeros([512,512]), cmap=plt.cm.gray)
#ax[0,0].plot(xc, yc, color='red', linewidth=1)
#ax[0,1].imshow(seg,cmap='gray')
#plt.imshow(seg, cmap=plt.cm.gray)
#plt.show()
#load_scan(input_path, output_path)
#exit()
file_list=glob(output_path+"images_*.npy")
j=0
for fname in file_list:
    print ("working on file ", fname)
    imgs = np.load(fname)
    masks = np.load(fname.replace("images","masks"))
    predict = np.load(fname.replace("images","predict"))
    maskContour = np.load(fname.replace("images","maskContour"))
    predictContour = np.load(fname.replace("images","predictContour"))
#imgs = np.load(output_path+'images_0000_0015.npy')
#masks = np.load(output_path+'masks_0000_0015.npy')
#predict = np.load(output_path+'predict_0000_0015.npy')
#maskContour = np.load(output_path+'maskContour_0000_0015.npy')
#predictContour = np.load(output_path+'predictContour_0000_0015.npy')
#print(masks.shape)
#exit()
    i=0
#for i in range(len(imgs)):
    print("image %d" % i)
#    plt.imshow(imgs[i], cmap=plt.cm.gray)
#    plt.plot([x[0] for x in masks[i]], [x[1] for x in masks[i]], color='red', linewidth=1)
#    plt.show()
    fig,ax = plt.subplots(1,3,figsize=[8,8])
    ax[0].imshow(imgs[i],cmap='gray')
    ax[0].plot([x[0] for x in maskContour[i]], [x[1] for x in maskContour[i]], color='red', linewidth=0.5)
    ax[0].plot([x[0] for x in predictContour[i]], [x[1] for x in predictContour[i]], color='blue', linewidth=1)
    #ax[0,1].imshow(masks[i],cmap='gray')
    ax[1].imshow(predict[i],cmap='gray')
    ax[2].imshow(imgs[i]*predict[i],cmap='gray')
    j=j+1
    plt.savefig(output_path+'test/'+str(j)+'.png')
    #plt.show()
#    #raw_input("hit enter to cont : ")
