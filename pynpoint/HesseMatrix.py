# -*- coding: utf-8 -*-
__author__ = "Gabriele Cugno"

# Filter out warnings when importing astropy
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=Warning)

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from numpy import math
from scipy.interpolate import interp2d,interp1d
from PynPoint.core import ProcessingModule
from GaboInsertPlanet import InsertFakePlanetModule
from PynPoint import Pypeline
from PynPoint.processing_modules import PSFSubtractionModule
from scipy.optimize import minimize
from astropy.nddata import Cutout2D
from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile
import numdifftools as nd
from scipy.ndimage.filters import gaussian_filter


class ADI_HesseMatrixModule(ProcessingModule):

    def __init__(self,
                 working_place_in,
                 input_place_in,
                 output_place_in,
                 psf_file,
                 image_in_tag,
                 name_in="HesseMatrix",
                 inner_mask=0.1,
                 pca_number=20,
                 rough_pos=(0,0),
                 rough_mag=8.,
                 subpix=2.,
                 ROI_size=10,
                 cutting_psf=30,
                 tolerance = 0.1,
                 conv=1,
                 num_pos = 100):


        super(ADI_HesseMatrixModule, self).__init__(name_in=name_in)

        # Variables
        self.m_inner_mask = inner_mask
        self.m_pca_number = pca_number
        self.m_rough_pos = rough_pos
        self.m_rough_mag = rough_mag
        self.m_subpix = subpix
        self.m_ROI_size = ROI_size
        self.m_psf_file = psf_file
        self.m_image_in_tag=image_in_tag
        self.m_working_place_in=working_place_in
        self.m_input_place_in=input_place_in
        self.m_output_place_in=output_place_in
        self.m_cutting_psf = cutting_psf
        self.m_tolerance = tolerance
        self.m_conv = conv
        self.m_num_pos = num_pos

        # Input port
        self.m_image_in_port = self.add_input_port(image_in_tag)


    def run(self):
            
        def best_position(x):

            # input parameter: vector x = (pos_x, pos_y, mag_contrast)
            # output parameter: determinant of the curvature in the planet position
            # Define Directories
            working_place_in = self.m_working_place_in
            input_place_in = self.m_input_place_in
            output_place_in = self.m_output_place_in

            pipeline = Pypeline(working_place_in,
                                input_place_in,
                                output_place_in)

            # Insert negative fake planets
            neg_fake_planet = InsertFakePlanetModule(name_in="insert_negative_Fake_planet",
                                                raw_data_in_tag=self.m_image_in_tag,
                                                fake_planet_out_tag="fake_planets",
                                                psf_fits=self.unsat,
                                                cutting_psf=self.m_cutting_psf,
                                                fake_planet_pos=(x[0], x[1]),
                                                mag=x[2],
                                                subpix_precision=self.m_subpix,
                                                negative_flux=True)

            # Calculate Residuals
            psf_subtraction = PSFSubtractionModule(pca_number=self.m_pca_number,
                                                   cent_size=self.m_inner_mask,
                                                   name_in="PSF_subtraction",
                                                   images_in_tag="fake_planets",
                                                   reference_in_tag="fake_planets",
                                                   res_arr_out_tag="res_arr_astroPhoto",
                                                   res_arr_rot_out_tag="res_arr_rot_astroPhoto",
                                                   res_mean_tag="res_mean_astroPhoto",
                                                   res_median_tag="res_median_astroPhoto",
                                                   res_var_tag="res_var_astroPhoto",
                                                   res_rot_mean_clip_tag="res_rot_mean_clip_astroPhoto")
            

            pipeline.add_module(neg_fake_planet)
            pipeline.add_module(psf_subtraction)
            pipeline.run()

            # Load processed image
            res_in_port = self.add_input_port("res_mean_astroPhoto")
            im_res = res_in_port.get_all()
            res_in_port.close_port()
            
            #guassian filter on the entire image
            im_res_smooth = gaussian_filter(im_res,1)
            
            # cut image at position
            im_res_cut = Cutout2D(data=im_res_smooth,
                                  position=self.m_rough_pos,
                                  size=(self.m_ROI_size, self.m_ROI_size)).data
                                  
            #define determinant
            sum_det=0.
            
            #interpolate the surface of the figure
            xx = np.arange(self.m_ROI_size)
            yy = np.arange(self.m_ROI_size)
            f = interp2d(xx,yy,im_res_cut, kind='cubic')
            
            # calculate the sum of determinants of the figure
            for i in np.linspace(0,self.m_ROI_size,sef.m_num_pos):
                for j in np.linspace(0,self.m_ROI_size,100,sef.m_num_pos):
                    H = nd.Hessian(lambda z:f(z[0],z[1]))
                    det = np.linalg.det(H(np.array([i,j])))
                    sum_det += np.abs(det)
        
            # print results
            print "Hessian determinant in ROI: " +str(sum_det)
            return np.abs(sum_det)
                    

        
        
        
        # Import unsaturated PSF
        tmp_data = fits.open(str(self.m_psf_file))[0]
        self.unsat = tmp_data.data
        

        # Perform minimization
        res = minimize(fun=best_position,
                       x0=[self.m_rough_pos[0], self.m_rough_pos[1], self.m_rough_mag],
                       method="Nelder-Mead",
                       tol=self.m_tolerance)
        pos_planet1 = np.round(res.x[0]*self.m_subpix)/self.m_subpix
        pos_planet2 = np.round(res.x[1]*self.m_subpix)/self.m_subpix
        mag_planet = res.x[2]

        print "Best position: (%s,%s)" % (pos_planet1,pos_planet2)
        
        print "Best magnitude: %f" % mag_planet
        


        pipeline = Pypeline(working_place_in,
                            input_place_in,
                            output_place_in)

        # Insert negative fake planets
        neg_fake_planet = InsertFakePlanetModule(name_in="insert_negative_Fake_planet_final",
                                            raw_data_in_tag=self.m_image_in_tag,
                                            fake_planet_out_tag="fake_planets",
                                            psf_fits=self.unsat,
                                            fake_planet_pos=(res.x[0], res.x[1]),
                                            mag=res.x[2],
                                            cutting_psf=self.m_cutting_psf,
                                            subpix_precision=self.m_subpix,
                                            negative_flux=True)

        # Calculate Residuals
        psf_subtraction = PSFSubtractionModule(pca_number=self.m_pca_number,
                                               cent_size=self.m_inner_mask,
                                               name_in="PSF_subtraction",
                                               images_in_tag="fake_planets",
                                               reference_in_tag="fake_planets",
                                               res_arr_out_tag="res_arr_astroPhoto",
                                               res_arr_rot_out_tag="res_arr_rot_astroPhoto",
                                               res_mean_tag="res_mean_astroPhoto_final",
                                               res_median_tag="res_median_astroPhoto",
                                               res_var_tag="res_var_astroPhoto",
                                               res_rot_mean_clip_tag="res_rot_mean_clip_astroPhoto")

        write_residuals = WriteAsSingleFitsFile(name_in="Fits_writing_final subtracted_residuals",
                                                     file_name="residuals_no_planets_HesseMatrix.fits",
                                                     data_tag="res_mean_astroPhoto_final")
    



        pipeline.add_module(neg_fake_planet)
        pipeline.add_module(psf_subtraction)
        pipeline.add_module(write_residuals)
        pipeline.run()
        
        
                       
        
        self.m_pix2mas = self.m_image_in_port.get_attribute("ESO INS PIXSCALE") * 1000.
        print self.m_pix2mas
        
        im_init = self.m_image_in_port.get_all()
        
        # size of the image:
        image_size = len(im_init[0])
        print image_size
        pos = np.array([res.x[0],res.x[1]])
        
        # Calculate radial distance and error:
        pos_err_x = np.sqrt((1. / self.m_subpix) ** 2 + (self.m_tolerance) ** 2)
        pos_err_y = np.sqrt((1. / self.m_subpix) ** 2 + (self.m_tolerance) ** 2)
        print pos_err_y
        
        rad_dist = (self.m_pix2mas * np.sqrt((pos[0] - image_size / 2.) ** 2 + (pos[1] - image_size / 2.) ** 2))  ###This is in mas
        rad_dist_err = (np.sqrt((self.m_pix2mas ** 2 * 1. / rad_dist * (pos[0] - image_size / 2.) * pos_err_x) ** 2 +
                            (self.m_pix2mas ** 2 * 1. / rad_dist * (pos[1] - image_size / 2.) * pos_err_y) ** 2))
        
        # Calculate Position Angle and error:
        # See which quadrant the planet is in, and give accordingly the correct additional factor for the arctan calculation:
        if pos[0] > image_size / 2. and pos[1] > image_size / 2.:
            arctan_fac = 270.
            pos_angle = (90-np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.)))))+arctan_fac
        if pos[0] < image_size / 2. and pos[1] < image_size / 2.:
            arctan_fac = 90.
            pos_angle = (90-np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.)))))+arctan_fac
        if pos[0] > image_size / 2. and pos[1] < image_size / 2.:
            arctan_fac = 180.
            pos_angle =np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.))))+arctan_fac
        if pos[0] < image_size / 2. and pos[1] > image_size / 2.:
            arctan_fac = 0.
            pos_angle =np.degrees(np.arctan((np.abs(pos[0]-image_size/2.))/(np.abs(pos[1]-image_size/2.))))+arctan_fac
        
        pos_angle_err = (180./np.pi)*np.sqrt((pos_err_y*((pos[0]-image_size/2.)/((pos[0]-image_size/2.)**2+(pos[1]-image_size/2.)**2)))**2+(pos_err_x*((pos[1]-image_size/2.)/((pos[0]-image_size/2.)**2+(pos[1]-image_size/2.)**2)))**2)


        print '\n########################################'
        print 'POSITION = (' + str(np.round(res.x[0]*self.m_subpix)/self.m_subpix)+', '+ str(np.round(res.x[1]*self.m_subpix)/self.m_subpix) + ' )\n'
        print 'ASTROMETRY:\nRadial distance = ' + str(rad_dist*10**(-3)) + ' +- ' + str(rad_dist_err*10**(-3)) + ' [arcsec]\n' + 'P.A. = ' + str(pos_angle) + ' +- ' + str(pos_angle_err) + ' [deg]\n'
        #print 'Linear FPF interpolation: mag = ' + str(res.x[2]) + ' + ' + str(mag_interp_err_r) + ' ; - ' + str(mag_interp_err_l)
        print 'mag contrast = %: mag = ' + str(res.x[2])
        print '\n########################################'


        # write the results
        names = ['Pos x', 'Pos y', 'Rad dist', 'dist err', 'PA', 'PA err', 'mag contrast']
    
        results = Table([[pos_planet1], [pos_planet2], [rad_dist * (10 ** (-3))], [rad_dist_err * (10 ** (-3))], [pos_angle], [pos_angle_err], [mag_planet]], names=names)
        
        results_name = output_place_in + '/Results_HesseMatrix.txt'
        
        results.write(results_name, format='ascii.basic', delimiter='\t')
        print 'The result has been saved in ' + str(results_name) + '\n'



        res_in_port_f = self.add_input_port("res_mean_astroPhoto_final")
        im_res_f = res_in_port_f.get_all()
        res_in_port_f.close_port()
        
        im_res_cut_f_before = Cutout2D(data=im_res_f,
                                      position=np.array([pos_planet1,pos_planet2]),
                                      size=(self.m_ROI_size, self.m_ROI_size)).data

        vmax = np.max(im_res_cut_f_before)
        vmin = np.min(im_res_cut_f_before)

        #guassian filter on the entire image
        im_res_smooth_f = gaussian_filter(im_res_f,1)
    
        # cut image at position
        im_res_cut_f_after = Cutout2D(data=im_res_smooth_f,
                          position=np.array([pos_planet1,pos_planet2]),
                          size=(self.m_ROI_size, self.m_ROI_size)).data

        surface, (before, after) = plt.subplots(1,2, figsize=(12,6))
        before.imshow(im_res_cut_f_before, vmax=vmax, vmin=vmin,cmap='autumn')
        for i in np.arange(self.m_ROI_size):
            for j in np.arange(self.m_ROI_size):
                before.text(i-0.4,j, '%s'%float('%.2g'%(im_res_cut_f_before[j,i]*1e7)))
        before.set_title('Before Convolution')



        after.imshow(im_res_cut_f_after, vmax=vmax, vmin=vmin,cmap='autumn')
        for i in np.arange(self.m_ROI_size):
            for j in np.arange(self.m_ROI_size):
                after.text(i-0.4,j, '%s'%float('%.2g'%(im_res_cut_f_after[j,i]*1e7)))
        after.set_title('After Convolution')
        
        #surface.show()
        surface.savefig(output_place_in+'/pixelMap.png')


