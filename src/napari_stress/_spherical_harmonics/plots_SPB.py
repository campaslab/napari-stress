#from numpy  import * # imported below at np
from scipy  import *
import mpmath
import cmath
from scipy.special import sph_harm
from scipy.spatial import Delaunay
from scipy.sparse import csgraph

# For Plotting: (import once for all fns below)
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
from matplotlib import cm

matplotlib.use('agg');  # This allows for headless plotting (changes plt.show() so non-modal...)

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
#mpl.rcParams['lines.linewidth'] = .25


#import mayavi.mlab as ml #NOT Supported in python 3.6 yet
import numpy as np
import scipy as sp

#for pickling:
#import cPickle as pkl # BJG: py2pt7 version
import pickle as pkl
import os, sys

#for timestamp:
import datetime as dt

from charts_SPB import *
from sph_func_SPB import *
import euc_k_form_SPB as euc_kf
import lebedev_write_SPB as leb_wr
import lbdv_info_SPB as lbdv_i

import itertools # for sorting

# For new meshing: 
import vtk

#Create Plot of Function
def Plot_Func(func, Theta_Res, Phi_Res, Theta_Min, Theta_Max, Phi_Min, Phi_Max, Plot_Title):

	Theta_Test_Vals = linspace(Theta_Min, Theta_Max, Theta_Res, endpoint = False)
	Phi_Test_Vals = linspace(Phi_Min, Phi_Max, Phi_Res)
	#linspace(pi/(Phi_Res+1), pi, Phi_Res, endpoint = False)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = Theta_Test_Vals #np.arange(-5, 5, 0.25)
	Y = Phi_Test_Vals #np.arange(-5, 5, 0.25)
	X, Y = np.meshgrid(X, Y)
	#R = np.sqrt(X**2 + Y**2)

	Z = zeros([Phi_Res, Theta_Res])

	for x in range(Theta_Res):
		for y in range(Phi_Res):
			Z[y][x] = func(Theta_Test_Vals[x], Phi_Test_Vals[y])


	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_zlim(amin(Z), amax(Z))

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf, shrink=0.5, aspect=5)

	ax.set_xlabel('Theta')
	ax.set_ylabel('Phi')
	ax.set_zlabel("func")
	fig.suptitle(Plot_Title, fontsize=20)

	
	plt.show()


#Create Appropriate Error Plots
def Error_Plots(func1, func2, func3, Theta_Res, Phi_Res, Theta_Min, Theta_Max, Phi_Min, Phi_Max, Title1, Title2, Title3):

	Plot_Func(lambda theta, phi: func1(theta, phi)-func2(theta, phi), Theta_Res, Phi_Res, Theta_Min, Theta_Max, Phi_Min, Phi_Max, "("+str(Title1)+") - ("+str(Title2)+")")

	Plot_Func(lambda theta, phi: func1(theta, phi)-func3(theta, phi), Theta_Res, Phi_Res, Theta_Min, Theta_Max, Phi_Min, Phi_Max, "("+str(Title1)+") - ("+str(Title3)+")")

	Plot_Func(lambda theta, phi: func2(theta, phi)-func3(theta, phi), Theta_Res, Phi_Res, Theta_Min, Theta_Max, Phi_Min, Phi_Max, "("+str(Title2)+") - ("+str(Title3)+")")
	


# Shows Quad Pts used for collocation in Least Sqaures approach
def Show_Quad_Pts_Used(fixed_ls_colloc_pts, lbdv):
	
	#x = Fixed_LS_Colloc_Pts[:,0]
	#y = Fixed_LS_Colloc_Pts[:,1]
	#colors = np.ones(generated_fixed_ls_colloc_pts)
	#area = np.pi*np.ones(generated_fixed_ls_colloc_pts)

	#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
	#plt.show()

	x1 = fixed_ls_colloc_pts[:,0]
	y1 = fixed_ls_colloc_pts[:,1]

	x2 =  lbdv.Lbdv_Sph_Pts_Quad[:, 0]
	y2 =  lbdv.Lbdv_Sph_Pts_Quad[:, 1]

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.scatter(x1, y1, s=10, c='b', marker="s", label='used')
	ax1.scatter(x2,y2, s=5, c='r', marker="o", label='not used')
	plt.legend(loc='upper left');
	plt.show()



#Plots Errors of SPH functions vs exact function at the quad pts
def Plot_Errors_At_Quad_Pts(Plot_Title, vals_at_quad_pts, exact_func, lbdv):

	#Theta_Test_Vals = linspace(Theta_Min, Theta_Max, Theta_Res, endpoint = False)
	#Phi_Test_Vals = linspace(Phi_Min, Phi_Max, Phi_Res)
	#linspace(pi/(Phi_Res+1), pi, Phi_Res, endpoint = False)

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	#X = Theta_Test_Vals #np.arange(-5, 5, 0.25)
	#Y = Phi_Test_Vals #np.arange(-5, 5, 0.25)
	#X, Y = np.meshgrid(X, Y)
	#R = np.sqrt(X**2 + Y**2)

	X, Y, W = hsplit(lbdv.Lbdv_Sph_Pts_Quad, 3)	
	Z = zeros(shape(X))
	
	for q in range(lbdv.lbdv_quad_pts):		

		theta = X[q]
		phi = Y[q]

		#Z[q] = lbdv.Eval_SPH_Der_Phi_At_Quad_Pts(0,1, q) - exact_func(theta, phi) 
		Z[q] = vals_at_quad_pts[q] - exact_func(theta, phi) 

	ax.scatter(X, Y, Z, c='r', marker='o')

	ax.set_zlim(amin(Z), amax(Z))

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	ax.set_xlabel('Theta')
	ax.set_ylabel('Phi')
	ax.set_zlabel("Error at Quad Pts (exact - sph)")
	
	fig.suptitle(Plot_Title, fontsize=20)

	
	plt.show()

	

#Used to plot results of Convergence Loop, in log-log and semi-log plots:
def Plot_Conv_Results(conv_title, q_vals, p_refs, errors):	

	plt.subplots_adjust(hspace=0.4)	

	plt.suptitle(str(conv_title)+" Convergence", fontsize=12)		

	## Log(Error) vs Quad Pts:
	
	#fit_exp_Q = polyfit(q_vals, log(errors), 1)
	#print("exp fit for Q = "+str(fit_exp_Q))
	
	#fit_exp_Q_fn = poly1d(fit_exp_Q)
	#print("fit vals ="+str(fit_exp_Q_fn(q_vals)))

	plt.subplot(221)
	plt.semilogy(q_vals, errors + np.finfo(float).eps/2, '-o')
	
	#plt.plot(q_vals, fit_exp_Q_fn(q_vals), 'r-')
	
	plt.title('log(Error) vs Quad Pts')
	plt.grid(True)

	plt.subplot(222)
	plt.loglog(q_vals, errors + np.finfo(float).eps/2, '-o')
	plt.grid(True)
	plt.title('log(Error) vs log(Quad Pts)')
		
	plt.subplot(223)
	plt.semilogy(p_refs, errors + np.finfo(float).eps/2, '-o')
	plt.grid(True)
	plt.title('log(Error) vs P-Refinement')

	plt.subplot(224)
	plt.loglog(p_refs, errors + np.finfo(float).eps/2, '-o')
	plt.grid(True)
	plt.title('log(Error) vs log(P-Refinement)')

	plt.show()



#Used to plot results of Convergence Loop, in log-log and semi-log plots, if P_Ref is the same, but R_0 Changes:
def Plot_Conv_Results_Const_P_Ref(conv_title, q_val, p_ref, r0_vals, errors):	

	fig = plt.figure()

	plt.subplots_adjust(hspace=0.4)	

	plt.suptitle(str(conv_title)+" Convergence for P = "+str(p_ref)+"and Q = "+str(q_val), fontsize=12)		

	plt.subplot(211)
	plt.semilogy(r0_vals, errors + np.finfo(float).eps/2, '-o')

	
	plt.title('log(Error) vs R_0')
	plt.grid(True)

	plt.subplot(212)
	plt.loglog(r0_vals, errors + np.finfo(float).eps/2, '-o')
	plt.grid(True)
	plt.title('log(Error) vs log(R_0)')
	
	
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = str(conv_title)+"_Convergence"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')

	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = conv_title
	Data_Dictionary['Quad_Refinement'] = q_val
	Data_Dictionary['P_Refinement'] = p_ref
	Data_Dictionary['Relative_Errors'] = errors
	Data_Dictionary['R_0_Values'] = r0_vals
	
	Dictionary_Name = str(conv_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)
	
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)

#Used to plot UNSCALED DATA, if P_Ref is the same, but R_0 Changes:
def Plot_Data_Const_P_Ref(conv_title, q_val, p_ref, r0_vals, data):	

	fig = plt.figure()
	
	plt.subplots_adjust(hspace=0.4)	
	
	plt.suptitle(str(conv_title)+" Data for P = "+str(p_ref)+"and Q = "+str(q_val), fontsize=12)		
	
	plt.subplot(111)
	plt.plot(r0_vals, data, '-o')
	
	
	plt.title('Data vs R_0')
	plt.grid(True)
	
	
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = str(conv_title)+"_Data"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)
	
	fig.savefig(fig_path, format='pdf')

	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = conv_title
	Data_Dictionary['Quad_Refinement'] = q_val
	Data_Dictionary['P_Refinement'] = p_ref
	Data_Dictionary['Data'] = data
	Data_Dictionary['R_0_Values'] = r0_vals
	
	Dictionary_Name = str(conv_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)
	
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)

#Used to plot results with variotion of parameter of Convergence Loop, in log-log and semi-log plots:
def Plot_Param_Conv_Results(conv_title, q_vals, p_refs, errors, Param_Values, Param_Name):	
	
	fig = plt.figure()

	plt.subplots_adjust(hspace=0.4)	

	plt.suptitle(str(conv_title)+" Convergence", fontsize=12)		

	## Log(Error) vs Quad Pts:
	
	#fit_exp_Q = polyfit(q_vals, log(errors), 1)
	#print("exp fit for Q = "+str(fit_exp_Q))
	
	#fit_exp_Q_fn = poly1d(fit_exp_Q)
	#print("fit vals ="+str(fit_exp_Q_fn(q_vals)))

	labels = array([]) #label by param value	

	# plot for each parameter value
	for param_iter in range(len(Param_Values)):
		Param_val = Param_Values[param_iter]

		errors_param  = []		

		if(len(Param_Values) > 1):
			errors_param = errors[:, param_iter]
		else:
			errors_param = errors

		plt.subplot(221)
		plt.semilogy(q_vals, errors_param + np.finfo(float).eps/2, '-o', label= Param_Name+" = "+str(Param_val))
	
		#plt.plot(q_vals, fit_exp_Q_fn(q_vals), 'r-')
	
		plt.title('log(Error) vs Quad Pts')
		plt.grid(True)

		plt.subplot(222)
		plt.loglog(q_vals, errors_param + np.finfo(float).eps/2, '-o', label=Param_Name+" = "+str(Param_val))
		plt.grid(True)
		plt.title('log(Error) vs log(Quad Pts)')
		
		plt.subplot(223)
		plt.semilogy(p_refs, errors_param + np.finfo(float).eps/2, '-o', label= Param_Name+" = "+str(Param_val))
		plt.grid(True)
		plt.title('log(Error) vs P-Refinement')

		plt.subplot(224)
		plt.loglog(p_refs, errors_param + np.finfo(float).eps/2, '-o', label= Param_Name+" = "+str(Param_val))
		plt.grid(True)
		plt.title('log(Error) vs log(P-Refinement)')

		labels = concatenate(( labels, [Param_Name+" = "+str(Param_val)] ))
	
	plt.legend(labels, loc='upper right') #show legend
	plt.legend(bbox_to_anchor=(1.25, 1.25))
	
	#mng = plt.get_current_fig_manager() #We want to save maximized image, need wx backend
	#mng.window.showMaximized()

	#plt.show() #plotting stops the code

	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = str(conv_title)+"_Convergence"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')
	

	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = conv_title
	Data_Dictionary['Quad_Refinement'] = q_vals
	Data_Dictionary['P_Refinement'] = p_refs
	Data_Dictionary['Relative_Errors'] = errors
	Data_Dictionary['Parameter_Values'] = Param_Values
	Data_Dictionary['Parameter_Name'] = Param_Name
	
	Dictionary_Name = str(conv_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)
	
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)

#Used to plot results with variotion of parameter of Convergence Loop, in log-log and semi-log plots:
def Plot_Histograms(conv_title, errors):	
	
	fig = plt.figure()

	plt.suptitle(str(conv_title)+" Error Histogram", fontsize=12)		
	plt.hist(errors)

	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = str(conv_title)+"_Error_Hist"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


#Used to plot mean curvature pdf and cdf for Droplet codes:
def Plot_Histograms(plot_title, X_prob, hist_dist, sorted_distr, delta, min_curv_excl, max_curv_excl, Input_Dir=[]):	
	
	fig = plt.figure()

	plt.suptitle(str(plot_title), fontsize=12)		

	plt.title("CDF of "+str(plot_title))

	num_data_pts = len(sorted_distr)
	weights_pdf = np.ones_like(sorted_distr)/float(num_data_pts)
	num_bins = num_data_pts
	if(num_data_pts > 40.):
		num_bins = int(num_data_pts/20)

	plt.subplot(221)
	plt.hist(sorted_distr, weights=weights_pdf, bins=num_bins, label = 'PDF', color = "purple") # can't use bins = 'auto' with weights
	plt.axvline(x=min_curv_excl, label= "min val excl = "+str(min_curv_excl)+", for delta = "+str(delta), c='red')			
	plt.axvline(x=max_curv_excl, label= "max val excl = "+str(max_curv_excl)+", for delta = "+str(delta), c='blue')
	plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.1), prop={'size': 6})
	#plt.plot(X_prob, hist_dist.ppf(X_prob), label='PPF')

	plt.subplot(222)
	plt.plot(X_prob, hist_dist.cdf(X_prob), label='CDF', c='green')
	
	plt.axvline(x=min_curv_excl, label= "min val excl = "+str(min_curv_excl)+", for delta = "+str(delta), c='red')			
	plt.axvline(x=max_curv_excl, label= "max val excl = "+str(max_curv_excl)+", for delta = "+str(delta), c='blue')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), prop={'size': 6})



	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	if(Input_Dir != []):
		Picked_Image_DIR = Input_Dir

	fig_name = str(plot_title)+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


#Used to plot mean curvature pdf ONLY Droplet codes:
def Plot_Histogram_PDF(plot_title, distr_vals, Input_Dir=[]):

	sorted_distr = np.sort(distr_vals)

	median_val = np.median(sorted_distr) # plot mean and median
	mean_val = np.average(sorted_distr)
	
	fig = plt.figure()
	plt.suptitle(str(plot_title), fontsize=12)		
	plt.title("PDF of "+str(plot_title))

	num_data_pts = len(sorted_distr)
	weights_pdf = np.ones_like(sorted_distr)/float(num_data_pts)

	num_bins = num_data_pts
	if(num_data_pts > 10.):
		num_bins = int(num_data_pts/5.)

	plt.hist(sorted_distr, weights=weights_pdf, bins=num_bins, label = 'PDF', color = "purple") # can't use bins = 'auto' with weights
	plt.axvline(x=median_val, label= "median = "+str(median_val), c='red')			
	plt.axvline(x=mean_val, label= "mean = "+str(mean_val), c='blue')
	plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.1), prop={'size': 6})

	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	if(Input_Dir != []):
		Picked_Image_DIR = Input_Dir

	fig_name = str(plot_title)+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


#Used to plot JUST CDF:
def Plot_Histogram_CDF_ONLY(plot_title, X_pts, sorted_normalized_distr, Input_Dir = []):	
	
	fig = plt.figure()

	plt.suptitle(str(plot_title), fontsize=12)		

	plt.title("CDF of "+str(plot_title))

	plt.plot(X_pts, np.cumsum(sorted_normalized_distr), label='CDF', c='green')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), prop={'size': 6})

	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images')
	
	if(Input_Dir != []):
		Picked_Image_DIR = Input_Dir

	fig_name = str(plot_title)+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


#Plots errors as function of BOTH (p_deg, q_deg)
def Plot_Multi_Conv_Results(multi_conv_title, q_vals, q_degs, p_degs, errors):	
 
	p_refs = np.square(p_degs + 1) #Number of basis elements

	labels = array([]) #label convergence by basis degree

	#plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
		
	plt.subplot(221)
	plt.grid(True)	

	for p in range(len(p_degs)):
		
		clr = plt.cm.YlOrBr(np.float64(p+1)/(len(p_degs)+1)) #different color for each degree plot
		plt.loglog(q_degs, errors[p,:], '-o', color = clr) #plot for different quads used
		str_lbl_p = str(p_degs[p]) #label by degree
		labels = concatenate(( labels, ["p_deg ="+str_lbl_p] )) #add to labels


	plt.legend(labels, loc='upper right') #show legend
	plt.title('log(Error) vs log(Quad Degree)')	
	
	

	plt.subplot(222)	
	plt.grid(True)


	for p in range(len(p_degs)):
		
		clr = plt.cm.YlOrBr(np.float64(p+1)/(len(p_degs)+1)) #different color for each degree plot
		plt.semilogy(q_degs, errors[p,:], '-o', color = clr) #plot for different quads used
		str_lbl_p = str(p_degs[p]) #label by degree
		labels = concatenate(( labels, ["p_deg ="+str_lbl_p] )) #add to labels
	
	plt.legend(labels, loc='upper right') #show legend
	plt.title('log(Error) vs Quad Degree')	

	plt.subplot(223)
	plt.grid(True)	

	for p in range(len(p_degs)):
		
		clr = plt.cm.YlOrBr(np.float64(p+1)/(len(p_degs)+1)) #different color for each degree plot
		plt.loglog(q_vals, errors[p,:], '-o', color = clr) #plot for different quads used
		str_lbl_p = str(p_degs[p]) #label by degree
		labels = concatenate(( labels, ["p_deg ="+str_lbl_p] )) #add to labels


	plt.legend(labels, loc='upper right') #show legend
	plt.title('log(Error) vs log(Quad Pts)')	

	plt.subplot(224)	
	plt.grid(True)


	for p in range(len(p_degs)):
		
		clr = plt.cm.YlOrBr(np.float64(p+1)/(len(p_degs)+1)) #different color for each degree plot
		plt.semilogy(q_vals, errors[p,:], '-o', color = clr) #plot for different quads used
		str_lbl_p = str(p_degs[p]) #label by degree
		labels = concatenate(( labels, ["p_deg ="+str_lbl_p] )) #add to labels
	
	plt.legend(labels, loc='upper right') #show legend
	plt.title('log(Error) vs Quad Pts')	

	plt.show()


# Plot 2-param values for fig 2 membrane flow paper:
def plot_mem_flows_fig2(manny_label, param_vals, dep_field_vals, indep_var_vals):
	'''
	param_val = list(delta_vals, beta_val (USE .6), gamma_vals, epsilon_vals) 
	dep_field_vals = (num_r0 (\delta), num_beta, num_gamma, num_eps, 3 {\V\|, T, k_D}, num_quad_pts (Q)
	indep_var_vals = {K, arclen} for each r_0, shape: {num_r0, Q, 2}
	'''

	num_quad_pts_to_plot = indep_var_vals.shape[1] 
	mk_size = 200./num_quad_pts_to_plot	

	delta_vals = param_vals[0]
	num_delta_vals = len(delta_vals)

	beta_val_used = param_vals[1][0] # WE ONLY HAVE ONE 

	gamma_vals = param_vals[2]
	num_gamma_vals = len(gamma_vals)

	epsilon_vals = param_vals[3]
	num_epsilon_vals = len(epsilon_vals)

	num_plots_per_subplot = num_gamma_vals*num_epsilon_vals # plots of params on each subplot
	num_plots = num_delta_vals*3

	fig = plt.figure(figsize=(8.5, 8.5))
	plt.subplots_adjust(hspace=0.4)	
	fig_title = manny_label +"_beta_"+str(beta_val_used).replace(".", "pt")+ "_mem_flow_paper_fig2_"
	#plt.suptitle(fig_title, fontsize=12, y=1.08) # plt.title(figure_title, y=1.08)
		
	plot_num = 1
	labels = array([]) #label by param value
	
	# find maximum V for plot scale, and min and max T for each delta:
	max_V_mag = 0.
	T_min_delta = np.zeros(( num_delta_vals )) 
	T_max_delta = np.zeros(( num_delta_vals ))
	
	for delta_i in range(num_delta_vals):
	
		T_append = np.array([])
		
		for epsilon_i in range(num_epsilon_vals):
			for gamma_i in range(num_gamma_vals):
				V_r0, T_r0, k_D_r0 = np.hsplit( np.squeeze(dep_field_vals[delta_i, 0, gamma_i, epsilon_i, :, :]), 3)
				
				T_append = np.concatenate(( T_append, T_r0.flatten() ))
				
				max_V_r0 =max(V_r0)
				if(max_V_mag < max_V_r0):
					max_V_mag = max_V_r0
						
		T_min_delta[delta_i] = min(T_append)
		T_max_delta[delta_i] = max(T_append)
		print("for delta_i = "+str(delta_i)+", T_min_delta[delta_i] = "+str(T_min_delta[delta_i] )+", T_max_delta[delta_i] = "+str(T_max_delta[delta_i] )+", T_append = "+str(np.sort(T_append)) )

	for delta_i in range(num_delta_vals):

		delta_val_i = param_vals[0][delta_i]
		arclens_used = indep_var_vals[delta_i, :, 1]

		inds_arclens_used_sort = arclens_used.argsort() # we need to sort these to connect plot lines
		arclens_used_sorted = np.sort(arclens_used)

		color_i = 0

		for epsilon_i in range(num_epsilon_vals):
			epsilon_val_i = epsilon_vals[epsilon_i]

			for gamma_i in range(num_gamma_vals):
				gamma_val_i = gamma_vals[gamma_i]

				V_r0, T_r0, k_D_r0 = np.hsplit( np.squeeze(dep_field_vals[delta_i, 0, gamma_i, epsilon_i, :, :]), 3)
				V_r0_sorted = V_r0[inds_arclens_used_sort] # sort these by arclength from exo pt
				T_r0_sorted = T_r0[inds_arclens_used_sort] 
				k_D_r0_sorted = k_D_r0[inds_arclens_used_sort]

				clr = plt.cm.tab10(np.float64(color_i+1)/(num_plots_per_subplot+1)) #different color for each params
				if(delta_i == 0):
					labels = concatenate(( labels, ['$\gamma$ ='+str(gamma_val_i)+', $\epsilon$ = '+str(epsilon_val_i)] )) #add to labels

				#\| V \| plot
				plt.subplot(num_delta_vals, 3, plot_num, aspect = 1./max_V_mag )
				plt.plot(arclens_used_sorted, V_r0_sorted, '-o', c=clr, markersize=mk_size) 
				plt.ylim(bottom=0., top= max_V_mag)
								
				plt.xlabel("arclength")
				plt.ylabel('$\|V\|$')

				#T plot
				min_T_delta = T_min_delta[delta_i]
				max_T_delta = T_max_delta[delta_i]
				plt.subplot(num_delta_vals, 3, plot_num+1, aspect = 1./(max_T_delta-min_T_delta) )
				plt.plot(arclens_used_sorted, T_r0_sorted, '-o', c=clr, markersize=mk_size) 
				plt.ylim(bottom=min_T_delta, top=max_T_delta )
				
				plt.xlabel("arclength")
				plt.ylabel("T")

				#k_D plot
				plt.subplot(num_delta_vals, 3, plot_num+2, aspect = 1. )
				plt.plot(arclens_used_sorted, k_D_r0_sorted, '-o', c=clr, markersize=mk_size) 
				plt.ylim(bottom=0., top= 1.)

				plt.xlabel("arclength")
				plt.ylabel('$k_D$')

				color_i = color_i + 1
		
		plot_num = plot_num + 3

	#plt.legend(labels, loc='upper right', bbox_to_anchor=(0.5, -0.05))
	plt.legend(labels, loc='upper left', bbox_to_anchor=(1.05, 1))
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


	'''
	for p in range(len(p_degs)):
		
		clr = plt.cm.YlOrBr(np.float64(p+1)/(len(p_degs)+1)) #different color for each degree plot
		plt.semilogy(q_vals, errors[p,:], '-o', color = clr) #plot for different quads used
		str_lbl_p = str(p_degs[p]) #label by degree
		labels = concatenate(( labels, ["p_deg ="+str_lbl_p] )) #add to labels
	
	plt.legend(labels, loc='upper right') #show legend
	# plt.legend(lns, labels, loc='upper right', bbox_to_anchor=(0.5, -0.05))
	'''

	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = fig_title + str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


# Plot 2-param values for fig 2 protein flow paper:
def plot_Prot_mem_driven_flows_fig2(manny_label, param_vals, dep_field_vals, indep_var_vals, use_linearization, gamma_i):
	'''
	param_val = list(delta_vals, beta_val (USE .6), gamma_vals, epsilon_vals, alpha_vals) #Params_Lists = [delta_list, beta_list, gamma_list, epsilon_list, alpha_list ] 
	dep_field_vals = (num_r0 (\delta), num_beta, num_gamma, num_eps, 2 {\rho_prot_LIN, \rho_prot_NON_LIN}, num_quad_pts (Q)
	indep_var_vals = {K, arclen} for each r_0, shape: {num_r0, Q, 2}
	gamma_i = index of gamma value to use
	'''

	num_quad_pts_to_plot = indep_var_vals.shape[1] 
	mk_size = min(200./num_quad_pts_to_plot, 1.)

	delta_vals = param_vals[0]
	num_delta_vals = len(delta_vals)

	beta_val_used = param_vals[1][0] # WE ONLY HAVE ONE 

	gamma_vals = param_vals[2]
	num_gamma_vals = len(gamma_vals)

	# USED GAMMA VALUE:
	gamma_val_i = gamma_vals[gamma_i]

	epsilon_vals = param_vals[3]
	num_epsilon_vals = len(epsilon_vals)

	alpha_vals = param_vals[4]
	num_alpha_vals = len(alpha_vals)

	num_plots_per_subplot = num_alpha_vals # plots of params on each subplot
	num_plots = num_delta_vals*num_epsilon_vals

	label_str_lin = ""
	if(use_linearization == True):
		label_str_lin = "_LINEARIZED_"
	else:
		label_str_lin = "_NL_"

	fig = plt.figure(figsize=(8.5, 8.5))
	plt.subplots_adjust(hspace=0.4)	
	fig_title = manny_label +"_beta_"+str(beta_val_used).replace(".", "pt") +"_gamma_"+str(gamma_val_i).replace(".", "pt") +"_"+ label_str_lin + "_Prot_mem_driven_flow_paper_fig2_"
	plt.ticklabel_format(axis='both', style='plain')
	#plt.suptitle(fig_title, fontsize=12, y=1.08) # plt.title(figure_title, y=1.08)
		
	plot_num = 1
	labels = array([]) #label by param value		

	for delta_i in range(num_delta_vals):

		delta_val_i = param_vals[0][delta_i]
		arclens_used = indep_var_vals[delta_i, :, 1]

		inds_arclens_used_sort = arclens_used.argsort() # we need to sort these to connect plot lines
		arclens_used_sorted = np.sort(arclens_used)

		for epsilon_i in range(num_epsilon_vals):
			epsilon_val_i = epsilon_vals[epsilon_i]

			color_i = 0			

			for alpha_i in range(num_alpha_vals):
				alpha_val_i = alpha_vals[alpha_i]

				# prot_rho_param_vals_for_plot[R_0_test, alpha_i, beta_ind, gamma_ind, epsilon_ind, 2, :]
				rho_r0_eps_alpha = []
				if(use_linearization == True):
					rho_r0_eps_alpha = np.squeeze(dep_field_vals[delta_i, alpha_i, 0, gamma_i, epsilon_i, 0, :])
				else:
					rho_r0_eps_alpha = np.squeeze(dep_field_vals[delta_i, alpha_i, 0, gamma_i, epsilon_i, 1, :])
				rho_r0_eps_alpha_sorted = rho_r0_eps_alpha[inds_arclens_used_sort] # sort these by arclength from exo pt

				clr = plt.cm.tab10(np.float64(color_i+1)/(num_plots_per_subplot+1)) #different color for each params
				if(delta_i == 0 and epsilon_i == 0):
					labels = concatenate(( labels, ['$\gamma$ ='+str(gamma_val_i)+", "+r'$\alpha$'+" = "+str(alpha_val_i)] )) #add to labels

				#\rho plot
				plt.subplot(num_delta_vals, 3, plot_num, aspect='equal') 
				plt.ylim(bottom=0., top= 1.)
				plt.plot(arclens_used_sorted, rho_r0_eps_alpha_sorted, '-o', c=clr, markersize=mk_size)

				plt.xlabel('arclength') #($\delta$ = '+str(delta_val_i)+', $\epsilon$ = '+str(epsilon_val_i)+")")
				plt.ylabel(r'$\rho$')#'$\rho$')
				plt.title('$\delta$'+" = "+str(delta_val_i)[0:4]+", "+'$\epsilon$'+" = "+str(epsilon_val_i))

				color_i = color_i + 1
		
			plot_num = plot_num + 1

	#plt.legend(labels, loc='upper right', bbox_to_anchor=(0.5, -0.05))
	plt.legend(labels, loc='upper left', bbox_to_anchor=(1.05, 1))
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = fig_title + str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


# Plot flow info for different geos, for fig 3 membrane flow paper:
def plot_mem_flows_fig3(manny_label, r0_vals, dep_field_vals, indep_var_vals, plot_vs_K_or_arclen):
	'''
	Indep = K, arclen
	dep = \| V \|, T, k_D
	dep_field_vals = (num_r0, num_beta, num_gamma, num_eps, 3 {fields}, num_quad_pts (Q)
	indep_var_vals = {K, arclen} for each r_0, shape: {num_r0, Q, 2}
	plot_vs_K_or_arclen = True: we plot vs K; False: we plot vs arclen
	'''

	num_r0_vals = len(r0_vals) # new row for each plot, col for {V, T, k_D}
	num_quad_pts_to_plot = indep_var_vals.shape[1] 
	mk_size = min(200./num_quad_pts_to_plot, 1.)
	mk_type = []

	num_beta_vals = dep_field_vals.shape[1]
	num_gamma_vals = dep_field_vals.shape[2]
	num_epsilon_vals = dep_field_vals.shape[3] 

	num_param_val = num_beta_vals*num_gamma_vals*num_epsilon_vals	

	fig = plt.figure()
	plt.subplots_adjust(hspace=0.4)	
	plt.ticklabel_format(axis='both', style='plain')

	indep_var_name = ""
	if(plot_vs_K_or_arclen == True):
		indep_var_name = "Gauss_Curv"
		mk_type = 'o'
	else:
		indep_var_name = "arclength"
		mk_type = '-o'	

	fig_title = manny_label + "_mem_flow_paper_fig3"+"_plots_against_"+indep_var_name
	#plt.suptitle(fig_title, fontsize=12)	

	plot_num = 1

	for r0_i in range(num_r0_vals):
		r0_val_i = r0_vals[r0_i]

		indep_var_val_used = []
		if(plot_vs_K_or_arclen == True):
			indep_var_val_used = indep_var_vals[r0_i, :, 0]
		else:
			indep_var_val_used = indep_var_vals[r0_i, :, 1]

		inds_indep_var_used_sort = indep_var_val_used.argsort() # we need to sort these to connect plot lines
		indep_var_used_sorted = np.sort(indep_var_val_used)

		V_r0, T_r0, k_D_r0 = np.hsplit( np.squeeze(dep_field_vals[r0_i, 0, 0, 0, :, :]), 3)
		V_r0_sorted = V_r0[inds_indep_var_used_sort]
		T_r0_sorted = T_r0[inds_indep_var_used_sort]
		k_D_r0_sorted = k_D_r0[inds_indep_var_used_sort]
		
		#\| V \| plot
		plt.subplot(num_r0_vals, 3, plot_num)
		plt.plot(indep_var_used_sorted, V_r0_sorted, mk_type, c='red', markersize=mk_size) 

		plt.xlabel(indep_var_name)
		plt.ylabel('$\|V\|$')

		# T plot
		plt.subplot(num_r0_vals, 3, plot_num+1)
		plt.plot(indep_var_used_sorted, T_r0_sorted, mk_type,  c='red',  markersize=mk_size)

		plt.xlabel(indep_var_name)
		plt.ylabel("T")

		# k_D  plot
		plt.subplot(num_r0_vals, 3, plot_num+2)
		plt.plot(indep_var_used_sorted, k_D_r0_sorted, mk_type,  c='red',  markersize=mk_size)

		plt.xlabel(indep_var_name)
		plt.ylabel('$k_D$')

		
		plot_num = plot_num + 3

	'''
	for p in range(len(p_degs)):
		
		clr = plt.cm.YlOrBr(np.float64(p+1)/(len(p_degs)+1)) #different color for each degree plot
		plt.semilogy(q_vals, errors[p,:], '-o', color = clr) #plot for different quads used
		str_lbl_p = str(p_degs[p]) #label by degree
		labels = concatenate(( labels, ["p_deg ="+str_lbl_p] )) #add to labels
	
	plt.legend(labels, loc='upper right') #show legend
	# plt.legend(lns, labels, loc='upper right', bbox_to_anchor=(0.5, -0.05))
	'''
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = fig_title + str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


# Plot 2-param values for fig 4 plot analyzing distribution of \| V \|:
def plot_mem_vel_prot_mag_fig4(manny_label, param_vals2, Vel_or_Prot_Seg_Outputs, Vel_or_Prot_field):
	'''
	param_val = list(r0_vals, gamma_vals, epsilon_vals, beta_vals) , Assume beta_val = .6, OR  list(r0_vals, gamma_vals, epsilon_vals, beta_vals, alpha_vals)
	Vel_or_Prot_Seg_Outputs = (num_r0_vals, num_gamma_vals, num_eps_vals) OR  (num_r0_vals, num_gamma_vals, num_eps_vals, num_alpha_vals) for protein case
	Vel_or_Prot_field: True if vel, false if prot
	'''

	mk_size = 1.	

	r0_vals_used = param_vals2[0]
	num_r0_vals = len(r0_vals_used)

	beta_val_used = param_vals2[3][0] # WE ONLY HAVE ONE 

	gamma_vals = param_vals2[1]
	num_gamma_vals = len(gamma_vals)

	epsilon_vals = param_vals2[2]
	num_epsilon_vals = len(epsilon_vals)

	alpha_vals = []
	num_alpha_vals = 0
	if(Vel_or_Prot_field == False):
		alpha_vals = param_vals2[4]
		num_alpha_vals = len(alpha_vals)

	num_plots_per_subplot = num_gamma_vals*num_epsilon_vals # plots of params on each subplot

	if(Vel_or_Prot_field == False):
		num_plots_per_subplot = num_plots_per_subplot*num_alpha_vals

	fig = plt.figure(figsize=(8.5, 8.5))
	plt.subplots_adjust(hspace=0.4)
	
	type_label = []
	if(Vel_or_Prot_field == True):
		type_label = "_mem_vel_mag"
	else:
		type_label = "_NL_prot_SS"

	fig_title = manny_label +"_beta_"+str(beta_val_used).replace(".", "pt")+ type_label + "_seg_fig4_"
	#plt.suptitle(fig_title, fontsize=12, y=1.08) # plt.title(figure_title, y=1.08)

	labels = array([]) #label by param value

	color_i = 0

	for epsilon_i in range(num_epsilon_vals):
		epsilon_val_i = epsilon_vals[epsilon_i]

		for gamma_i in range(num_gamma_vals):
			gamma_val_i = gamma_vals[gamma_i]

			if(Vel_or_Prot_field == True):
				Vel_Seg_Outputs_gamma_eps = Vel_or_Prot_Seg_Outputs[:, gamma_i, epsilon_i].flatten()

				clr = plt.cm.tab10(np.float64(color_i+1)/(num_plots_per_subplot+1)) #different color for each params			
				labels = concatenate(( labels, ['$\gamma$ ='+str(gamma_val_i)+', $\epsilon$ = '+str(epsilon_val_i)] )) #add to labels

				#\| V \| plot
				plt.subplot(1, 1, 1)
				plt.plot(r0_vals_used, Vel_Seg_Outputs_gamma_eps, '-o', c=clr, markersize=mk_size) 

				plt.xlabel('$r_0$')
				plt.ylabel('$\|V\|$'+" density ratio")

				color_i = color_i + 1		
			else:
				for alpha_i in range(num_alpha_vals):
					alpha_val_i = alpha_vals[alpha_i]

					Prot_Seg_Outputs_gamma_eps_alpha = Vel_or_Prot_Seg_Outputs[:, gamma_i, epsilon_i, alpha_i].flatten()

					clr = plt.cm.tab10(np.float64(color_i+1)/(num_plots_per_subplot+1)) #different color for each params			
					labels = concatenate(( labels, ['$\gamma$ ='+str(gamma_val_i)+', $\epsilon$ = '+str(epsilon_val_i)+
", "+r'$\alpha$'+" = "+str(alpha_val_i)] )) #add to labels

					# rho density plot
					plt.subplot(1, 1, 1)
					plt.plot(r0_vals_used, Prot_Seg_Outputs_gamma_eps_alpha, '-o', c=clr, markersize=mk_size) 

					plt.xlabel('$r_0$')
					plt.ylabel(r'$\rho$'+" density ratio")

					color_i = color_i + 1		

	#plt.legend(labels, loc='upper right', bbox_to_anchor=(0.5, -0.05))
	plt.legend(labels, loc='upper left', bbox_to_anchor=(1.05, 1))
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = fig_title + str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')

	
# Plot scalar fields vs time
def plot_scalar_fields_over_time(plt_title, times, dependent_variables, labels, log=False, markersize = .2, plot_type = 'o', plot_dir = [], bspline=False, y_lim_bottom_in = [], y_lim_top_in = [], Input_Dir=[]):
	
	fig = plt.figure()	

	num_plots = dependent_variables.shape[1]
	print("num_plots = "+str(num_plots))
	Dependent_Fields = np.hsplit(dependent_variables, num_plots)
	
	# inputs, for switching for b-spline plots:
	markersize_in = markersize
	plot_type_in = plot_type

	for plot_i in range(num_plots):

		#out_bspline = [] # if we use b-spline interpolation with pts as knots
		u_spl = []
		spl_out = []
		if(bspline == True):
			# From: https://github.com/kawache/Python-B-spline-examples
			'''
			l=len(times.flatten())
			t=np.linspace(0,1,l-2,endpoint=True)
			t=np.append([0,0,0],t)
			t=np.append(t,[1,1,1])

			#tck,u = sp.interpolate.splprep(t.flatten(), [times.flatten(), Dependent_Fields[plot_i].flatten() ],k=3,s=0)
			tck = [t.flatten(), [times.flatten(), Dependent_Fields[plot_i].flatten()], 3 ]
			u = np.linspace(0,1,(max(l*2,70)),endpoint=True)
			out_bspline = sp.interpolate.splev(u,tck)
			'''
			
			# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
			spl = sp.interpolate.UnivariateSpline(times.flatten(), Dependent_Fields[plot_i].flatten())			
			spl.set_smoothing_factor(0.5)
			u_spl = np.linspace(min(times),max(times),1000,endpoint=True)
			spl_out = spl(u_spl)


		# https://matplotlib.org/examples/color/colormaps_reference.html
		clr = plt.cm.seismic(np.float64(plot_i+1)/(np.float64(num_plots))) #different color for each plot		
		
		if(log == False):
			if(bspline == True):
				markersize = markersize_in
				plot_type = plot_type_in
				#plt.plot(out_bspline[0], out_bspline[1], plot_type, color = clr, markersize=markersize)
				plt.plot(u_spl, spl_out, plot_type, color = clr, markersize=markersize, label='_nolegend_')
				plot_type = 'o' # '^'
				markersize = 3.*markersize_in

			if(y_lim_bottom_in != []): # reset the y-limit at bottom
				plt.ylim(bottom=y_lim_bottom_in, top= 1.1*max(Dependent_Fields[plot_i]))

				if(y_lim_top_in != []):
					plt.ylim(bottom=y_lim_bottom_in, top=y_lim_top_in)
					print(plt_title +": y_lim_bottom_in = "+str(y_lim_bottom_in)+", y_lim_top_in = "+str(y_lim_top_in))
				else:
					print(plt_title +": y_lim_bottom_in = "+str(y_lim_bottom_in))

			if(y_lim_top_in != [] and y_lim_bottom_in == []):
				plt.ylim(bottom=min(Dependent_Fields[plot_i]) - abs(min(Dependent_Fields[plot_i]))/10., top=y_lim_top_in)
				print(plt_title +": y_lim_top_in = "+str(y_lim_top_in))

			if(y_lim_bottom_in != [] and y_lim_top_in != []):
				plt.ylim(bottom=y_lim_bottom_in, top=y_lim_top_in)
				print(plt_title +": y_lim_bottom_in = "+str(y_lim_bottom_in)+", y_lim_top_in = "+str(y_lim_top_in))

			plt.plot(times, Dependent_Fields[plot_i], plot_type, color = clr, markersize=markersize) # CHANGED FOR DROPLET CODE
 
		else:
			if(bspline == True):
				markersize = markersize_in
				plot_type = plot_type_in
				#plt.semilogy(out_bspline[0], out_bspline[1], plot_type, color = clr, markersize=markersize)
				plt.semilogy(u_spl, spl_out, plot_type, color = clr, markersize=markersize, label='_nolegend_')
				plot_type = 'o' # '^'
				markersize = 3.*markersize_in

			plt.semilogy(times, Dependent_Fields[plot_i], plot_type, color = clr, markersize=markersize)

	#plt.legend(labels, loc='upper right', prop={'size': 6}) #show legend
	plt.axes().set_aspect(1.0/plt.axes().get_data_ratio(), adjustable='box')
	plt.margins(0)
	for axis in ['top','bottom','left','right']:
		plt.axes().spines[axis].set_linewidth(0.25)
	plt.legend(labels, loc='upper left', bbox_to_anchor=(1.05, 1))
	#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	#plt.legend(labels, loc='upper center', bbox_to_anchor=(0.45, -0.1), prop={'size': 6})

	plt.title(plt_title)
	
	#plt.show()

	# Pickle Result in Pickled_Images:
	Picked_Image_DIR = []
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 

	if(plot_dir == []):
		Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images')
	else:
		Picked_Image_DIR = plot_dir

	if(Input_Dir != []):
		Picked_Image_DIR = Input_Dir

	fig_name = str(plt_title)+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf', bbox_inches='tight', transparent=True)
	

	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = plt_title
	Data_Dictionary['times'] = times
	Data_Dictionary['dependent_variables'] = dependent_variables
	Data_Dictionary['labels'] = labels

	
	Dictionary_Name = str(plt_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)
	
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)


# Plot scalar fields vs time, from list of times and dep vars (times NOT the same)
def plot_scalar_fields_over_time_From_List(plt_title, times_list, dependent_variables_list, labels, log=False, markersize = .2, plot_type = 'o', plot_dir = [], bspline=False, y_lim_bottom_in = [], y_lim_top_in = []):
	
	fig = plt.figure()	

	num_plots = len(dependent_variables_list)
	print("num_plots = "+str(num_plots))
	#Dependent_Fields = np.hsplit(dependent_variables, num_plots)
	
	# inputs, for switching for b-spline plots:
	markersize_in = markersize
	plot_type_in = plot_type

	for plot_i in range(num_plots):
		times = times_list[plot_i]		

		#out_bspline = [] # if we use b-spline interpolation with pts as knots
		u_spl = []
		spl_out = []
		if(bspline == True):
			# From: https://github.com/kawache/Python-B-spline-examples
			'''
			l=len(times.flatten())
			t=np.linspace(0,1,l-2,endpoint=True)
			t=np.append([0,0,0],t)
			t=np.append(t,[1,1,1])

			#tck,u = sp.interpolate.splprep(t.flatten(), [times.flatten(), Dependent_Fields[plot_i].flatten() ],k=3,s=0)
			tck = [t.flatten(), [times.flatten(), Dependent_Fields[plot_i].flatten()], 3 ]
			u = np.linspace(0,1,(max(l*2,70)),endpoint=True)
			out_bspline = sp.interpolate.splev(u,tck)
			'''
			
			# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
			spl = sp.interpolate.UnivariateSpline(times.flatten(), dependent_variables_list[plot_i].flatten())			
			spl.set_smoothing_factor(0.5)
			u_spl = np.linspace(min(times),max(times),1000,endpoint=True)
			spl_out = spl(u_spl)


		# https://matplotlib.org/examples/color/colormaps_reference.html
		clr = plt.cm.seismic(np.float64(plot_i+1)/(np.float64(num_plots))) #different color for each plot		
		
		if(log == False):
			if(bspline == True):
				markersize = markersize_in
				plot_type = plot_type_in
				#plt.plot(out_bspline[0], out_bspline[1], plot_type, color = clr, markersize=markersize)
				plt.plot(u_spl, spl_out, plot_type, color = clr, markersize=markersize, label='_nolegend_')
				plot_type = 'o' #'^'
				markersize = 3.*markersize_in

			#print("plot_i = "+str(plot_i)+", times.shape = "+str(times.shape)+", dependent_variables_list[plot_i].shape = "+str(dependent_variables_list[plot_i].shape))

			if(y_lim_bottom_in != []): # reset the y-limit at bottom
				plt.ylim(bottom=y_lim_bottom_in, top= 1.1*max(dependent_variables_list[plot_i]))

				if(y_lim_top_in != []):
					plt.ylim(bottom=y_lim_bottom_in, top=y_lim_top_in)

			if(y_lim_top_in != [] and y_lim_bottom_in == []):
				plt.ylim(bottom=min(dependent_variables_list[plot_i]) - abs(min(dependent_variables_list[plot_i]))/10., top=y_lim_top_in)

			plt.plot(times, dependent_variables_list[plot_i], plot_type, color = clr, markersize=markersize) # CHANGED FOR DROPLET CODE
 
		else:
			if(bspline == True):
				markersize = markersize_in
				plot_type = plot_type_in
				#plt.semilogy(out_bspline[0], out_bspline[1], plot_type, color = clr, markersize=markersize)
				plt.semilogy(u_spl, spl_out, plot_type, color = clr, markersize=markersize, label='_nolegend_')
				plot_type = 'o' #'^'
				markersize = 3.*markersize_in

			plt.semilogy(times, dependent_variables_list[plot_i], plot_type, color = clr, markersize=markersize)

	plt.axes().set_aspect(1.0/plt.axes().get_data_ratio(), adjustable='box')
	plt.margins(0)
	for axis in ['top','bottom','left','right']:
		plt.axes().spines[axis].set_linewidth(0.25)
	plt.legend(labels, loc='upper right', bbox_to_anchor=(1.2, 1), prop={'size': 6}) #show legend
	#plt.legend(labels, loc='upper center', bbox_to_anchor=(0.45, -0.1), prop={'size': 6})
	plt.title(plt_title)
	
	plt.show()

	# Pickle Result in Pickled_Images:
	Picked_Image_DIR = []
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 

	if(plot_dir == []):
		Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images')
	else:
		Picked_Image_DIR = plot_dir

	fig_name = str(plt_title)+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf', bbox_inches='tight', transparent=True)

	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = plt_title
	Data_Dictionary['times_list'] = times_list
	Data_Dictionary['dependent_variables_list'] = dependent_variables_list
	Data_Dictionary['labels'] = labels

	
	Dictionary_Name = str(plt_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)
	
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)


# plot 3d data, for spatio-temporal droplet corrs:
def Plot3DWireFrame(plt_title, indep_var1, indep_var2, dep_var_mat, labels, plot_dir = []):
	
	#fig = plt.figure()
	fig, ax = plt.subplots()


	#plt.plot(times, Dependent_Fields[plot_i], plot_type, markersize=markersize) # CHANGED FOR DROPLET CODE
	#axis.plot_wireframe(indep_var1, indep_var2, dep_var_mat)
	c = ax.pcolormesh(indep_var1, indep_var2, dep_var_mat, cmap='jet',vmin=min(dep_var_mat.flatten()), vmax=max(dep_var_mat.flatten()))
	fig.colorbar(c, ax=ax)
	ax.set_xlabel(labels[0])
	ax.set_ylabel(labels[1])
	#ax.set_zlabel(labels[2])

	plt.legend(labels[2], loc='upper right', prop={'size': 6}) #show legend
	#plt.legend(labels, loc='upper center', bbox_to_anchor=(0.45, -0.1), prop={'size': 6})
	plt.title(plt_title)
	
	plt.show()

	# Pickle Result in Pickled_Images:
	Picked_Image_DIR = []
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 

	if(plot_dir == []):
		Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images')
	else:
		Picked_Image_DIR = plot_dir

	fig_name = str(plt_title)+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')
	
	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = plt_title
	Data_Dictionary['indep_var1'] = indep_var1
	Data_Dictionary['indep_var2'] = indep_var2
	Data_Dictionary['dependent_variables'] = dep_var_mat
	Data_Dictionary['labels'] = labels

	Dictionary_Name = str(plt_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)
	
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)


# for droplet code, plot magnitude of terms:
def Plot_3D_Heatmap(plt_title, indep_var1, indep_var2, dep_var_mat, labels, plot_dir = []):

	# USE THIS CODE, from: https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
	import numpy as np
	import seaborn as sns
	import matplotlib.pylab as plt

	fig = plt.figure()	
	plt.title(plt_title)

	ax = sns.heatmap(
	    dep_var_mat,
	    indep_var1, 
	    indep_var2,
	)

	ax.set_xlabel(labels[0])
	ax.set_ylabel(labels[1])

	plt.show()

	# Pickle Result in Pickled_Images:
	Picked_Image_DIR = []
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 

	if(plot_dir == []):
		Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images')
	else:
		Picked_Image_DIR = plot_dir

	fig_name = str(plt_title)+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')


# for droplet code, plot magnitude of terms:
def Plot_Matrix_Heatmap(plt_title, matrix_input):

	# USE THIS CODE, from: https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
	import numpy as np
	import seaborn as sns
	import matplotlib.pylab as plt
	from matplotlib.colors import LogNorm


	fig = plt.figure()	
	plt.title(plt_title)

	#ax = sns.heatmap(matrix_input, linewidth=0.5)
	log_norm = LogNorm(vmin=matrix_input.min().min(), vmax=matrix_input.max().max()) # log-norm from: https://www.thetopsites.net/article/50429431.shtml
	cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(matrix_input.min().min())), 1+math.ceil(math.log10(matrix_input.max().max())))]

	ax = sns.heatmap(
	    matrix_input,
	    norm=log_norm,
	    cbar_kws={"ticks": cbar_ticks}
	)

	plt.show()

	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images') 

	fig_name = plt_title+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')

# for droplet code, plot magnitude of terms (LINEAR, Theshold to SUM):
def Plot_Matrix_Heatmap_Linear_Adj(plt_title, matrix_input, threshold_val, Only_Upper_Tri = False, Input_Dir = []):

	# USE THIS CODE, from: https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
	import numpy as np
	import seaborn as sns
	import matplotlib.pylab as plt

	fig = plt.figure()	
	plt.title(plt_title)

	mode_sum = np.sum(matrix_input.flatten())
	matrix_in_adj = matrix_input/mode_sum # adjust to largest val
	matrix_in_adj = np.where(matrix_in_adj >= threshold_val, matrix_in_adj, 0.) # get rid of entries below threshold

	#print("matrix_in_adj = "+str(matrix_in_adj))
	#print("mode_sum = "+str(mode_sum))
	#print("	matrix_input = "+str(matrix_input))

	mask = np.zeros_like(matrix_input, dtype=np.bool)
	if(Only_Upper_Tri == True):
		mask[np.tril_indices_from(mask, k=-1)] = True

	ax = sns.heatmap(
	    matrix_in_adj,
 	    mask=mask,
	    vmin=0., 
	    vmax= np.max( matrix_in_adj.flatten() )#1.
	)

	plt.show()

	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
	Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images')

	if(Input_Dir != []):
		Picked_Image_DIR = Input_Dir

	fig_name = plt_title+"_LINEAR_ADJ_Thesholded_"+str(threshold_val).replace(".","pt")+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')

	return matrix_in_adj # for further analysis in Droplet code


# for Droplet code, we want to do some scatter plots:
def SimpleScatterPlot(plt_title, X_scatter_pts, Y_scatter_pts, xlabel, ylabel, plot_type = 'o', plot_dir = []):

	fig = plt.figure()
	plt.title(plt_title)	
	plt.scatter(X_scatter_pts.flatten(), Y_scatter_pts.flatten(), marker=plot_type) # CHANGED FOR DROPLET CODE
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

	# Pickle Result in Pickled_Images:
	Picked_Image_DIR = []
	MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 

	if(plot_dir == []):
		Picked_Image_DIR = os.path.join(MY_DIR, 'Pickled_Images')
	else:
		Picked_Image_DIR = plot_dir

	fig_name = str(plt_title)+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Picked_Image_DIR, fig_name)

	fig.savefig(fig_path, format='pdf')
	
	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = plt_title
	Data_Dictionary['x_vals'] = X_scatter_pts
	Data_Dictionary['x_label'] = xlabel
	Data_Dictionary['x_vals'] = Y_scatter_pts
	Data_Dictionary['x_label'] = ylabel

	Dictionary_Name = str(plt_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)
	
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)


# Plot scalar fields vs time From DICTIONARY, allow data to be output to specified directory:
def plot_scalar_fields_over_time_From_Dict(Scalar_Field_Plot_Dict):#plt_title, times, dependent_variables, labels, log=False):

	plt_title = Scalar_Field_Plot_Dict['plot_title']
	times = Scalar_Field_Plot_Dict['times_used']
	dependent_variables = Scalar_Field_Plot_Dict['dependent_vars_plotted']
	labels = Scalar_Field_Plot_Dict['plot_labels']
	log = Scalar_Field_Plot_Dict['Use_Semilog_Plot'] # false or 0 = normal, 1 or True = semi-log, 2 = log-log 
	Pickled_Image_sub_dir = Scalar_Field_Plot_Dict['Pickled_Image_Subfolder'] # [] gives us default settings

	fig = plt.figure()	

	num_plots = dependent_variables.shape[1]
	print("num_plots = "+str(num_plots))
	Dependent_Fields = np.hsplit(dependent_variables, num_plots)
	
	# whether we have a connected or disconnected set of points:
	plot_connector = '-o'
	plot_pts_connected = True

	if('connect_plot_pts' in Scalar_Field_Plot_Dict):
		if(Scalar_Field_Plot_Dict['connect_plot_pts'] == False):
			plot_pts_connected = False
			plot_connector = 'ro'
		

	for plot_i in range(num_plots):	
		
		# https://matplotlib.org/examples/color/colormaps_reference.html
		clr = plt.cm.Dark2(np.float64(plot_i)/(np.float64(num_plots))) #different color for each plot		

		if('Different_Times_Used_for_plots' in Scalar_Field_Plot_Dict):
			if(Scalar_Field_Plot_Dict['Different_Times_Used_for_plots'] == True):
				times_i = np.hsplit(times, num_plots)[plot_i]
		else:
			times_i = times

		lists_sorted_vars = sorted(itertools.izip(*[times_i, Dependent_Fields[plot_i]]))
		sorted_times_i, sorted_dependent_field_i = list(itertools.izip(*lists_sorted_vars))		

		if(log == False or log == 0):
			plt.plot(sorted_times_i, sorted_dependent_field_i, plot_connector, color = clr)
		elif(log == True or log == 1):
			plt.semilogy(sorted_times_i, sorted_dependent_field_i, plot_connector, color = clr)
		else:
			plt.loglog(sorted_times_i, sorted_dependent_field_i, plot_connector, color = clr)
	
	# Test whether or not we put legend in seperate figure
	include_legend_in_same_plt = True 

	if('seperate_legend' in Scalar_Field_Plot_Dict):
		if(Scalar_Field_Plot_Dict['seperate_legend'] == False):
			include_legend_in_same_plt = False

	if(include_legend_in_same_plt == True):
		plt.legend(labels, loc='upper right') #show legend in same plot

	plt.title(plt_title)
	
	plt.show()

	# Pickle Result in Pickled_Images:
	MY_DIR  = os.path.realpath(os.path.dirname(__file__))
	Pickle_Folder_Path  = os.path.join(MY_DIR, 'Pickled_Images') 

	# Modify to particular path:
	if(Pickled_Image_sub_dir != []):
		Pickle_Folder_Path = os.path.join(Pickle_Folder_Path, Pickled_Image_sub_dir)

	fig_name = str(plt_title)+"_"+str(Time_and_Date_Str())+".pdf" #Creates unique filename
	fig_path = os.path.join(Pickle_Folder_Path, fig_name)
	

	fig.savefig(fig_path, format='pdf')
	

	# Pickled Plot Data in Pickled_Data:
	Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')

	Data_Dictionary = {}
	Data_Dictionary['Plot_Title'] = plt_title
	Data_Dictionary['times'] = times
	Data_Dictionary['dependent_variables'] = dependent_variables
	Data_Dictionary['labels'] = labels

	
	Dictionary_Name = str(plt_title)+"_dictionary"+str(Time_and_Date_Str())+".pickle"
	dictionary_path = os.path.join(Pickle_Folder_Path, Dictionary_Name)

	# Default: dump to Pickled_Data, otherwise dump: with other images in folder as well
	if(Pickled_Image_sub_dir != []):	
		new_dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)

		with open(new_dictionary_path, 'wb') as dict_file:
			pkl.dump(Data_Dictionary, dict_file)

	else:
		Picked_Data_DIR = os.path.join(MY_DIR, 'Pickled_Data')
		new_dictionary_path = os.path.join(Picked_Data_DIR, Dictionary_Name)

		with open(new_dictionary_path, 'wb') as dict_file:
			pkl.dump(Data_Dictionary, dict_file)
		
	# Default: dump to Pickled_Data, otherwise dump: with other images in folder as well
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Data_Dictionary, dict_file)


# Allows a dictionary to be pickled, with a specific label and path:
def Pickle_A_Dictionary(Dict_Pickled, Full_Folder_Path, dict_name):
	
	Dictionary_Name = dict_name + ".pickle"
	dictionary_path = os.path.join(Full_Folder_Path, Dictionary_Name)
		
	# Default: dump to Pickled_Data, otherwise dump: with other images in folder as well
	with open(dictionary_path, 'wb') as dict_file:
		pkl.dump(Dict_Pickled, dict_file)


# Plots convergence of each coeficient in SPH
def Coef_Wise_Conv_Plots(coef_conv_title, sph_deg, coef_errors, quad_degs, quad_vals, Diag_Only, Single_Elt, Coef_Matrix, Num_Non_Zero_Elts):
	
	# Plot log(Error) vs Quad Deg:
	plt.suptitle(str(coef_conv_title)+" Convergence [log(Coef Conv)+10^("+str(round(log10(np.finfo(float).eps/2), 1))+") vs Quad_Deg]", fontsize=12)

	plot_num = 1

	for row in range(sph_deg+1):
		for col in range(sph_deg+1):
			
			if(Single_Elt == True):
				if(Coef_Matrix[row][col] == 1):	
					plt.subplot(ceil(sqrt(Num_Non_Zero_Elts)), ceil(sqrt(Num_Non_Zero_Elts)), plot_num)					
					
					plt.grid(True)
					plt.semilogy(quad_degs, abs(coef_errors[row,col,:])+np.finfo(float).eps/2, '-o')			
					plt.title("("+str(row)+", "+str(col)+")", fontsize=6)

					plot_num = plot_num + 1

			else:
				if(Diag_Only == False):
					plt.subplot(sph_deg+1, sph_deg+1, plot_num)

					plt.grid(True)			
					plt.semilogy(quad_degs, abs(coef_errors[row,col,:])+np.finfo(float).eps/2, '-o')
					plt.title("("+str(row)+", "+str(col)+")", fontsize=6)			

					plot_num = plot_num + 1
			
				elif(Diag_Only == True and row == col):
					plt.subplot(ceil(sqrt(sph_deg+1)), ceil(sqrt(sph_deg+1)), plot_num)
				
					plt.grid(True)			
					plt.semilogy(quad_degs, abs(coef_errors[row,col,:])+np.finfo(float).eps/2, '-o')
					plt.title("("+str(row)+", "+str(col)+")", fontsize=6)			

					plot_num = plot_num + 1
			
	plt.show()		

	
	
	# Plot log(Error) vs log(Quad_Pts):	
	plt.suptitle(str(coef_conv_title)+" Convergence [log(Coef Conv)+10^"+str(round(log10(np.finfo(float).eps/2), 1))+" vs log(Quad_Pts)]", fontsize=12)

	plot_num = 1

	for row in range(sph_deg+1):
		for col in range(sph_deg+1):
			
			if(Single_Elt == True):
				if(Coef_Matrix[row][col] == 1):	
					plt.subplot(ceil(sqrt(Num_Non_Zero_Elts)), ceil(sqrt(Num_Non_Zero_Elts)), plot_num)					
					
					plt.grid(True)					
					plt.loglog(quad_vals, abs(coef_errors[row,col,:])+np.finfo(float).eps/2, '-o')			
					plt.title("("+str(row)+", "+str(col)+")", fontsize=6)

					plot_num = plot_num + 1

			else:

				if(Diag_Only == False):
					plt.subplot(sph_deg+1, sph_deg+1, plot_num)
				
					plt.grid(True)			
					plt.loglog(quad_vals, abs(coef_errors[row,col,:])+np.finfo(float).eps/2, '-o')
					plt.title("("+str(row)+", "+str(col)+")", fontsize=6)	
				
					plot_num = plot_num + 1
			
				elif(Diag_Only == True and row == col):
					plt.subplot(ceil(sqrt(sph_deg+1)), ceil(sqrt(sph_deg+1)), plot_num)
				
					plt.grid(True)			
					plt.loglog(quad_vals, abs(coef_errors[row,col,:])+np.finfo(float).eps/2, '-o')
					plt.title("("+str(row)+", "+str(col)+")", fontsize=6)				

					plot_num = plot_num + 1			

	plt.show()	


# Plot a 3_D representation of Star Shaped Manifold
def Plot_Manifold_3D(R_Fn, Theta_Inc, Phi_Inc):

	#Creates Grid of Points to plot
	dtheta, dphi = 2*np.pi/Theta_Inc, np.pi/Phi_Inc
	[phi, theta] = np.mgrid[0:pi + dphi *.5:dphi,
		       0:2 * pi + dtheta *.5:dtheta]
	
	def Eval_R_Fn(Theta, Phi):	
		return R_Fn.Eval_SPH(Theta, Phi)
	
	r = Eval_R_Fn(theta, phi)
		
	x = r*np.sin(phi)*np.cos(theta)
	z = r*np.cos(phi)
	y = r*np.sin(phi)*np.sin(theta)

	ml.mesh(x, y, z, colormap="bone")
	ml.show()



#Plot the Sharp of a 1-form on a Manifold in R^3, as well as a scalar field normal to manifold, if increments specifies
def Plot_Manifold_Vectors_3D(R_Fn, Theta_Res, Phi_Res, lbdv, One_Form_Sharped_Vals, P_deg):

	# grid:
	phi_pts, theta_pts = np.mgrid[0:np.pi:Theta_Res*1j, 0:2*np.pi:Phi_Res*1j] 

	# Spherical Coordinates:
	x_S2 = np.sin(phi_pts)*np.cos(theta_pts)
	y_S2 = np.sin(phi_pts)*np.sin(theta_pts)
	z_S2 = np.cos(phi_pts)

	# Manifold, and Scalar Vector Mag Fn:
	s = R_Fn(theta_pts, phi_pts) 

	U_tan, V_tan, W_tan = np.hsplit(One_Form_Sharped_Vals, 3)
	Vec_Mags = np.sqrt(U_tan**2 + V_tan**2 + W_tan**2)

	V_mag_sph_vals = Faster_Double_Proj(Vec_Mags, P_deg, lbdv).Eval_SPH(theta_pts, phi_pts)
	
	w = V_mag_sph_vals

	plt = ml.mesh(s*x_S2, s*y_S2, s*z_S2, scalars=w, colormap='jet')

	Label = "Vector Magnitude"	
	ml.colorbar(plt, title=Label, orientation='vertical')
	

	'''
	#Creates Grid of Points to plot
	dtheta_man, dphi_man = 2*np.pi/Theta_Inc_Man, np.pi/Phi_Inc_Man
	[phi_man, theta_man] = np.mgrid[0:pi + dphi_man *.5:dphi_man,
		       0:2 * pi + dtheta_man *.5:dtheta_man]
	
	def Eval_R_Fn(Theta, Phi):	
		return R_Fn.Eval_SPH(Theta, Phi)
	
	r_man = Eval_R_Fn(theta_man, phi_man)
		
	x_man = r_man*np.sin(phi_man)*np.cos(theta_man)
	z_man = r_man*np.cos(phi_man)
	y_man = r_man*np.sin(phi_man)*np.sin(theta_man)

	#ml.mesh(x_man, y_man, z_man, colormap="bone", transparent = True, opacity = .8)
	ml.mesh(x_man, y_man, z_man, scalars=V_mags, colormap='jet')
	
	'''


	R_quad_pts = euc_kf.Extract_Quad_Pt_Vals_From_Fn(R_Fn, lbdv)
	X_tan = R_quad_pts*lbdv.X	
	Y_tan = R_quad_pts*lbdv.Y
	Z_tan = R_quad_pts*lbdv.Z


	Normalize_Tan = np.where(Vec_Mags > 1e-3, .4*1./Vec_Mags, 1)
	
	U_dir = U_tan*Normalize_Tan
	V_dir = V_tan*Normalize_Tan
	W_dir = W_tan*Normalize_Tan

	ml.quiver3d(X_tan, Y_tan, Z_tan, U_dir, V_dir, W_dir, line_width=3, scale_factor=1, colormap = 'summer')

	
	
	ml.show()



# Plot heat map on Surface for LB case:
def Plot_Heat_Map(R_fn, Scalar_Fn, Theta_Res, Phi_Res, Label):

	#Scale:
	r = 1
	
	# grid:
	phi_pts, theta_pts = np.mgrid[0:np.pi:Theta_Res*1j, 0:2*np.pi:Phi_Res*1j] #np.mgrid[0:np.pi:50j, 0:2*np.pi:40j]

	# Spherical Coordinates:
	x_S2 = r*np.sin(phi_pts)*np.cos(theta_pts)
	y_S2 = r*np.sin(phi_pts)*np.sin(theta_pts)
	z_S2 = r*np.cos(phi_pts)

	ml.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
	ml.clf()

	# Spacing between plots
	n = 3
	m = 3
	
	# Manifold, and Scalar Fn:
	s = R_fn(theta_pts, phi_pts) 
	w = Scalar_Fn(theta_pts, phi_pts)

	plt = ml.mesh(x_S2 - m, y_S2 - n, z_S2, scalars=w, colormap='jet')

	s[s < 0] *= 0.97

	s /= s.max()
	ml.mesh(s * x_S2 - m, s * y_S2 - m, s * z_S2 -m, scalars=w, colormap='jet')

	ml.colorbar(plt, title=Label, orientation='vertical')

	ml.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
	ml.show()


# returns string of present time and date, for pickling:
def Time_and_Date_Str():
	fmt='%Y_%m_%d_%H_%M_%S'
	return dt.datetime.now().strftime(fmt)
	

# Gets Triangulation of Sphere to use with quad_pts verticies, otherwise generates it 
def get_Sphere_Triangulation(path, quad_pts):
	
	# where triangulation is pickled
	Sphere_Triangulation_Folder = os.path.join(path, 'Pickled_Sphere_Triangulations')
	Sphere_Triangulation_Name = "Sphererical_Triangulation_"+str(quad_pts)+".p" 
	Sphere_Triangulation_File = os.path.join(Sphere_Triangulation_Folder, Sphere_Triangulation_Name)

	Sphere_Triangulation_Array = []	

	# Check to see if already created
	if(os.path.isfile(Sphere_Triangulation_File)):

		with open(Sphere_Triangulation_File, 'rb') as sphere_tri:
			Sphere_Triangulation_Array = pkl.load(sphere_tri)
		
	# Need to create it
	else:

		Lbdv_Cart_Pts_and_Wt_Quad = leb_wr.Lebedev(quad_pts)
		lbdv_coordinate_array = Lbdv_Cart_Pts_and_Wt_Quad[:,:-1]

		lbdv_plus_center = np.vstack(( lbdv_coordinate_array, np.array([0, 0, 0]) ))
		delauney_tetras = Delaunay(lbdv_plus_center)

		tetras = delauney_tetras.simplices
		num_tris = len(delauney_tetras.simplices)

		delauney_tris = np.zeros(( num_tris, 3 ))

		for tri_i in range(num_tris):
			vert_ind = 0

			for tetra_vert in range(4):
				vertex = tetras[tri_i, tetra_vert]		

				if(vertex != quad_pts and vert_ind < 3):

					delauney_tris[tri_i, vert_ind] = vertex
					vert_ind = vert_ind + 1	

		
		Sphere_Triangulation_Array = delauney_tris	

		with open(Sphere_Triangulation_File, 'wb') as sphere_tri_gen:
			pkl.dump(Sphere_Triangulation_Array, sphere_tri_gen)


	return Sphere_Triangulation_Array


# Gets Triangulation of Sphere to use with quad_pts verticies AND 5810 VERTS MERGED UNIQUELY, otherwise generates it 
def get_Sphere_Triangulation_lbdv_with_Max(path, quad_pts, pdeg_2nd_set):
	
	# where triangulation is pickled
	Sphere_Triangulation_Folder = os.path.join(path, 'Pickled_Sphere_Triangulations')
	Sphere_Triangulation_Name = "Sphererical_Triangulation_WITH_5810_MAX_"+str(quad_pts)+".p" 
	Sphere_Triangulation_File = os.path.join(Sphere_Triangulation_Folder, Sphere_Triangulation_Name)

	Sphere_Triangulation_Array = []	

	# Check to see if already created
	if(os.path.isfile(Sphere_Triangulation_File) and 1==2): # DISABLE PICKLING

		with open(Sphere_Triangulation_File, 'rb') as sphere_tri:
			Sphere_Triangulation_Array = pkl.load(sphere_tri)
		
	# Need to create it
	else:

		Lbdv_Cart_Pts_and_Wt_Quad = leb_wr.Lebedev(quad_pts)
		lbdv_coordinate_array = Lbdv_Cart_Pts_and_Wt_Quad[:,:-1]


		
		quad_2nd = lbdv_i.look_up_lbdv_pts(pdeg_2nd_set+1)
		LBDV_2nd = lbdv_i.lbdv_info(pdeg_2nd_set, quad_2nd)	
		lbdv_max_S2_coor_array = np.hstack(( LBDV_2nd.X, LBDV_2nd.Y, LBDV_2nd.Z ))
		'''
		X_S2_5810, Y_S2_5810, Z_S2_5810, Theta_S2_5810, Phi_S2_5810 = lbdv_i.get_5810_quad_pts()
		lbdv_max_S2_coor_array = np.hstack(( X_S2_5810, Y_S2_5810, Z_S2_5810 ))
		'''

		lbdv_plus_center = np.vstack(( lbdv_coordinate_array, lbdv_max_S2_coor_array, np.array([0, 0, 0]) ))
		#lbdv_plus_center_just = np.vstack(( np.array([0, 0, 0]), lbdv_coordinate_array ))
		#lbdv_plus_center_just2 = np.vstack(( np.array([0, 0, 0]), lbdv_max_S2_coor_array ))

		lbdv_plus_center_sorted, first_unique_inds = np.unique(lbdv_plus_center, return_index=True, axis=0)
		first_unique_inds_sorted = np.sort(first_unique_inds)
		lbdv_plus_center_unique = lbdv_plus_center[first_unique_inds_sorted, :]
		num_pts_total = lbdv_plus_center_unique.shape[0] # INCLUDES 0 (center)
		print("num_pts_total = "+str(num_pts_total))

		delauney_tetras = Delaunay(lbdv_plus_center_unique)
		#delauney_tetras.add_points(lbdv_plus_center_unique[(quad_pts+1):,:])
		#delauney_tetras = Delaunay(lbdv_plus_center_just, incremental=True)
		#delauney_tetras = Delaunay(lbdv_plus_center_just2, incremental=True)

		tetras = delauney_tetras.simplices
		num_tris = len(delauney_tetras.simplices)

		delauney_tris = np.zeros(( num_tris, 3 ))

		for tri_i in range(num_tris):
			vert_ind = 0

			for tetra_vert in range(4):
				vertex = tetras[tri_i, tetra_vert]		

				if(vertex != num_pts_total and vert_ind < 3):

					delauney_tris[tri_i, vert_ind] = vertex
					vert_ind = vert_ind + 1	

		
		Sphere_Triangulation_Array = delauney_tris # since this was shifted by adding center	

		with open(Sphere_Triangulation_File, 'wb') as sphere_tri_gen:
			pkl.dump(Sphere_Triangulation_Array, sphere_tri_gen)

	XYZ_S2_unique = lbdv_plus_center_unique[:-1, :]
	return Sphere_Triangulation_Array, XYZ_S2_unique

# get distances to each quad pt from inclusions
def Triangulation_Inclusion_Distance_Max(Suface_Pts_Cart, Inclusion_Spherical_Coors, num_tri_quad_pts, path):
	
	Cart_Coors =  Suface_Pts_Cart # Includes Inclusions
	
	triangulation = []
	
	# used pickled 5810 triangulation if no inclusions
	if(Inclusion_Spherical_Coors == []):
		#triangulation = get_Sphere_Triangulation(path, 5810)
		triangulation = get_Sphere_Triangulation(path, num_tri_quad_pts)
	
	
	else:
		triangulation = create_Max_Spherical_Triangulation_with_Inclusions(Inclusion_Spherical_Coors, num_tri_quad_pts)

	num_vert = num_tri_quad_pts + len(Inclusion_Spherical_Coors)

	distance_mat = np.inf * np.ones(( num_vert, num_vert ))

	#print("triangulation.shape = "+str(triangulation.shape))

	for tri in range(len(triangulation)):

		vertex_1 = int(triangulation[tri, 0])
		vertex_2 = int(triangulation[tri, 1])
		vertex_3 = int(triangulation[tri, 2])		
		
		Coors_1 = Cart_Coors[vertex_1, :]
		Coors_2 = Cart_Coors[vertex_2, :]
		Coors_3 = Cart_Coors[vertex_3, :]

		dist_12 = euc_dist(Coors_1, Coors_2)
		dist_23 = euc_dist(Coors_2, Coors_3)
		dist_13 = euc_dist(Coors_1, Coors_3)

		distance_mat[vertex_1, vertex_2] = dist_12
		distance_mat[vertex_2, vertex_1] = dist_12

		distance_mat[vertex_2, vertex_3] = dist_23
		distance_mat[vertex_3, vertex_2] = dist_23

		distance_mat[vertex_1, vertex_3] = dist_13
		distance_mat[vertex_3, vertex_1] = dist_13			
	
	#scipy.sparse.csgraph.shortest_path
	distances = csgraph.dijkstra(distance_mat)

	#print("distances.shape = "+str(distances.shape))

	return distances, triangulation


# Euc Distance given cartesian coordinates:
def euc_dist(Cart_Coors_1, Cart_Coors_2):

	#print("Cart_Coors_1.shape = "+str(Cart_Coors_1.shape))

	return np.sqrt((Cart_Coors_1[0] - Cart_Coors_2[0])**2 + (Cart_Coors_1[1] - Cart_Coors_2[1])**2 + (Cart_Coors_1[2] - Cart_Coors_2[2])**2)


# We use this triangulation to create a distance function from our inclusions:
def create_Max_Spherical_Triangulation_with_Inclusions(Inclusion_Spherical_Coors, num_quad_plot_pts):

	theta_inclusion_coors, phi_inclusion_coors = np.hsplit(Inclusion_Spherical_Coors, 2)

	num_inclusions = len(theta_inclusion_coors)
	Cart_Coors_Inclusions = np.zeros(( num_inclusions, 3 ))

	for pt in range(num_inclusions):
		
		theta_pt = theta_inclusion_coors[pt, 0]
		phi_pt = phi_inclusion_coors[pt, 0]		

		Cart_Coors_Inclusions[pt, 0] = np.sin(phi_pt)*np.cos(theta_pt)
		Cart_Coors_Inclusions[pt, 1] = np.sin(phi_pt)*np.sin(theta_pt)
		Cart_Coors_Inclusions[pt, 2] = np.cos(phi_pt)

	Lbdv_Cart_Pts_and_Wt_Quad = leb_wr.Lebedev(num_quad_plot_pts)
	lbdv_coordinate_array = Lbdv_Cart_Pts_and_Wt_Quad[:,:-1]

	#print("lbdv_coordinate_array.shape = "+str(lbdv_coordinate_array.shape))
	#print("Cart_Coors_Inclusions.shape = "+str(Cart_Coors_Inclusions.shape))
	#print("np.array([0, 0, 0]).shape = "+str(np.array([[0, 0, 0]]).shape))
	

	lbdv_plus_center = np.append( np.append(lbdv_coordinate_array, Cart_Coors_Inclusions, axis=0), np.array([[0, 0, 0]]), axis=0 )
	delauney_tetras = Delaunay(lbdv_plus_center)

	tetras = delauney_tetras.simplices
	num_tris = len(delauney_tetras.simplices)

	delauney_tris = np.zeros(( num_tris, 3 ))

	for tri_i in range(num_tris):
		vert_ind = 0

		for tetra_vert in range(4):
			vertex = tetras[tri_i, tetra_vert]		

			if(vertex != (num_quad_plot_pts + num_inclusions) and vert_ind < 3):

				delauney_tris[tri_i, vert_ind] = vertex
				vert_ind = vert_ind + 1	

	return delauney_tris


# Create vtu of triangulated mesh, using values at lbdv points, add in possible scalar/vector fields
def write_surface_and_fields_to_VTU(path, surf_name, num_surf_pts, Surface_Values, Scalar_Field_Values_Arrays, Vector_Field_Values_Arrays, Scalar_Name_List, Vector_Name_List, Pickled_Image_Subfolder_Name = [],Specific_Triangulation = []):

	Spherical_Triangulation_Used = []

	# We can use this to add inclusions:
	if(Specific_Triangulation == []):
		Spherical_Triangulation_Used = get_Sphere_Triangulation(path, num_surf_pts)

	else:
		Spherical_Triangulation_Used = Specific_Triangulation
	
	
	num_triangles = len(Spherical_Triangulation_Used)

	Folder_Path = os.path.join(path, 'Pickled_Images')
	name = str(surf_name)+".vtp" # +str(Time_and_Date_Str())+

	# we add in option to create subfolder for specific runs:
	if(Pickled_Image_Subfolder_Name != []):	# if we are given subfolder to place this in
		Folder_Path = os.path.join(Folder_Path, Pickled_Image_Subfolder_Name)
	
		if not os.path.exists(Folder_Path): # create folder, if it does not exist yet
			os.makedirs(Folder_Path)

	filename = os.path.join(Folder_Path, name)
	
	num_scalar_fields = 0 
	num_vector_fields = 0 
	
	if(Scalar_Field_Values_Arrays != []):		
		num_scalar_fields = Scalar_Field_Values_Arrays.shape[1]

	if(Vector_Field_Values_Arrays != []):	
		num_vector_fields = Vector_Field_Values_Arrays.shape[2]	

	#print("num_scalar_fields = "+str(num_scalar_fields))
	#print("num_vector_fields = "+str(num_vector_fields))	

	# save vtu_file
	#name_out = "%s_L%d"%(fileName,int(self.L_plus))
	#name_vtu = name_out+'.vtu'
	#name_vtu = '%sSPH_surface_%5.5d.vtu' % (directoryName,Step);

	fieldData_list = []

	for s in range(num_scalar_fields):
	
		scalar_field_data = {}
		scalar_field_data['fieldName'] = Scalar_Name_List[s]
		scalar_field_data['fieldValues'] = Scalar_Field_Values_Arrays[:, s].T
		scalar_field_data['NumberOfComponents'] = 1;
		fieldData_list.append(scalar_field_data);
		
	for v in range(num_vector_fields):

		vector_field_data = {}
		vector_field_data['fieldName'] = Vector_Name_List[v]
		vector_field_data['fieldValues'] = np.squeeze(Vector_Field_Values_Arrays[:, :, v]).T
		vector_field_data['NumberOfComponents'] = 3;
		fieldData_list.append(vector_field_data);	

	#print("Surface_Values.shape = "+str(Surface_Values.shape))
	#print("\n"+"PLOTTING: "+str(filename)+"\n")

	write_vtp(filename, Surface_Values.T, fieldData_list, Spherical_Triangulation_Used, flagVerboseLevel=0,flagFieldDataMode=None)


# USES DICTIONARY INPUT: Create vtu of triangulated mesh, using values at lbdv points, add in possible scalar/vector fields
#BJG: CAREFUL WITH INPUT, currently use .vtk, but make sure .vtu or .vtp files have appropriate structure
def write_surface_and_fields_to_VTU_from_Dict(Vtu_Plot_Dict):
#path, surf_name, num_surf_pts, Surface_Values, Scalar_Field_Values_Arrays, Vector_Field_Values_Arrays, Scalar_Name_List, Vector_Name_List, Specific_Triangulation = []):

	My_Dir_Path = Vtu_Plot_Dict['My_Dir_Path'] # For finding pickled triangulations
	Pickled_Image_Subfolder_Name = Vtu_Plot_Dict['Pickled_Image_Subfolder_Name'] # for storing in subfolder, if not == []
	surf_name = Vtu_Plot_Dict['surf_name']
	num_surf_pts = Vtu_Plot_Dict['num_surf_pts']
	Surface_Values = Vtu_Plot_Dict['Surface_Values']
	Scalar_Field_Values_Arrays = Vtu_Plot_Dict['Scalar_Field_Values_Arrays']
	Vector_Field_Values_Arrays = Vtu_Plot_Dict['Vector_Field_Values_Arrays']
	Scalar_Name_List = Vtu_Plot_Dict['Scalar_Name_List']
	Vector_Name_List = Vtu_Plot_Dict['Vector_Name_List']
	Specific_Triangulation = Vtu_Plot_Dict['Specific_Triangulation']

	Spherical_Triangulation_Used = []

	# We can use this to add inclusions:
	if(Specific_Triangulation == []):
		Spherical_Triangulation_Used = get_Sphere_Triangulation(My_Dir_Path, num_surf_pts)

	else:
		Spherical_Triangulation_Used = Specific_Triangulation
	
	num_triangles = len(Spherical_Triangulation_Used)


	Folder_Path = os.path.join(My_Dir_Path, 'Pickled_Images')

	if(Pickled_Image_Subfolder_Name != []):	# if we are given subfolder to place this in
		Folder_Path = os.path.join(Folder_Path, Pickled_Image_Subfolder_Name)
	
		if not os.path.exists(Folder_Path): # create folder, if it does not exist yet
			os.makedirs(Folder_Path)		

	name = str(surf_name)+".vtu" # +str(Time_and_Date_Str())+
	filename = os.path.join(Folder_Path, name)

	
	num_scalar_fields = 0 
	num_vector_fields = 0 
	
	if(Scalar_Field_Values_Arrays != []):		
		num_scalar_fields = Scalar_Field_Values_Arrays.shape[1]

	if(Vector_Field_Values_Arrays != []):	
		num_vector_fields = Vector_Field_Values_Arrays.shape[2]	

	#print("num_scalar_fields = "+str(num_scalar_fields))
	#print("num_vector_fields = "+str(num_vector_fields))	

	# save vtu_file
	#name_out = "%s_L%d"%(fileName,int(self.L_plus))
	#name_vtu = name_out+'.vtu'
	#name_vtu = '%sSPH_surface_%5.5d.vtu' % (directoryName,Step);
	
	fid = open(filename, 'w')

	print >> fid, '<?xml version="1.0"?>'
	print >> fid, '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
	print >> fid, ''
	print >> fid, '  <UnstructuredGrid>'
	print >> fid, ''
	print >> fid, '<Piece NumberOfPoints="%d" NumberOfCells="%d">'%(num_surf_pts, num_triangles)
	print >> fid, ''
	print >> fid, '<PointData Scalars="Scalar Field(s)" Vectors="Vector Field(s)">'
	print >> fid, ''
	
	
	for sc_fld_i in range(num_scalar_fields):
	
		print >> fid, '<DataArray type="Float64" Name="%s" NumberOfComponents="1" format="ascii" >'%(Scalar_Name_List[sc_fld_i])  

		for k in range(0,num_surf_pts):
			print >> fid, '%.16e '%(np.float64(Scalar_Field_Values_Arrays[k, sc_fld_i]))
		print >> fid, '</DataArray>'
	
		print >> fid, ''


	for vec_fld_i in range(num_vector_fields):		
	
		print >> fid, '<DataArray type="Float64" Name="%s" NumberOfComponents="3" format="ascii" >'%(Vector_Name_List[vec_fld_i])
 
		for k in range(0,num_surf_pts):
			print >> fid, '%.16e %.16e %.16e '%(np.float64(Vector_Field_Values_Arrays[k, 0, vec_fld_i]),np.float64(Vector_Field_Values_Arrays[k, 1, vec_fld_i]), np.float64(Vector_Field_Values_Arrays[k, 2, vec_fld_i]))
		print >> fid, '</DataArray>'
		
		print >> fid, ''
	
	
	print >> fid, ' </PointData>'
	

	print >> fid, ''
	print >> fid, '<CellData>'
	print >> fid, '</CellData>'
	print >> fid, ''
	print >> fid, '<Points>'
	print >> fid, '<DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="-1" RangeMax="1">'
	
	for i in range(0,num_surf_pts):
		print >> fid, '%.16e %.16e %.16e '%(np.float64(Surface_Values[i, 0]),np.float64(Surface_Values[i, 1]),np.float64(Surface_Values[i, 2]))

	print >> fid, ''
	print >> fid, '</DataArray>'
	print >> fid, '</Points>'
	print >> fid, ''
	print >> fid, '<Cells>'
	print >> fid, '<DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="1e+299" RangeMax="-1e+299">'
	
	for i in range(0,num_triangles):
		print >> fid,'%d %d %d ' % (Spherical_Triangulation_Used[i][0],Spherical_Triangulation_Used[i][1],Spherical_Triangulation_Used[i][2])

	print >> fid, '</DataArray>'
	print >> fid, ''
	print >> fid, '<DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1e+299" RangeMax="-1e+299">'
	for i in range(0,num_triangles):
		print >> fid,'%d ' %((i + 1)*3);

	print >> fid, '</DataArray>'
	print >> fid, ''
	print >> fid, '<DataArray type="UInt8" Name="types" format="ascii" RangeMin="1e+299" RangeMax="-1e+299">'
	vtk_type = 5; # type for triangle
	
	for i in range(0,num_triangles):
		print >> fid,'%d ' %vtk_type;

	print >> fid, '</DataArray>'
	print >> fid, ''
	print >> fid, '</Cells>'
	print >> fid, ''
	print >> fid, '</Piece>'
	print >> fid, ''
	print >> fid, '  </UnstructuredGrid>'
	print >> fid, ''
	print >> fid, '</VTKFile>'

	fid.close()


#BJG: CAREFUL WITH INPUT, currently use .vtk, but make sure .vtu or .vtp files have appropriate structure
def write_vtp(vtp_filename, points, fieldData_list, triangles = [],flagVerboseLevel=0,flagFieldDataMode=None):
  ambient_num_dim =3; # assumed data is embedded in R^3

  if flagFieldDataMode == None:
    if isinstance(fieldData_list, list) == True:
      flagFieldDataMode='list';
    elif isinstance(fieldData_list, dict) == True:
      flagFieldDataMode='dict';

  # Check parameter types are correct given the flags
  flagFieldTypeErr = False;
  if (flagFieldDataMode == 'list') and (isinstance(fieldData_list, list) == False):
    flagFieldTypeErr = True;
    type_str = str(list);
  elif (flagFieldDataMode == 'dict') and (isinstance(fieldData_list, dict) == False):
    flagFieldTypeErr = True;
    type_str = str(dict);

  if flagFieldTypeErr == True:
    err_s = "";
    err_s += "Expected that the fieldData_list is of type '%s'. \n"%type_str;
    err_s += "May need to adjust and set the flagFieldDataMode. \n";
    err_s += "flagFieldDataMode = %s\n"%str(flagFieldDataMode);
    err_s += "type(fieldData_list) = %s\n"%str(type(fieldData_list));
    raise Exception(err_s);

  # Output a VTP file with the fields
  #
  # record the data for output
  #N_list = np.shape(ptsX)[0];  
  vtpData = vtk.vtkPolyData();

  # check format of the points array
  #n1 = np.size(points_data,0);
  #if (n1 == 0): # we assume array already flat
  #  # nothing to do
  #else:
  #  points = points_data.T.flatten();

  # setup the points data
  Points = vtk.vtkPoints();
  
  #numPoints = int(len(points)/ambient_num_dim);
  numPoints=np.size(points,1);
  for I in range(numPoints): 
    Points.InsertNextPoint(points[0,I], points[1,I], points[2,I]);
  vtpData.SetPoints(Points);

  #BJG: adding triangulation possibility:
  if(triangles != []):
    Triangles = vtk.vtkCellArray();
    numTriangles=np.size(triangles, 0);

    for I in range(numTriangles):
      Triangle_I = vtk.vtkTriangle();
      Triangle_I.GetPointIds().SetId(0, int(triangles[I, 0]) );
      Triangle_I.GetPointIds().SetId(1, int(triangles[I, 1]) );
      Triangle_I.GetPointIds().SetId(2, int(triangles[I, 2]) );
      Triangles.InsertNextCell(Triangle_I);

    vtpData.SetPolys(Triangles)
 
  # Get data from the vtu object
  #print("Getting data from the vtu object.");
  #nodes_vtk_array = vtuData.GetPoints().GetData();
  #ptsX            = vtk_to_numpy(nodes_vtk_array);

  # -- setup data arrays
  if fieldData_list != None: # if we have field data

    numFields = len(fieldData_list);
    if flagVerboseLevel==1:
      print("numFields = " + str(numFields));

    if flagFieldDataMode=='list':
      f_list = fieldData_list;
    elif flagFieldDataMode=='dict': 
      # convert dictionary to a list
      f_list=[];
      for k, v in fieldData_list.items():
        f_list.append(v);

    for fieldData in f_list:
      #print("fieldData = " + str(fieldData));
      fieldName = fieldData['fieldName'];
      if flagVerboseLevel==1:
        print("fieldName = " + str(fieldName));
      fieldValues = fieldData['fieldValues'];
      NumberOfComponents = fieldData['NumberOfComponents'];

      if (NumberOfComponents == 1):
        N_list     = len(fieldValues);
        if flagVerboseLevel==1:
          print("N_list = " + str(N_list));
        atzDataPhi = vtk.vtkDoubleArray();
        atzDataPhi.SetNumberOfComponents(1);
        atzDataPhi.SetName(fieldName);
        atzDataPhi.SetNumberOfTuples(N_list);
        #print("fieldName = "+str(fieldName))
        for I in np.arange(0,N_list):
          #print("I = "+str(I))
          #print("fieldValues[I] = "+str(fieldValues[I]))
          atzDataPhi.SetValue(I,fieldValues[I]);
        #vtpData.GetPointData().SetScalars(atzDataPhi);
        vtpData.GetPointData().AddArray(atzDataPhi);
      elif (NumberOfComponents == 3): 
        #print(fieldValues);
        #print("fieldValues.shape = "+str(fieldValues.shape));
        N_list   = fieldValues.shape[1];
        if flagVerboseLevel==1:
          print("N_list = " + str(N_list));
        atzDataV = vtk.vtkDoubleArray();
        atzDataV.SetNumberOfComponents(3);
        atzDataV.SetName(fieldName);
        atzDataV.SetNumberOfTuples(N_list);
        #print("fieldName = "+str(fieldName))
        for I in np.arange(0,N_list):
          #print("fieldValues[0,I] = "+str(fieldValues[0,I]))
          #print("fieldValues[1,I] = "+str(fieldValues[1,I]))
          #print("fieldValues[2,I] = "+str(fieldValues[2,I]))
          atzDataV.SetValue(I*ambient_num_dim + 0,fieldValues[0,I]);
          atzDataV.SetValue(I*ambient_num_dim + 1,fieldValues[1,I]);
          atzDataV.SetValue(I*ambient_num_dim + 2,fieldValues[2,I]);
        #vtpData.GetPointData().SetVectors(atzDataV);
        vtpData.GetPointData().AddArray(atzDataV);

      else:
        #print("ERROR: " + error_code_file + ":" + error_func);
        s = "";
        s += "NumberOfComponents invalid. \n";
        s += "NumberOfComponents = " + str(NumberOfComponents);
        raise Exception(s);

        #exit(1);

  #vtuData.GetPointData().SetVectors(atzDataVec);
  #vtuData.GetPointData().AddArray(atzDataScalar1);
  #vtuData.GetPointData().AddArray(atzDataScalar2);
  #vtuData.GetPointData().AddArray(atzDataVec1);
  #vtuData.GetPointData().AddArray(atzDataVec2);

  # write the XML file
  writerVTP = vtk.vtkXMLPolyDataWriter();
  writerVTP.SetFileName(vtp_filename);
  writerVTP.SetInputData(vtpData);
  writerVTP.SetCompressorTypeToNone(); # help ensure ascii output (as opposed to binary)
  writerVTP.SetDataModeToAscii(); # help ensure ascii output (as opposed to binary)
  writerVTP.Write();   
  #writerVTP.Close();

# Loads a VTP file into python data structures.
#
# fieldNames: collection of fields to load, if None loads all. 

