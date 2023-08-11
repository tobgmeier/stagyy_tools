import numpy as np
from stagpy import stagyydata
from stagpy import rprof
from stagpy import field  
import math

###Definition of Object Annulus_Slice###
class Annulus_Slice:
    def __init__(self, r_min, r_max, phi_min, phi_max):
        self.r_min = r_min 
        self.r_max = r_max
        self.phi_min = phi_min
        self.phi_max = phi_max



    def area_selection(self, r_mesh, phi_mesh):

        r_min = self.r_min
        r_max = self.r_max
        phi_min = self.phi_min
        phi_max = self.phi_max

        if phi_min > 0.0:
            return np.logical_not(np.logical_and(np.logical_and(phi_mesh < phi_max, phi_mesh > phi_min),np.logical_and(r_mesh < r_max, r_mesh > r_min)))

        elif (phi_min < 0.0):
            return np.logical_and(phi_mesh > phi_max, phi_mesh < (2*math.pi-abs(phi_min)),np.logical_not(np.logical_and(r_mesh < r_max, r_mesh > r_min)))


    def meshfield_mask(self, r_mesh, p_mesh,field_mesh):
        area_select = self.area_selection(r_mesh=r_mesh, phi_mesh=p_mesh)
        r_mesh = np.ma.masked_where(area_select, r_mesh)
        p_mesh = np.ma.masked_where(area_select, p_mesh)
        field_mesh = np.ma.masked_where(area_select, field_mesh)
        return r_mesh, p_mesh, field_mesh

    def mesh_mask(self, r_mesh, p_mesh,field_mesh):
        area_select = self.area_selection(r_mesh=r_mesh, phi_mesh=p_mesh)
        r_mesh = np.ma.masked_where(area_select, r_mesh)
        p_mesh = np.ma.masked_where(area_select, p_mesh)
        return r_mesh, p_mesh

    def field_mask(self, r_mesh, p_mesh,field_mesh):
        area_select = self.area_selection(r_mesh=r_mesh, phi_mesh=p_mesh)
        field_mesh = np.ma.masked_where(area_select, field_mesh)
        return field_mesh


    def radial_profile(self, r_mesh,p_mesh, field_mesh, n_bins=1000):
        r_mesh, p_mesh, field_mesh = self.meshfield_mask(r_mesh, p_mesh, field_mesh)
        field_profile = []
        r_profile = []
        for i in range(0,n_bins-1):
            r_step = np.ma.max(r_mesh[:,i])
            fld = field_mesh[:,i]
            field_profile.append(np.ma.mean(fld))

            r_profile.append(r_step)

            i = i+1

        field_profile = np.asarray(field_profile)
        r_profile = np.asarray(r_profile)
        return r_profile, field_profile

###End of Object Annulus_Slice###

narrower = 16
####Dictionary with useful Slices######
slices_dict = { 'total': Annulus_Slice(r_min=0, r_max=math.inf, phi_min=-0.1, phi_max=6.4), 
                'day-side': Annulus_Slice(0,math.inf,math.pi/2,3*math.pi/2),
                'night-side': Annulus_Slice(0,math.inf,-math.pi/2,math.pi/2), 
                'subsolar': Annulus_Slice(0,math.inf, 3*math.pi/8, 5*math.pi/8),
                'antisolar': Annulus_Slice(0,math.inf, -math.pi/8, math.pi/8),
                'narrow-day-side': Annulus_Slice(0,math.inf,math.pi/2+math.pi/narrower,3*math.pi/2-math.pi/narrower),
                'narrow-night-side': Annulus_Slice(0,math.inf,-math.pi/2+math.pi/narrower,math.pi/2-math.pi/narrower),
                'annulus-border': Annulus_Slice(0,math.inf,-2*math.pi/256,2*math.pi/256),
                'box': Annulus_Slice(-math.inf,math.inf,0, math.inf)
                }