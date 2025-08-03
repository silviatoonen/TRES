from amuse.units import units, constants, quantities
import numpy as np
import sys 
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

from TRES_options import REPORT_BINARY_EVOLUTION, REPORT_FUNCTION_NAMES, REPORT_MASS_TRANSFER_STABILITY
from TRES_options import density_BA_in_TSMT, c_s_BA_in_TSMT, max_iter_TSMT, eps_TSMT
from TRES_options import INCLUDE_GDF_IN_TSMT, model_I_GDF, INCLUDE_ECC_GDF_IN_TSMT, INCLUDE_CBD_IN_TSMT, INCLUDE_RETROGRADE_CBD_IN_TSMT, INCLUDE_OUTFLOW_CBD_IN_TSMT, INCLUDE_HYDR_IN_TSMT, hydro_drag_coefficient_in_TSMT
from TRES_options import minimum_time_step

#constants
numerical_error  = 1.e-6
small_numerical_error  = 1.e-10
minimum_eccentricity = 1.e-5

const_common_envelope_efficiency = 4.0 #1.0, 4 for now for easier testing with SeBa
const_envelope_structure_parameter = 0.5
const_common_envelope_efficiency_gamma = 1.75

stellar_types_compact_objects = [10,11,12,13,14]|units.stellar_type
stellar_types_giants = [2,3,4,5,6,8,9]|units.stellar_type
stellar_types_planetary_objects = [18,19]|units.stellar_type # planets & brown dwarfs

stellar_types_SN_remnants = [13,14,15]|units.stellar_type # remnant types created through a supernova
stellar_types_remnants = [7,8,9,10,11,12,13,14,15]|units.stellar_type
stellar_types_dr = [2,4,7,8,9,10,11,12,13,14,15]|units.stellar_type #stars which go through a instantaneous radius change at formation; hertzsprung gap stars (small envelope perturbation) + horizontal branch stars + remnants



#q_crit = 3.
#q_crit_giants_conv_env = 0.9
nucleair_efficiency = 0.007 # nuc. energy production eff, Delta E = 0.007 Mc^2



    

#dictionaries
bin_type = {    
                'unknown': 'unknown',       
                'merger': 'merger', 
                'disintegrated': 'disintegrated', 
                'dyn_inst': 'dynamical_instability', 

                'detached': 'detached',       
                'contact': 'contact',    
                'collision': 'collision',    
                'semisecular': 'semisecular',    
                'rlof': 'rlof',   #only used for stopping conditions
                'olof' : 'olof',  #only used for stopping conditions
 
                'stable_mass_transfer': 'stable_mass_transfer',
                'common_envelope': 'common_envelope',     
                'common_envelope_energy_balance': 'common_envelope_energy_balance',     
                'common_envelope_angular_momentum_balance': 'common_envelope_angular_momentum_balance',
                'double_common_envelope': 'double_common_envelope',                
                
            }            

#-------------------------
#general functions
def roche_radius_dimensionless(M, m):
    # Assure that the q is calculated in identical units.
    unit = M.unit
    # and that q itself has no unit
    q = M.value_in(unit)/m.value_in(unit)
    q13 =  q**(1./3.)
    q23 =  q13**2
    return  0.49*q23/(0.6*q23 + np.log(1 + q13))

def roche_radius(bin, primary, self):
    if not bin.is_star and primary.is_star:
        return bin.semimajor_axis * roche_radius_dimensionless(primary.mass, self.get_mass(bin)-primary.mass)

    sys.exit('error in roche radius: Roche radius can only be determined in a binary')

def L2_radius_dimensionless(M,m):
    # approximation for l2 overflow
    # see Marchant+ 2016 equation 2
    q = M/m
    rl1 = roche_radius_dimensionless(M, m)
    rl2_div_rl1 = 0.299 * np.arctan(1.83*q**0.397) + 1 
    return rl2_div_rl1 * rl1

def L2_radius(bin, primary, self):
    # note: this prescription is based on the Eggleton approximation for how to adjust a circular RL to an eccentric one
    # may not be consistent with Sepinsky's method for eccentric RL (L1)
    if not bin.is_star and primary.is_star:
        return bin.semimajor_axis * L2_radius_dimensionless(primary.mass, self.get_mass(bin)-primary.mass)*(1-bin.eccentricity)
    sys.exit('Error: L2 radius can only be determined in a binary')

#for comparison with kozai timescale
def stellar_evolution_timescale(star):
    if REPORT_FUNCTION_NAMES:
        print("Stellar evolution timescale")
        
    if star.stellar_type in [0,1,7]|units.stellar_type:
        return (0.1 * star.mass * nucleair_efficiency * constants.c**2 / star.luminosity).in_(units.Gyr)
    elif star.stellar_type in stellar_types_compact_objects:
        return np.inf|units.Myr 
    elif star.stellar_type in stellar_types_planetary_objects:
        return np.inf|units.Myr 
    else:        
        return 0.1*star.age


#for mass transfer rate
def nuclear_evolution_timescale(star):
    if REPORT_FUNCTION_NAMES:
        print("Nuclear evolution timescale:")
        
    if star.stellar_type in [0,1,7]|units.stellar_type:
        return (0.1 * star.mass * nucleair_efficiency * constants.c**2 / star.luminosity).in_(units.Gyr)
    elif star.stellar_type in stellar_types_planetary_objects:
#        print('nuclear evolution timescale for planetary objects requested')
#        return np.inf|units.Myr         
        return dynamic_timescale(star)
    else: #t_nuc ~ delta t * R/ delta R, other prescription gave long timescales in SeBa which destables the mass transfer
        if star.time_derivative_of_radius <= (quantities.zero+numerical_error**2)|units.RSun/units.yr:
        #when star is shrinking
#            t_nuc = 0.1*main_sequence_time() # in SeBa
            t_nuc = 0.1*star.age         
        else: 
            t_nuc = star.radius / star.time_derivative_of_radius #does not include the effect of mass loss on R

        return t_nuc

def kelvin_helmholds_timescale(star):
    if star.stellar_type in stellar_types_planetary_objects:
#        print('thermal evolution timescale for planetary objects requested')
        return dynamic_timescale(star)

    if REPORT_FUNCTION_NAMES:
        print("KH timescale:", (constants.G*star.mass**2/star.radius/star.luminosity).in_(units.Myr))
    return constants.G*star.mass**2/star.radius/star.luminosity

def dynamic_timescale(star):
    if REPORT_FUNCTION_NAMES:
        print("Dynamic timescale:", (np.sqrt(star.radius**3/star.mass/constants.G)[0]).in_(units.yr))
    return np.sqrt(star.radius**3/star.mass/constants.G)   
    
def corotating_spin_angular_frequency_binary(semi, m1, m2):
    return 1./np.sqrt(semi**3/constants.G / (m1+m2))

#Hurley, Pols en Tout 2000, eq 107-108
def lang_spin_angular_frequency(star):
    v_rot = 330*star.mass.value_in(units.MSun)**3.3/(15.0+star.mass.value_in(units.MSun)**3.45)
    w = 45.35 * v_rot/star.radius.value_in(units.RSun)
    return w|1./units.yr

def break_up_angular_frequency(object):
    return np.sqrt( constants.G * object.mass / object.radius ) / object.radius


def criticial_angular_frequency_CHE(m, Z):
    #angular frequency of spin for CHE threshold
    #Fitting formula for CHE from Riley+ 2021

    a_coeff = np.array([5.7914 * 10 ** - 4, -1.9196 * 10 ** - 6,
                        -4.0602 * 10 ** - 7, 1.0150 * 10 ** - 8,
                        -9.1792 * 10 ** - 11, 2.9051 * 10 ** - 13])
    mass_power = np.linspace(0,5,6)
    omega_at_z_0d004 = np.sum(a_coeff * m.value_in(units.MSun)** mass_power / m.value_in(units.MSun) ** 0.4)
    omega_crit = omega_at_z_0d004/(0.09 * np.log(Z/0.004) + 1) |1./units.s
    return omega_crit


def copy_outer_orbit_to_inner_orbit(bs, self):
    if REPORT_FUNCTION_NAMES:
        print('Copy_outer_orbit_to_inner_orbit')

    if self.is_triple():
        bs.semimajor_axis = self.triple.semimajor_axis
        bs.eccentricity = self.triple.eccentricity
        bs.argument_of_pericenter = self.triple.argument_of_pericenter
        bs.longitude_of_ascending_node = self.triple.longitude_of_ascending_node
        bs.mass_transfer_rate = self.triple.mass_transfer_rate
        bs.accretion_efficiency_mass_transfer, = self.triple.accretion_efficiency_mass_transfer,
        bs.accretion_efficiency_wind_child1_to_child2, = self.triple.accretion_efficiency_wind_child1_to_child2,
        bs.accretion_efficiency_wind_child2_to_child1, = self.triple.accretion_efficiency_wind_child2_to_child1,
        bs.specific_AM_loss_mass_transfer, = self.triple.specific_AM_loss_mass_transfer,
        bs.is_mt_stable = self.triple.is_mt_stable
    
        self.triple.semimajor_axis = 1e100|units.RSun
        self.triple.eccentricity = 0
        
        
        
        
def copy_outer_star_to_accretor(self):
    if REPORT_FUNCTION_NAMES:
        print('Copy_outer_star_to_accretor')

    if self.is_triple():        
        if self.triple.child1.is_star:
            tertiary_star = self.triple.child1
            bs = self.triple.child2
        else:
            tertiary_star = self.triple.child2 
            bs = self.triple.child1
            
        if not bs.child1.is_donor:
            bs.child1 = tertiary_star
        else:
            bs.child2 = tertiary_star

    
#-------------------------

#-------------------------
# functions for mass transfer in a binary

def perform_inner_collision(self):
    if self.is_triple():
        if self.triple.child1.is_star:
            self.triple.child2
        else:
            self.triple.child1
        
        # smaller star is added to big star
        if bs.child1.radius >= bs.child2.radius:
            donor = bs.child1
            accretor = bs.child2
        else:
            donor = bs.child2
            accretor = bs.child1
    
        donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]                
        accretor_in_stellar_code = accretor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]                
    
        #no additional mass and Jspin loss from merged object for now
        J_spin_donor_previous = self.spin_angular_momentum(donor)
        J_spin_accretor_previous = self.spin_angular_momentum(accretor)
        J_orbit = self.orbital_angular_momentum(bs)
        J_spin_new = J_spin_donor_previous + J_spin_accretor_previous + J_orbit
    
        #merger
        donor_in_stellar_code.merge_with_other_star(accretor_in_stellar_code) 
#    self.stellar_code.evolve_model(minimum_time_step) #to get updates radii, not just inflation of stars due to accretion #silvia
        self.copy_from_stellar()
              
        donor.moment_of_inertia_of_star = self.moment_of_inertia(donor)        
    
        #assuming conservation of total angular momentum of the inner binary
        spin_angular_frequency = J_spin_new / donor.moment_of_inertia_of_star                
        critical_spin_angular_frequency = np.sqrt(constants.G * donor.mass/donor.radius**3)
        donor.spin_angular_frequency = min(donor.spin_angular_frequency, critical_spin_angular_frequency)

        self.stellar_code.particles.remove_particle(accretor)
        accretor.mass = 0|units.MSun # necessary for adjust_system_after_ce_in_inner_binary
        #adjust outer orbit, needs to be before the system becomes a binary
        #and copy to inner orbit 
        # weird structure necessary for secular code -> outer orbit is redundant
        adjust_system_after_ce_in_inner_binary(bs, self)                    
        copy_outer_orbit_to_inner_orbit(bs, self)
        copy_outer_star_to_accretor(self)
        #functions are skipped in binaries, needs to be checked if this works well
        
        self.secular_code.parameters.ignore_tertiary = True
        self.secular_code.parameters.check_for_dynamical_stability = False
        self.secular_code.parameters.check_for_outer_collision = False
        self.secular_code.parameters.check_for_outer_RLOF = False

        bs.bin_type = bin_type['collision']         
        self.save_snapshot() # just to note that it the system has merged

#use of stopping condition in this way (similar to perform inner merger) is not necessary. 
#TRES.py takes care of it
#        if self.check_stopping_conditions_stellar_interaction()==False:
#            print('stopping conditions stellar interaction')
#            return False
    
        self.check_RLOF()       
        if self.has_donor():
            print(self.triple.child2.child1.mass, self.triple.child2.child2.mass, self.triple.child2.semimajor_axis, self.triple.child2.eccentricity, self.triple.child2.child1.is_donor, self.triple.child2.child2.is_donor)
            print(self.triple.child1.mass, self.triple.semimajor_axis, self.triple.eccentricity, self.triple.child1.is_donor)
            sys.exit("error in adjusting triple after collision: RLOF")
            
        donor.is_donor = False
        bs.is_mt_stable = True
        bs.bin_type = bin_type['detached']

        donor.spin_angular_frequency = corotating_spin_angular_frequency_binary(bs.semimajor_axis, bs.child1.mass, bs.child2.mass)

#use of stopping condition in this way (similar to perform inner merger) is not necessary
#TRES.py takes care of it
#        return True 
        
def perform_inner_merger(bs, donor, accretor, self):
    if REPORT_BINARY_EVOLUTION:
        print('Merger in inner binary through common envelope phase')

    donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]                
    accretor_in_stellar_code = accretor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]                
    
    #no additional mass and Jspin loss from merged object for now
    J_spin_donor_previous = self.spin_angular_momentum(donor)
    J_spin_accretor_previous = self.spin_angular_momentum(accretor)
    J_orbit = self.orbital_angular_momentum(bs)
    J_spin_new = J_spin_donor_previous + J_spin_accretor_previous + J_orbit
            
    #merger
    donor_in_stellar_code.merge_with_other_star(accretor_in_stellar_code) 
#    self.stellar_code.evolve_model(minimum_time_step) #to get updates radii, not just inflation of stars due to accretion #silvia
    self.copy_from_stellar()
    
    donor.moment_of_inertia_of_star = self.moment_of_inertia(donor)        
    
    #assuming conservation of total angular momentum of the inner binary
    spin_angular_momentum = J_spin_new / donor.moment_of_inertia_of_star                
    critical_spin_angular_frequency = np.sqrt(constants.G * donor.mass/donor.radius**3)
    donor.spin_angular_frequency = min(donor.spin_angular_frequency, critical_spin_angular_frequency)
        
    self.stellar_code.particles.remove_particle(accretor)
    accretor.mass = 0|units.MSun # necessary for adjust_system_after_ce_in_inner_binary   
    #adjust outer orbit, needs to be before the system becomes a binary
    #and copy to inner orbit 
    # weird structure necessary for secular code -> outer orbit is redundant
    adjust_system_after_ce_in_inner_binary(bs, self)                    
    copy_outer_orbit_to_inner_orbit(bs, self)
    copy_outer_star_to_accretor(self)
    #functions are skipped in binaries, needs to be checked if this works well
    
    self.secular_code.parameters.ignore_tertiary = True
    self.secular_code.parameters.check_for_dynamical_stability = False
    self.secular_code.parameters.check_for_outer_collision = False
    self.secular_code.parameters.check_for_outer_RLOF = False
    bs.bin_type = bin_type['merger']    
    self.save_snapshot() # just to note that the system has merged

    if self.check_stopping_conditions_stellar_interaction()==False:
        print('stopping conditions stellar interaction')
        return False
    else: 
        return True 
    
#    print(self.secular_code.give_roche_radii(self.triple),)
#    print(roche_radius(self.triple.child2, self.triple.child2.child1, self), roche_radius(self.triple.child2, self.triple.child2.child2,  self))
#
#    print(donor.spin_angular_frequency, corotating_spin_angular_frequency_binary(bs.semimajor_axis, bs.child1.mass, bs.child2.mass), critical_spin_angular_frequency)
#    donor.spin_angular_frequency = corotating_spin_angular_frequency_binary(bs.semimajor_axis, bs.child1.mass, bs.child2.mass)



def common_envelope_efficiency(donor, accretor):
    return const_common_envelope_efficiency

def envelope_structure_parameter(donor):
    return const_envelope_structure_parameter
    
def common_envelope_efficiency_gamma(donor, accretor):
    return const_common_envelope_efficiency_gamma
    
    

# ang.mom balance: \Delta J = \gamma * J * \Delta M / M
# See Eq. 5 of Nelemans VYPZ 2000, 360, 1011 A&A
def common_envelope_angular_momentum_balance(bs, donor, accretor, self):
    if REPORT_FUNCTION_NAMES:
        print('Common envelope angular momentum balance')

    if REPORT_BINARY_EVOLUTION:
        if bs.eccentricity > 0.05:
            print('gamma common envelope in eccentric binary')
        print('Before common envelope angular momentum balance' )
        self.print_binary(bs) 

    bs.bin_type = bin_type['common_envelope_angular_momentum_balance']
    self.save_snapshot()        

    gamma = common_envelope_efficiency_gamma(donor, accretor)
    J_init = np.sqrt(bs.semimajor_axis) * (donor.mass * accretor.mass) / np.sqrt(donor.mass + accretor.mass) * np.sqrt(1-bs.eccentricity**2)
    J_f_over_sqrt_a_new = (donor.core_mass * accretor.mass) / np.sqrt(donor.core_mass + accretor.mass)
    J_lost = gamma * (donor.mass-donor.core_mass) * J_init/(donor.mass + accretor.mass)
    sqrt_a_new = max(0.|units.RSun**0.5, (J_init -J_lost)/J_f_over_sqrt_a_new)
    a_new = pow(sqrt_a_new, 2)

    Rl_donor_new = roche_radius_dimensionless(donor.core_mass, accretor.mass)*a_new
    Rl_accretor_new = roche_radius_dimensionless(accretor.mass, donor.core_mass)*a_new    
    if REPORT_BINARY_EVOLUTION:
        print('donor:', donor.radius, donor.core_radius, Rl_donor_new)
        print('accretor:', accretor.radius, accretor.core_radius, Rl_accretor_new)
              
    if (donor.core_radius > Rl_donor_new) or (accretor.radius > Rl_accretor_new):
        stopping_condition = perform_inner_merger(bs, donor, accretor, self)
        if not stopping_condition: #stellar interaction
            return False                                                           
    else:
        donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
        #reduce_mass not subtrac mass, want geen adjust_donor_radius
        #check if star changes type     
        donor_in_stellar_code.change_mass(-1*(donor.mass-donor.core_mass+(small_numerical_error|units.MSun)), 0.|units.yr) 
        self.copy_from_stellar()
        
        donor.moment_of_inertia_of_star = self.moment_of_inertia(donor)        
        accretor.moment_of_inertia_of_star = self.moment_of_inertia(accretor)        

        bs.semimajor_axis = a_new
        bs.eccentricity = minimum_eccentricity

        #set to synchronization
        corotating_frequency = corotating_spin_angular_frequency_binary(a_new, donor.mass, accretor.mass)
        donor.spin_angular_frequency = corotating_frequency
        accretor.spin_angular_frequency = corotating_frequency
        
        self.check_RLOF()       
        if self.has_donor():
            print(self.triple.child2.child1.mass, self.triple.child2.child2.mass, self.triple.child2.child1.radius, self.triple.child2.child2.radius,self.triple.child2.semimajor_axis, self.triple.child2.eccentricity, self.triple.child2.child1.is_donor, self.triple.child2.child2.is_donor)
            print(self.triple.child2.child1.core_mass, self.triple.child2.child1.mass-self.triple.child2.child1.core_mass, self.triple.child2.child1.stellar_type)
            print(self.triple.child1.mass, self.triple.semimajor_axis, self.triple.eccentricity, self.triple.child1.is_donor)

#           sys.exit("error in adjusting triple after gamma CE: RLOF")
            stopping_condition = perform_inner_merger(bs, donor, accretor, self)
            if not stopping_condition: #stellar interaction
                return False

#        adjusting of stellar system
#        in previous case of merger, the adjustment is done there as mass may be lost during the merger
        adjust_system_after_ce_in_inner_binary(bs, self) 
                           

        
    donor.is_donor = False
    bs.is_mt_stable = True
    bs.bin_type = bin_type['detached']
    self.instantaneous_evolution = True #skip secular evolution    
    
    return True
    
#Following Webbink 1984
def common_envelope_energy_balance(bs, donor, accretor, self):
    if REPORT_FUNCTION_NAMES:
        print('Common envelope energy balance')

    if REPORT_BINARY_EVOLUTION:
        print('Before common envelope energy balance' )
        self.print_binary(bs) 

    bs.bin_type = bin_type['common_envelope_energy_balance']                
    self.save_snapshot()        

    alpha = common_envelope_efficiency(donor, accretor) 
    lambda_donor = envelope_structure_parameter(donor)

    Rl_donor = roche_radius(bs, donor, self)
    donor_radius = min(donor.radius, Rl_donor)

    #based on Glanz & Perets 2021  2021MNRAS.507.2659G
    #eccentric CE -> end result depends on pericenter distance more than semi-major axis
    pericenter_init =  bs.semimajor_axis * (1-bs.eccentricity)
    orb_energy_new = donor.mass * (donor.mass-donor.core_mass) / (alpha * lambda_donor * donor_radius) + donor.mass * accretor.mass/2/pericenter_init
    a_new = donor.core_mass * accretor.mass / 2 / orb_energy_new
#    a_new = bs.semimajor_axis * (donor.core_mass/donor.mass) / (1. + (2.*(donor.mass-donor.core_mass)*bs.semimajor_axis/(alpha_lambda*donor_radius*accretor.mass)))
    
    Rl_donor_new = roche_radius_dimensionless(donor.core_mass, accretor.mass)*a_new
    Rl_accretor_new = roche_radius_dimensionless(accretor.mass, donor.core_mass)*a_new    
    if REPORT_BINARY_EVOLUTION:
        print('donor:', donor.radius, donor.core_radius, Rl_donor_new)
        print('accretor:', accretor.radius, accretor.core_radius, Rl_accretor_new)
    
    if (donor.core_radius > Rl_donor_new) or (accretor.radius > Rl_accretor_new):
        stopping_condition = perform_inner_merger(bs, donor, accretor, self)
        if not stopping_condition: #stellar interaction
            return False                                                           
    else:
        donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
        #reduce_mass not subtrac mass, want geen adjust_donor_radius
        #check if star changes type     
        donor_in_stellar_code.change_mass(-1*(donor.mass-donor.core_mass+(small_numerical_error|units.MSun)), 0.|units.yr)    
        self.copy_from_stellar()

        donor.moment_of_inertia_of_star = self.moment_of_inertia(donor)        
        accretor.moment_of_inertia_of_star = self.moment_of_inertia(accretor)        

        bs.semimajor_axis = a_new
        bs.eccentricity = minimum_eccentricity

        #set to synchronization
        corotating_frequency = corotating_spin_angular_frequency_binary(a_new, donor.mass, accretor.mass)
        donor.spin_angular_frequency = corotating_frequency
        accretor.spin_angular_frequency = corotating_frequency


        self.check_RLOF()       
        if self.has_donor():
            print(self.triple.child2.child1.mass, self.triple.child2.child2.mass, self.triple.child2.semimajor_axis, self.triple.child2.eccentricity, self.triple.child2.child1.is_donor, self.triple.child2.child2.is_donor)
            print(self.triple.child1.mass, self.triple.semimajor_axis, self.triple.eccentricity, self.triple.child1.is_donor)
    
    #        sys.exit("error in adjusting triple after alpha CE: RLOF")
            stopping_condition = perform_inner_merger(bs, donor, accretor, self)
            if not stopping_condition: #stellar interaction
               return False

#        adjusting of stellar system
#        in previous case of merger, the adjustment is done there as mass may be lost during the merger
        adjust_system_after_ce_in_inner_binary(bs, self)                    
        
    donor.is_donor = False
    bs.is_mt_stable = True
    bs.bin_type = bin_type['detached']
    self.instantaneous_evolution = True #skip secular evolution                
    return True

# See appendix of Nelemans YPZV 2001, 365, 491 A&A
def double_common_envelope_energy_balance(bs, donor, accretor, self):
    if REPORT_FUNCTION_NAMES:
        print('Double common envelope energy balance')

    if REPORT_BINARY_EVOLUTION:
        print('Before double common envelope energy balance' )
        self.print_binary(bs) 

    bs.bin_type = bin_type['double_common_envelope']                
    self.save_snapshot()        

    alpha = common_envelope_efficiency(donor, accretor)
    lambda_donor = envelope_structure_parameter(donor) 
    lambda_accretor = envelope_structure_parameter(accretor)

    Rl_donor = roche_radius(bs, donor, self)
    donor_radius = min(donor.radius, Rl_donor)
    accretor_radius = accretor.radius
    
    
    #based on Glanz & Perets 2021  2021MNRAS.507.2659G
    #eccentric CE -> end result depends on pericenter distance more than semi-major axis
    pericenter_init =  bs.semimajor_axis * (1-bs.eccentricity)
    orb_energy_new = donor.mass * (donor.mass-donor.core_mass) / (alpha * lambda_donor * donor_radius) + accretor.mass * (accretor.mass-accretor.core_mass) / (alpha * lambda_accretor * accretor_radius) + donor.mass * accretor.mass/2/pericenter_init
    a_new = donor.core_mass * accretor.core_mass / 2 / orb_energy_new

    Rl_donor_new = roche_radius_dimensionless(donor.core_mass, accretor.core_mass)*a_new
    Rl_accretor_new = roche_radius_dimensionless(accretor.core_mass, donor.core_mass)*a_new    
    if REPORT_BINARY_EVOLUTION:
        print('donor:', donor.radius, donor.core_radius, Rl_donor_new)
        print('accretor:', accretor.radius, accretor.core_radius, Rl_accretor_new)
    
    if (donor.core_radius > Rl_donor_new) or (accretor.core_radius > Rl_accretor_new):
        stopping_condition = perform_inner_merger(bs, donor, accretor, self)
        if not stopping_condition: #stellar interaction
            return False                                                           
    else:
        #reduce_mass not subtrac mass, want geen adjust_donor_radius
        #check if star changes type     

        donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
        donor_in_stellar_code.change_mass(-1*(donor.mass-donor.core_mass+(small_numerical_error|units.MSun)), 0.|units.yr)    
        accretor_in_stellar_code = accretor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
        accretor_in_stellar_code.change_mass(-1*(accretor.mass-accretor.core_mass), 0.|units.yr)    
        self.copy_from_stellar()


        donor.moment_of_inertia_of_star = self.moment_of_inertia(donor)        
        accretor.moment_of_inertia_of_star = self.moment_of_inertia(accretor)        

        bs.semimajor_axis = a_new
        bs.eccentricity = minimum_eccentricity

        #set to synchronization
        corotating_frequency = corotating_spin_angular_frequency_binary(a_new, donor.mass, accretor.mass)
        donor.spin_angular_frequency = corotating_frequency
        accretor.spin_angular_frequency = corotating_frequency

        self.check_RLOF()       
        if self.has_donor():
            print(self.triple.child2.child1.mass, self.triple.child2.child2.mass, self.triple.child2.semimajor_axis, self.triple.child2.eccentricity, self.triple.child2.child1.is_donor, self.triple.child2.child2.is_donor)
            print(self.triple.child1.mass, self.triple.semimajor_axis, self.triple.eccentricity, self.triple.child1.is_donor)
    
    #        sys.exit("error in adjusting triple after double CE: RLOF")
            stopping_condition = perform_inner_merger(bs, donor, accretor, self)
            if not stopping_condition: #stellar interaction
               return False

#        adjusting of stellar system
#        in previous case of merger, the adjustment is done there as mass may be lost during the merger
        adjust_system_after_ce_in_inner_binary(bs, self)                    

    donor.is_donor = False
    bs.is_mt_stable = True
    bs.bin_type = bin_type['detached']
    self.instantaneous_evolution = True #skip secular evolution                
    return True

def common_envelope_phase(bs, donor, accretor, self):
    stopping_condition = True
    
    if REPORT_FUNCTION_NAMES:
        print('Common envelope phase', self.which_common_envelope)
        print('donor:', donor.stellar_type)
        print('accretor:', accretor.stellar_type)

    if donor.stellar_type not in stellar_types_giants and accretor.stellar_type not in stellar_types_giants:
#        possible options: MS+MS, MS+remnant, remnant+remnant,
#                          HeMS+HeMS, HeMS+MS, HeMS+remnant
        bs.bin_type = bin_type['common_envelope']                
        self.save_snapshot()        
        stopping_condition = perform_inner_merger(bs, donor, accretor, self)
        if not stopping_condition: #stellar interaction
            return False                                                           

        self.check_RLOF()       
        if self.has_donor():
            print(self.triple.child2.child1.mass, self.triple.child2.child2.mass, self.triple.child2.semimajor_axis, self.triple.child2.eccentricity, self.triple.child2.child1.is_donor, self.triple.child2.child2.is_donor)
            print(self.triple.child1.mass, self.triple.semimajor_axis, self.triple.eccentricity, self.triple.child1.is_donor)
            print(self.triple.child2.child1.radius, self.triple.child2.child2.radius,self.triple.child1.radius)
            print(self.secular_code.give_roche_radii(self.triple))
            print('binary Roche lobe radii:', roche_radius(bs, bs.child1, self), roche_radius(bs, bs.child2, self))

#            sys.exit("error in adjusting system after CE: RLOF")
            stopping_condition = perform_inner_merger(bs, donor, accretor, self)
            if not stopping_condition: #stellar interaction
               return False
            
        donor.is_donor = False
        bs.is_mt_stable = True
        bs.bin_type = bin_type['detached']
        self.instantaneous_evolution = True #skip secular evolution                

        return True
    

        
    if self.which_common_envelope == 0:
        if donor.stellar_type in stellar_types_giants and accretor.stellar_type in stellar_types_giants:
           stopping_condition = double_common_envelope_energy_balance(bs, donor, accretor, self)
        else:
            stopping_condition = common_envelope_energy_balance(bs, donor, accretor, self)
    elif self.which_common_envelope == 1:
        if donor.stellar_type in stellar_types_giants and accretor.stellar_type in stellar_types_giants:
            stopping_condition = double_common_envelope_energy_balance(bs, donor, accretor, self)
        else:
            stopping_condition = common_envelope_angular_momentum_balance(bs, donor, accretor, self)
    elif self.which_common_envelope == 2:
        Js_d = self.spin_angular_momentum(donor)
        Js_a = self.spin_angular_momentum(accretor)        
        Jb = self.orbital_angular_momentum(bs)
        Js = max(Js_d, Js_a)
#        print("Darwin Riemann instability? donor/accretor:", Js_d, Js_a, Jb, Jb/3.)
        if donor.stellar_type in stellar_types_giants and accretor.stellar_type in stellar_types_giants:
            #giant+giant
            stopping_condition = double_common_envelope_energy_balance(bs, donor, accretor, self)
        elif donor.stellar_type in stellar_types_compact_objects or accretor.stellar_type in stellar_types_compact_objects:
            #giant+remnant
            stopping_condition = common_envelope_energy_balance(bs, donor, accretor, self)
        elif Js >= Jb/3. :            
            #darwin riemann instability
            stopping_condition = common_envelope_energy_balance(bs, donor, accretor, self)
        else:
            #giant+normal(non-giant, non-remnant)
            stopping_condition = common_envelope_angular_momentum_balance(bs, donor, accretor, self)   

    return stopping_condition
    

def adiabatic_expansion_due_to_mass_loss(a_i, Md_f, Md_i, Ma_f, Ma_i):

    d_Md = Md_f - Md_i #negative mass loss rate
    d_Ma = Ma_f - Ma_i #positive mass accretion rate  

    Mt_f = Md_f + Ma_f
    Mt_i = Md_i + Ma_i

    if d_Md < 0|units.MSun and d_Ma >= 0|units.MSun:
        eta = d_Ma / d_Md
        a_f = a_i * ((Md_f/Md_i)**eta * (Ma_f/Ma_i))**-2 * Mt_i/Mt_f
        return a_f
    return a_i
   
       
def adjust_system_after_ce_in_inner_binary(bs, self):
    # Assumption: Unstable mass transfer (common-envelope phase) in the inner binary, affects the outer binary as a wind. 
    # Instanteneous effect
    if REPORT_FUNCTION_NAMES:
        print('Adjust system after ce in inner binary')

    if self.is_triple():
        M_com_after_ce = self.get_mass(bs)
        M_com_before_ce = bs.previous_mass
        
        if self.triple.child1.is_star:
            tertiary_star = self.triple.child1
        else:
            tertiary_star = self.triple.child2 
        # accretion_efficiency
        M_accretor_before_ce = tertiary_star.mass 
        M_accretor_after_ce = tertiary_star.mass 
        
        a_new = adiabatic_expansion_due_to_mass_loss(self.triple.semimajor_axis, M_com_after_ce, M_com_before_ce, M_accretor_after_ce, M_accretor_before_ce)
        self.triple.semimajor_axis = a_new
#        print('outer orbit', a_new)
    

# nice but difficult to update self.triple       
#    system = bs
#    while True:
#        try:    
#            system = system.parent
#            if not system.child1.is_star and system.child2.is_star:
#                system = adjust_triple_after_ce_in_inner_binary(system, system.child1, system.child2, self)                
#            elif not system.child2.is_star and system.child1.is_star:
#                system = adjust_triple_after_ce_in_inner_binary(system, system.child2, system.child1, self)                            
#            else:
#                print('adjust_system_after_ce_in_inner_binary: type of system unknown')
#                exit(2)
#                                        
#        except AttributeError:
#            #when there is no parent
#            break
#


def stable_mass_transfer(bs, donor, accretor, self):
    # orbital evolution is being taken into account in secular_code        
    if REPORT_FUNCTION_NAMES:
        print('Stable mass transfer')

    if bs.bin_type != bin_type['stable_mass_transfer']:
        bs.bin_type = bin_type['stable_mass_transfer']                
        self.save_snapshot()        
    else:
        bs.bin_type = bin_type['stable_mass_transfer']                

    self.secular_code.parameters.check_for_inner_RLOF = False
    self.secular_code.parameters.include_spin_radius_mass_coupling_terms_star1 = False
    self.secular_code.parameters.include_spin_radius_mass_coupling_terms_star2 = False

    Md = donor.mass
    Ma = accretor.mass
    
    dt = self.triple.time - self.previous_time
    dm_desired = bs.mass_transfer_rate * dt
    if REPORT_FUNCTION_NAMES:
        print(bs.mass_transfer_rate, dt, dm_desired)
    donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
    donor_in_stellar_code.change_mass(dm_desired+(small_numerical_error|units.MSun), dt)
    
    # dm != dm_desired e.g. when the envelope of the star becomes empty
    dm = donor_in_stellar_code.mass - Md
    bs.part_dt_mt = 1.
    if dm - dm_desired > numerical_error|units.MSun:
#        print('WARNING:the envelope is empty, mass transfer rate should be lower or dt should be smaller... ')
        bs.part_dt_mt = dm/dm_desired
        
    # there is an implicit assumption in change_mass that the accreted mass is of solar composition (hydrogen)
    accretor_in_stellar_code = accretor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
#    accretor_in_stellar_code.change_mass(dm, dt)
    # for now, only conservative mass transfer   
    accretor_in_stellar_code.change_mass(-1.*dm, -1.*dt)
    #if you want seba to determine the accretion efficiency, use 
    #accretor_in_stellar_code.change_mass(-1.*dm, dt)
    #note doesnt work perfectly, as seba is oblivious to the roche lobe radius
    
    #    self.stellar_code.evolve_model(minimum_time_step) #to get updates radii, not just inflation of stars due to accretion #silvia


    #to adjust radius to mass loss and increase  
    self.stellar_code.evolve_model(self.triple.time)
    self.copy_from_stellar()

    self.update_stellar_parameters()
            
    Md_new = donor.mass
    Ma_new = accretor.mass
    accretion_efficiency = (Ma_new-Ma)/(Md-Md_new)
    if abs(accretion_efficiency - 1.0) > numerical_error and abs(Md-Md_new - -1.*(Ma-Ma_new)) > numerical_error |units.MSun:
        self.save_snapshot()
        print('stable_mass_transfer: non conservative mass transfer')
        print(Md, Ma, donor.previous_mass, accretor.previous_mass)
        print(Md_new, Ma_new, Md-Md_new, Ma-Ma_new, accretion_efficiency)
        print(donor.stellar_type, accretor.stellar_type)
        sys.exit('error in stable mass transfer')
        
    bs.accretion_efficiency_mass_transfer = accretion_efficiency

    corotation_spin = corotating_spin_angular_frequency_binary(bs.semimajor_axis, donor.mass, accretor.mass)
    donor.spin_angular_frequency = corotation_spin
    accretor.spin_angular_frequency = corotation_spin


    
def semi_detached(bs, donor, accretor, self):
#only for binaries (consisting of two stars)
    if REPORT_FUNCTION_NAMES:
        print('Semi-detached')
        print(bs.semimajor_axis, donor.mass, accretor.mass, donor.stellar_type, accretor.stellar_type, bs.is_mt_stable)
        

    stopping_condition = True 
    if bs.is_mt_stable:
        stable_mass_transfer(bs, donor, accretor, self)
        #adjusting triple is done in secular evolution code
    else:        
        stopping_condition = common_envelope_phase(bs, donor, accretor, self)

    return stopping_condition
               
    #possible problem if companion or tertiary accretes significantly from this
#    self.update_previous_stellar_parameters() #previous_mass, previous_radius for safety check
#-------------------------
#functions for contact mass transfer in a multiple / triple

#change parameters assuming fully conservative mass transfer
def perform_mass_equalisation_for_contact(bs, donor, accretor, self):
    if REPORT_FUNCTION_NAMES:
        print('perform_mass_equalisation_for_contact')
    if REPORT_BINARY_EVOLUTION:
        print('Start of stable mass transfer of contact systems' ) 

    if donor.mass != accretor.mass: 
    # if abs(donor.mass - accretor.mass)> 1e-4|units.MSun: #could be better. see if problems arise
        new_mass = 0.5*(donor.mass + accretor.mass)
        bs.semimajor_axis = bs.semimajor_axis * (donor.mass * accretor.mass / new_mass ** 2) ** 2

        delta_mass_donor = new_mass - donor.mass
        delta_mass_accretor = new_mass - accretor.mass
        donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
        accretor_in_stellar_code = accretor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]

        donor_in_stellar_code.change_mass(delta_mass_donor, -1.0|units.yr)
        accretor_in_stellar_code.change_mass(delta_mass_accretor, -1.0|units.yr)
        self.stellar_code.evolve_model(minimum_time_step) #to get updates radii, not just inflation of stars due to accretion
        self.copy_from_stellar()                     
        self.update_stellar_parameters() #makes secular code use update values: mass equalisation happens instantaneously
        
        #set to synchronization
        if self.include_CHE:
            corotating_frequency = corotating_spin_angular_frequency_binary(bs.semimajor_axis, donor.mass, accretor.mass)
            donor.spin_angular_frequency = corotating_frequency
            accretor.spin_angular_frequency = corotating_frequency
            donor.rotation_period = (2*np.pi/donor.spin_angular_frequency)
            accretor.rotation_period = (2*np.pi/accretor.spin_angular_frequency)
            self.channel_to_stellar.copy_attributes(['rotation_period']) #only defined when include_CHE

        self.secular_code.parameters.include_inner_RLOF_terms = False 
        self.secular_code.parameters.include_outer_RLOF_terms = False 



def contact_system(bs, star1, star2, self):
    #if this implementation changes, then also change the is_mt_stable
    if REPORT_FUNCTION_NAMES:
        print("Contact system")

    bs.bin_type = bin_type['contact']                
    self.save_snapshot()        
    self.check_RLOF() #@andris: is this necessary?

    #for now no W Ursae Majoris evolution
    #so for now contact binaries merge in common_envelope_phase, and MS-MS contact binaries will have mass equalisation
    if bs.is_mt_stable: # happens when star1 & star2 are both on MS 
        perform_mass_equalisation_for_contact(bs, bs.child1, bs.child2, self)
        stopping_condition = True 
    else:
        if star1.mass >= star2.mass:
            stopping_condition = common_envelope_phase(bs, star1, star2, self)
        else:
            stopping_condition = common_envelope_phase(bs, star2, star1, self)

    return stopping_condition       


#-------------------------
#functions for mass transfer in a multiple / triple
def CBD_check(m_donor, m_accretor, a_outer, e_outer, a_inner, e_inner):
    # Condition to check whether a CBD should form  
    if not INCLUDE_CBD_IN_TSMT:
        return False, 0
      
    q_out = m_donor/m_accretor
    R_CBD = 0.0425*a_outer*(1-e_outer)*(1/q_out*(1+(1/q_out)))**(0.25)
    CBD = False
    if a_inner*(1+e_inner) < R_CBD:
        CBD = True
#    print('CBD', CBD, a_inner*(1+e_inner), R_CBD)
    CBD_frac = a_inner*(1+e_inner) / R_CBD
    return CBD, CBD_frac

def calculate_adot_GW(a, e, m1, m2):
    # Semi-major axis evolution due to gravitational waves, from Peters (1964)
    alpha = (a)**3*(1-e**2)**(7/2)
    beta = 64/5*constants.G**3*m1*m2*(m1+m2)/constants.c**5
    gamma = 1 + (e**2)*73/24 + (e**4)*37/96
    adot = -beta*gamma/alpha
    return adot
    
def calculate_edot_GW(a, e, m1, m2):
    # Circularization of orbit due to gravitational waves, from Peters (1964)
    alpha = (a)**4*(1-e**2)**(5/2)
    beta = 304/15*e*constants.G**3*m1*m2*(m1+m2)/constants.c**5
    gamma = 1 + e**2*121/304
    edot = -beta*gamma/alpha
    return edot
 
 
def factor_GDF_Ostriker(mach, rmax, rmin):
    # Factor to include to drag force due to the wake created by a linear single perturber. Values adopted from Ostriker 1999
    if mach < 1.0:
        I = 0.5*np.log((1+mach)/(1-mach)) - mach
    else:
        I = 0.5*np.log(1-1/(mach**2)) + np.log(rmax/rmin)
    return I

def factor_GDF_Kim08(mach, rmin, a_in):
    # Factor to include to drag force due to the wake created by binary objects. Values adopted from Kim et al. (2007,2008)
    I1_0 = 0.7706*np.log((1+mach)/(1.0004-0.9185)) - 1.4703*mach
    I1_1 = np.log(330*a_in/(rmin)*(mach-0.71)**(5.72)*mach**(-9.58))
    
    if mach < 1.0:
        I1 = 0.7706*np.log((1+mach)/(1.0004-0.9185)) - 1.4703*mach
    elif mach < 4.4:
        I1 = np.log(330*a_in/(rmin)*(mach-0.71)**(5.72)*mach**(-9.58))
    else:
        I1 = np.log(a_in/(rmin)/(0.11*mach+1.65))
    if mach < 2.97:
        I2 = -0.022*(10-mach)*np.tanh(3*mach/2)
    else:
        I2 = -0.13 + 0.07*np.arctan(5*mach-15)
        
    return I1+I2
        
def integrand_da_GDF(f, e): # Integral in the GDF equation
    return 1/((1+e*np.cos(f))**2*np.sqrt(1+2*e*np.cos(f)+e**2))
    
def integrand_da_hydr(f, e): # Integral in the GDF equation
    return (1+2*e*np.cos(f)+e**2)**(3/2)/((1+e*np.cos(f))**2)

def integrand_de_GDF(f, e): # Integral in the GDF equation for eccentric binaries
    return (e+np.cos(f))/((1+e*np.cos(f))**2*(1+2*e*np.cos(f)+e**2)**(3/2))
    
def integrand_de_hydr(f, e): # Integral in the GDF equation for eccentric binaries
    return np.sqrt(1+2*e*np.cos(f)+e**2)*(e+np.cos(f))/((1+e*np.cos(f))**2)
  
 
def calculate_adot_GDF(a, e, q, m1, m2, mbin, CBD, r1, r2):
    # Semi-major axis evolution due to gaseous drag forces
    
    if CBD: # if a CBD has formed, we assume the binary orbit is clear of gas
        return 0|units.RSun/units.Myr
    
    v_orb = np.sqrt(constants.G*mbin/a) # orbital velocity of the binary
    mach = v_orb / c_s_BA_in_TSMT # Mach number (for simplicity omitted dependency on the true anomaly)

    rmax = 2*a
    rmin = a/10
    if model_I_GDF == 'Ostriker99':
        I = factor_GDF_Ostriker(mach, rmax, rmin)
    elif model_I_GDF == 'Kim08':
        I = factor_GDF_Kim08(mach, rmin, a)#@floris uses self.a_in
    
    integral_da = 2*np.pi

    n = np.sqrt(constants.G*mbin/a**3)
    mu = m1*m2/mbin
    adot_GDF = 0|units.RSun/units.Myr
    adot_hydr = 0|units.RSun/units.Myr
    
    if INCLUDE_GDF_IN_TSMT: # Contribution of gravitational gas drag
        # Equation B.6 from Kummer et al. (2024)
        if INCLUDE_ECC_GDF_IN_TSMT: # Gas drag in eccentric binaries
            integral_da = integrate.quad(integrand_da_GDF, 0, 2*np.pi, args=(e,))[0]#@floris uses self.e_in
        A0_GDF = -4*np.pi*constants.G**2*density_BA_in_TSMT*I
        adot_GDF = A0_GDF * (1-e**2)**2*mbin**2/(np.pi*n**3*a**2*mu) * (1/q**2 + q**2) * integral_da
        
    if INCLUDE_HYDR_IN_TSMT: # Contribution of hydrodynamic gas drag
        if INCLUDE_ECC_GDF_IN_TSMT: # Gas drag in eccentric binaries
            integral_da = integrate.quad(integrand_da_hydr, 0, 2*np.pi, args=(e,))[0]#@floris uses self.e_in
        A0_hydr = -0.5*hydro_drag_coefficient_in_TSMT*np.pi*density_BA_in_TSMT
        adot_hydr = A0_hydr * n*a**2/(np.pi*mu*mbin**2) * (r1**2*m2**2+r2**2*m1**2) * integral_da

    adot = adot_GDF + adot_hydr

    return adot
    
    
def calculate_edot_GDF(a, e, q, m1, m2, mbin, CBD, r1, r2): 
    # Evolution of eccentricity due to gas drag (Rozner & Perets 2022)

    if not INCLUDE_ECC_GDF_IN_TSMT: 
        return 0|1./units.Myr    
    if CBD: # if a CBD has formed, we assume the binary orbit is clear of gas
        return 0|1./units.Myr
   
    # Inspiral due to gas drag
    v_orb = np.sqrt(constants.G*mbin/a)
    mach = v_orb / c_s_BA_in_TSMT

    rmax = 2*a
    rmin = a/10.
    if model_I_GDF == 'Ostriker99':
        I = factor_GDF_Ostriker(mach, rmax, rmin)
    elif model_I_GDF == 'Kim08':
        I = factor_GDF_Kim08(mach, rmin, a)#@floris uses self.a_in
    
    A_0 = -4*np.pi*constants.G**2*density_BA_in_TSMT*I
    n = np.sqrt(constants.G*mbin/a**3)
    mu = m1*m2/mbin
    edot_GDF = 0|1./units.Myr
    edot_hydr = 0|1./units.Myr
    
    if (INCLUDE_GDF_IN_TSMT): # Contribution of gravitational gas drag
        # Equation B.9 from Kummer et al. (2024)
        if INCLUDE_ECC_GDF_IN_TSMT: # Gravitational gas drag in eccentric binaries
            integral_de_GDF = integrate.quad(integrand_de_GDF, 0, 2*np.pi, args=(e,))[0]#@floris uses self.e_in
        A0_GDF = -4*np.pi*constants.G**2*density_BA_in_TSMT*I
        edot_GDF = A0_GDF * (1-e**2)**3*mbin**2/(np.pi*n**3*a**3*mu) * (1/q**2 + q**2) * integral_de_GDF

    if INCLUDE_HYDR_IN_TSMT: # contribution of hydrodynamic gas drag
        if INCLUDE_ECC_GDF_IN_TSMT: # Gravitational gas drag in eccentric binaries
            integral_de_hydr = integrate.quad(integrand_de_hydr, 0, 2*np.pi, args=(e))[0]#@floris uses self.e_in
        A0_hydr = -0.5*hydro_drag_coefficient_in_TSMT*np.pi*density_BA_in_TSMT
        edot_hydr = A0_hydr * n*a*(1-e**2)/(np.pi*mu*mbin**2) * (r1**2*m2**2+r2**2*m1**2) * integral_de_hydr
    
    edot = edot_GDF + edot_hydr
    
    return edot

def calculate_adot_CBD(a, e, q, incl, mbin, mdot_bin, CBD, self):
    # Semi-major axis evolution due to torques between CBD and inner binary.

    if not CBD:
        return 0|units.RSun/units.Myr

    if (incl <= np.pi/2) | (not INCLUDE_RETROGRADE_CBD_IN_TSMT): # Adopted values from Siwek et al. (2023)
        try:
            # If e>0.8 we don't extrapolate, but use edge value
            factor = self.intp_grid_a((q, min(e, 0.8)))
        except AttributeError:
            # Interpolated values for a_dot and e_dot in CBD from Siwek et al. (2023)
            # We define the interpolator at the start so that we only have to do it once (saves a lot of time)
            self.intp_grid_a, _, _, _ = grid_interpolation(param='a')
            self.intp_grid_e, _, _, _ = grid_interpolation(param='e')

            # If e>0.8 we don't extrapolate, but use edge value
            factor = self.intp_grid_a((q, min(e, 0.8)))

    elif (incl > np.pi/2) & (incl <= np.pi): # Retrograde orbits adopted from Tiede & D'Orazio (2023)
        factor = -10
    else:
        print('Non-physical inclination: ', incl)
    
    adot = factor * a * mdot_bin / (mbin)
#    print('adot CBD', adot,  a, factor, mdot_bin, mbin, incl)

    return adot


def calculate_edot_CBD(e, q, incl, mbin, mdot_bin, CBD, self):
    # Evolution of the eccentricity due to CBD torques

    if not CBD:
        return 0|1./units.Myr

    if (incl <= np.pi/2) | (not INCLUDE_RETROGRADE_CBD_IN_TSMT): # Adopted values from Siwek et al. (2023)
        try:
            # If e>0.8 we don't extrapolate, but use edge value
            factor = self.intp_grid_e((q, min(e, 0.8)))
        except AttributeError:
            # Interpolated values for a_dot and e_dot in CBD from Siwek et al. (2023)
            # We define the interpolator at the start so that we only have to do it once (saves a lot of time)
            self.intp_grid_a, _, _, _ = grid_interpolation(param='a')
            self.intp_grid_e, _, _, _ = grid_interpolation(param='e')

            # If e>0.8 we don't extrapolate, but use edge value
            factor = self.intp_grid_e((q, min(e, 0.8)))
    elif (incl > np.pi/2) & (incl <= np.pi): # Retrograde orbits adopted from Tiede & D'Orazio (2023)
        if e <= 0.1:
            factor = 30*e
        else:
            factor = 2
    else:
        print('Non-physical inclination: ', incl)
        
    edot = factor * mdot_bin / mbin
#    print('edot_CBD', edot, e, factor, mdot_bin, mbin)
    
    return edot

def grid_interpolation(param):
        # Values for a_dot and e_dot in systems with a CBD. Data from Siwek et al. (2023)
        
        adot_data = {'e=0.00_q=0.10_ab_dot_ab_sum_grav_acc': -1.2588927484346075, 'e=0.00_q=0.20_ab_dot_ab_sum_grav_acc': -0.7030327450679021, 'e=0.00_q=0.30_ab_dot_ab_sum_grav_acc': 1.1529081340202179, 'e=0.00_q=0.40_ab_dot_ab_sum_grav_acc': 1.277961324597141, 'e=0.00_q=0.50_ab_dot_ab_sum_grav_acc': 1.4195367749323786, 'e=0.00_q=0.60_ab_dot_ab_sum_grav_acc': 1.5680280340458284, 'e=0.00_q=0.70_ab_dot_ab_sum_grav_acc': 1.6615004112961926, 'e=0.00_q=0.80_ab_dot_ab_sum_grav_acc': 1.7087208207988906, 'e=0.00_q=0.90_ab_dot_ab_sum_grav_acc': 1.7314879550329025, 'e=0.00_q=1.00_ab_dot_ab_sum_grav_acc': 1.7534726250735466, 'e=0.02_q=1.00_ab_dot_ab_sum_grav_acc': 1.7002876283117645, 'e=0.03_q=1.00_ab_dot_ab_sum_grav_acc': 1.6081955329524986, 'e=0.04_q=1.00_ab_dot_ab_sum_grav_acc': 1.5595213761160105, 'e=0.05_q=0.10_ab_dot_ab_sum_grav_acc': -3.33794314967003, 'e=0.05_q=0.20_ab_dot_ab_sum_grav_acc': -0.4878574634899152, 'e=0.05_q=0.30_ab_dot_ab_sum_grav_acc': 0.8744350518418839, 'e=0.05_q=0.40_ab_dot_ab_sum_grav_acc': 1.1370063804234058, 'e=0.05_q=0.50_ab_dot_ab_sum_grav_acc': 1.4126349071380913, 'e=0.05_q=0.60_ab_dot_ab_sum_grav_acc': 1.4965637338473716, 'e=0.05_q=0.70_ab_dot_ab_sum_grav_acc': 1.5203706193024829, 'e=0.05_q=0.80_ab_dot_ab_sum_grav_acc': 1.5421167860902194, 'e=0.05_q=0.90_ab_dot_ab_sum_grav_acc': 1.5485050264876217, 'e=0.05_q=1.00_ab_dot_ab_sum_grav_acc': 1.540844661306394, 'e=0.10_q=0.10_ab_dot_ab_sum_grav_acc': -5.0571681159957915, 'e=0.10_q=0.20_ab_dot_ab_sum_grav_acc': -1.9339580701250532, 'e=0.10_q=0.30_ab_dot_ab_sum_grav_acc': -2.251490562749219, 'e=0.10_q=0.40_ab_dot_ab_sum_grav_acc': -1.2215179723238372, 'e=0.10_q=0.50_ab_dot_ab_sum_grav_acc': -0.8367284116270806, 'e=0.10_q=0.60_ab_dot_ab_sum_grav_acc': -0.8097620665737617, 'e=0.10_q=0.70_ab_dot_ab_sum_grav_acc': -0.8886922901256215, 'e=0.10_q=0.80_ab_dot_ab_sum_grav_acc': -0.9604008726655703, 'e=0.10_q=0.90_ab_dot_ab_sum_grav_acc': -0.9214870630601979, 'e=0.10_q=1.00_ab_dot_ab_sum_grav_acc': -0.9809361330347988, 'e=0.20_q=0.10_ab_dot_ab_sum_grav_acc': 1.0424403269129217, 'e=0.20_q=0.20_ab_dot_ab_sum_grav_acc': -0.19939941769747618, 'e=0.20_q=0.30_ab_dot_ab_sum_grav_acc': -1.953628426276366, 'e=0.20_q=0.40_ab_dot_ab_sum_grav_acc': -0.6486894781725229, 'e=0.20_q=0.50_ab_dot_ab_sum_grav_acc': -0.20949241199853907, 'e=0.20_q=0.60_ab_dot_ab_sum_grav_acc': -0.4546503025120044, 'e=0.20_q=0.70_ab_dot_ab_sum_grav_acc': -0.49332138407726533, 'e=0.20_q=0.80_ab_dot_ab_sum_grav_acc': -0.7246914681996611, 'e=0.20_q=0.90_ab_dot_ab_sum_grav_acc': -1.0640292985942033, 'e=0.20_q=1.00_ab_dot_ab_sum_grav_acc': -1.364061619122991, 'e=0.30_q=0.10_ab_dot_ab_sum_grav_acc': 3.503605883308187, 'e=0.30_q=0.20_ab_dot_ab_sum_grav_acc': 0.9028435325239866, 'e=0.30_q=0.30_ab_dot_ab_sum_grav_acc': -0.21607001774721982, 'e=0.30_q=0.40_ab_dot_ab_sum_grav_acc': -2.4865544311318644, 'e=0.30_q=0.50_ab_dot_ab_sum_grav_acc': -2.4247811179306824, 'e=0.30_q=0.60_ab_dot_ab_sum_grav_acc': -2.3305411398131177, 'e=0.30_q=0.70_ab_dot_ab_sum_grav_acc': -2.3456293472156124, 'e=0.30_q=0.80_ab_dot_ab_sum_grav_acc': -2.6127293026279035, 'e=0.30_q=0.90_ab_dot_ab_sum_grav_acc': -4.048571604304598, 'e=0.30_q=1.00_ab_dot_ab_sum_grav_acc': -4.850313932871829, 'e=0.40_q=0.10_ab_dot_ab_sum_grav_acc': 3.806236733517315, 'e=0.40_q=0.20_ab_dot_ab_sum_grav_acc': 2.7194712808229418, 'e=0.40_q=0.30_ab_dot_ab_sum_grav_acc': -1.4543339398009754, 'e=0.40_q=0.40_ab_dot_ab_sum_grav_acc': -2.5910064909814245, 'e=0.40_q=0.50_ab_dot_ab_sum_grav_acc': -2.2130570681981876, 'e=0.40_q=0.60_ab_dot_ab_sum_grav_acc': -3.1159071887554943, 'e=0.40_q=0.70_ab_dot_ab_sum_grav_acc': -5.274646166880953, 'e=0.40_q=0.80_ab_dot_ab_sum_grav_acc': -6.2079834967683105, 'e=0.40_q=0.90_ab_dot_ab_sum_grav_acc': -6.3075868066968255, 'e=0.40_q=1.00_ab_dot_ab_sum_grav_acc': -6.14002468104283, 'e=0.50_q=0.10_ab_dot_ab_sum_grav_acc': 4.055637000482147, 'e=0.50_q=0.20_ab_dot_ab_sum_grav_acc': 2.65874061811691, 'e=0.50_q=0.30_ab_dot_ab_sum_grav_acc': -0.9603773405190716, 'e=0.50_q=0.40_ab_dot_ab_sum_grav_acc': -2.8808002208052392, 'e=0.50_q=0.50_ab_dot_ab_sum_grav_acc': -3.961916226182325, 'e=0.50_q=0.60_ab_dot_ab_sum_grav_acc': -4.377272084304601, 'e=0.50_q=0.70_ab_dot_ab_sum_grav_acc': -4.171980234634676, 'e=0.50_q=0.80_ab_dot_ab_sum_grav_acc': 0.598365917841959, 'e=0.50_q=0.90_ab_dot_ab_sum_grav_acc': 0.8641031382970823, 'e=0.50_q=1.00_ab_dot_ab_sum_grav_acc': 0.877397138454011, 'e=0.60_q=0.10_ab_dot_ab_sum_grav_acc': 3.157706904114237, 'e=0.60_q=0.20_ab_dot_ab_sum_grav_acc': -1.2873737083279844, 'e=0.60_q=0.30_ab_dot_ab_sum_grav_acc': -2.4379105283348284, 'e=0.60_q=0.40_ab_dot_ab_sum_grav_acc': -1.4704287385170687, 'e=0.60_q=0.50_ab_dot_ab_sum_grav_acc': -1.297418566336738, 'e=0.60_q=0.60_ab_dot_ab_sum_grav_acc': -0.2837682548094211, 'e=0.60_q=0.70_ab_dot_ab_sum_grav_acc': 0.30322709438863127, 'e=0.60_q=0.80_ab_dot_ab_sum_grav_acc': 0.5201687004910162, 'e=0.60_q=0.90_ab_dot_ab_sum_grav_acc': 0.4780718143406398, 'e=0.60_q=1.00_ab_dot_ab_sum_grav_acc': 0.376879952712717, 'e=0.80_q=0.10_ab_dot_ab_sum_grav_acc': -5.366299777195529, 'e=0.80_q=0.20_ab_dot_ab_sum_grav_acc': -6.498385865377792, 'e=0.80_q=0.30_ab_dot_ab_sum_grav_acc': -3.3067963024144644, 'e=0.80_q=0.40_ab_dot_ab_sum_grav_acc': -3.480897204885583, 'e=0.80_q=0.50_ab_dot_ab_sum_grav_acc': -3.521311997035641, 'e=0.80_q=0.60_ab_dot_ab_sum_grav_acc': -2.746699994636253, 'e=0.80_q=0.70_ab_dot_ab_sum_grav_acc': -2.956680828764333, 'e=0.80_q=0.80_ab_dot_ab_sum_grav_acc': -3.1657741705136586, 'e=0.80_q=0.90_ab_dot_ab_sum_grav_acc': -3.001088696745606, 'e=0.80_q=1.00_ab_dot_ab_sum_grav_acc': -2.892418724373073}
        
        edot_data = {'e=0.00_q=0.10_eb_dot_sum_grav_acc': 3.488635249010523e-05, 'e=0.00_q=0.20_eb_dot_sum_grav_acc': 7.442294134508703e-06, 'e=0.00_q=0.30_eb_dot_sum_grav_acc': 1.6933772285208686e-05, 'e=0.00_q=0.40_eb_dot_sum_grav_acc': 9.298533107418273e-06, 'e=0.00_q=0.50_eb_dot_sum_grav_acc': -8.878210719378644e-06, 'e=0.00_q=0.60_eb_dot_sum_grav_acc': 1.9557428804050647e-05, 'e=0.00_q=0.70_eb_dot_sum_grav_acc': 1.0281107471336626e-05, 'e=0.00_q=0.80_eb_dot_sum_grav_acc': -3.088926860884904e-05, 'e=0.00_q=0.90_eb_dot_sum_grav_acc': -8.539577730279929e-07, 'e=0.00_q=1.00_eb_dot_sum_grav_acc': 1.871404863945317e-05, 'e=0.02_q=1.00_eb_dot_sum_grav_acc': 0.3743280300991139, 'e=0.03_q=1.00_eb_dot_sum_grav_acc': 0.47382997856152176, 'e=0.04_q=1.00_eb_dot_sum_grav_acc': 0.5355490843846645, 'e=0.05_q=0.10_eb_dot_sum_grav_acc': -0.4684303050118692, 'e=0.05_q=0.20_eb_dot_sum_grav_acc': 0.02263375780103669, 'e=0.05_q=0.30_eb_dot_sum_grav_acc': 0.13470403236369005, 'e=0.05_q=0.40_eb_dot_sum_grav_acc': 0.5427739657742825, 'e=0.05_q=0.50_eb_dot_sum_grav_acc': 0.7735635300373794, 'e=0.05_q=0.60_eb_dot_sum_grav_acc': 1.029296443982413, 'e=0.05_q=0.70_eb_dot_sum_grav_acc': 0.9168445555194249, 'e=0.05_q=0.80_eb_dot_sum_grav_acc': 0.8126672238205267, 'e=0.05_q=0.90_eb_dot_sum_grav_acc': 0.7476545394168524, 'e=0.05_q=1.00_eb_dot_sum_grav_acc': 0.692091816898143, 'e=0.10_q=0.10_eb_dot_sum_grav_acc': 1.5504011121815902, 'e=0.10_q=0.20_eb_dot_sum_grav_acc': 1.8183572855353745, 'e=0.10_q=0.30_eb_dot_sum_grav_acc': 3.90952703020401, 'e=0.10_q=0.40_eb_dot_sum_grav_acc': 4.175613133752098, 'e=0.10_q=0.50_eb_dot_sum_grav_acc': 4.562835763640335, 'e=0.10_q=0.60_eb_dot_sum_grav_acc': 4.911391593859991, 'e=0.10_q=0.70_eb_dot_sum_grav_acc': 5.068415905856453, 'e=0.10_q=0.80_eb_dot_sum_grav_acc': 5.340847000214718, 'e=0.10_q=0.90_eb_dot_sum_grav_acc': 5.274270071209753, 'e=0.10_q=1.00_eb_dot_sum_grav_acc': 5.406401814667789, 'e=0.20_q=0.10_eb_dot_sum_grav_acc': 0.6488927191944633, 'e=0.20_q=0.20_eb_dot_sum_grav_acc': 2.1914972004333824, 'e=0.20_q=0.30_eb_dot_sum_grav_acc': 5.669103545016428, 'e=0.20_q=0.40_eb_dot_sum_grav_acc': 3.3710710541138145, 'e=0.20_q=0.50_eb_dot_sum_grav_acc': 3.8298553045887007, 'e=0.20_q=0.60_eb_dot_sum_grav_acc': 4.986611934218368, 'e=0.20_q=0.70_eb_dot_sum_grav_acc': 5.557890572373414, 'e=0.20_q=0.80_eb_dot_sum_grav_acc': 6.078860672454515, 'e=0.20_q=0.90_eb_dot_sum_grav_acc': 6.707204607122081, 'e=0.20_q=1.00_eb_dot_sum_grav_acc': 7.188775573084456, 'e=0.30_q=0.10_eb_dot_sum_grav_acc': -1.7896526848136787, 'e=0.30_q=0.20_eb_dot_sum_grav_acc': 0.07757532678305445, 'e=0.30_q=0.30_eb_dot_sum_grav_acc': 0.19236677200001306, 'e=0.30_q=0.40_eb_dot_sum_grav_acc': 2.5882196798559822, 'e=0.30_q=0.50_eb_dot_sum_grav_acc': 3.362923747952593, 'e=0.30_q=0.60_eb_dot_sum_grav_acc': 4.5230423777706354, 'e=0.30_q=0.70_eb_dot_sum_grav_acc': 5.268203818640236, 'e=0.30_q=0.80_eb_dot_sum_grav_acc': 6.124410529143764, 'e=0.30_q=0.90_eb_dot_sum_grav_acc': 8.223391449350354, 'e=0.30_q=1.00_eb_dot_sum_grav_acc': 9.521288212023046, 'e=0.40_q=0.10_eb_dot_sum_grav_acc': -4.184212679532789, 'e=0.40_q=0.20_eb_dot_sum_grav_acc': -1.953532560321229, 'e=0.40_q=0.30_eb_dot_sum_grav_acc': -0.41337492723649594, 'e=0.40_q=0.40_eb_dot_sum_grav_acc': 0.24481350668945567, 'e=0.40_q=0.50_eb_dot_sum_grav_acc': 1.5048676028128227, 'e=0.40_q=0.60_eb_dot_sum_grav_acc': 3.543842173457423, 'e=0.40_q=0.70_eb_dot_sum_grav_acc': 5.656875229581482, 'e=0.40_q=0.80_eb_dot_sum_grav_acc': 6.5113173252052, 'e=0.40_q=0.90_eb_dot_sum_grav_acc': 7.0991178999755995, 'e=0.40_q=1.00_eb_dot_sum_grav_acc': 6.933509837783115, 'e=0.50_q=0.10_eb_dot_sum_grav_acc': -4.792735124472431, 'e=0.50_q=0.20_eb_dot_sum_grav_acc': -3.9475753656667734, 'e=0.50_q=0.30_eb_dot_sum_grav_acc': -2.7378009846402596, 'e=0.50_q=0.40_eb_dot_sum_grav_acc': -1.7096071112614364, 'e=0.50_q=0.50_eb_dot_sum_grav_acc': -1.7541323255788448, 'e=0.50_q=0.60_eb_dot_sum_grav_acc': -0.03805751777859892, 'e=0.50_q=0.70_eb_dot_sum_grav_acc': 0.5437329180055392, 'e=0.50_q=0.80_eb_dot_sum_grav_acc': -1.7636007674516854, 'e=0.50_q=0.90_eb_dot_sum_grav_acc': -1.8381672944153815, 'e=0.50_q=1.00_eb_dot_sum_grav_acc': -1.8569328012861426, 'e=0.60_q=0.10_eb_dot_sum_grav_acc': -5.908017840443841, 'e=0.60_q=0.20_eb_dot_sum_grav_acc': -4.605516831316371, 'e=0.60_q=0.30_eb_dot_sum_grav_acc': -3.9505394655778505, 'e=0.60_q=0.40_eb_dot_sum_grav_acc': -2.839562660147846, 'e=0.60_q=0.50_eb_dot_sum_grav_acc': -2.3870738928579285, 'e=0.60_q=0.60_eb_dot_sum_grav_acc': -2.2136858045854977, 'e=0.60_q=0.70_eb_dot_sum_grav_acc': -2.1397409162831007, 'e=0.60_q=0.80_eb_dot_sum_grav_acc': -2.0835290635783794, 'e=0.60_q=0.90_eb_dot_sum_grav_acc': -2.129030745573739, 'e=0.60_q=1.00_eb_dot_sum_grav_acc': -2.122080860608072, 'e=0.80_q=0.10_eb_dot_sum_grav_acc': -7.378638265729229, 'e=0.80_q=0.20_eb_dot_sum_grav_acc': -5.377888608443883, 'e=0.80_q=0.30_eb_dot_sum_grav_acc': -3.435554307708856, 'e=0.80_q=0.40_eb_dot_sum_grav_acc': -2.6144355479495367, 'e=0.80_q=0.50_eb_dot_sum_grav_acc': -2.155093066314607, 'e=0.80_q=0.60_eb_dot_sum_grav_acc': -1.9869900684709847, 'e=0.80_q=0.70_eb_dot_sum_grav_acc': -1.8769639616864813, 'e=0.80_q=0.80_eb_dot_sum_grav_acc': -1.6840438069544894, 'e=0.80_q=0.90_eb_dot_sum_grav_acc': -1.6894204612697994, 'e=0.80_q=1.00_eb_dot_sum_grav_acc': -1.818735569936334}
    
    
        q = np.arange(0.1, 1.1, 0.1)
        e = np.array([0.0, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        Q, E = np.meshgrid(q, e)
        
        if param == 'a':
            data = list(adot_data.values())[:10] + list(adot_data.values())[13:]
            data_2d = np.array(data).reshape(9,10)
            extra_points = [np.nan]*27 + list(adot_data.values())[10:13]
        elif param == 'e':
            data = list(edot_data.values())[:10] + list(edot_data.values())[13:]
            data_2d = np.array(data).reshape(9,10)
            extra_points = [np.nan]*27 + list(edot_data.values())[10:13]
        else:
            print("Wrong interpolation parameter")
            return
            

        # The grid is not regular, so we insert nan for the missing values
        extra_2d = np.array(extra_points).reshape(10,3).T
        combined_data = np.insert(data_2d, 1, extra_2d, axis=0)
        nan_mask = np.isnan(combined_data)
        nan_indices = np.isnan(combined_data)
        
        # We first fill the missing grid values by interpolating with scipy griddata (unfortunately RegularGridInterpolator does not work with irregular grids)
        indices_to_interpolate = np.where(nan_indices.flatten())
        row_indices, col_indices = np.unravel_index(indices_to_interpolate, combined_data.shape)
        points_to_interpolate = np.column_stack((Q.flatten(), E.flatten()))[indices_to_interpolate]
        valid_data_mask = ~np.isnan(combined_data.flatten())
        interpolated_values = griddata((Q.flatten()[valid_data_mask], E.flatten()[valid_data_mask]),
                                       combined_data.flatten()[valid_data_mask],
                                       points_to_interpolate, method='linear')

        combined_data[row_indices, col_indices] = interpolated_values
        
        # Finally, we set up the interpolator
        interpolator = RegularGridInterpolator((q, e), combined_data.T, method='linear', bounds_error=False)

        return interpolator, Q, E, combined_data

def calculate_adot_AM(a, m_donor, m_acc, mass_transfer_rate, conservative_CBD=True, iso=False):
    # Evolution of the outer orbit due to mass transfer based on angular momentum balance

#    if (iso == True) & ((CBD == False) | (conservative_CBD == True)):#@Floris: why adot_AM is 0 in this case Silvia
#        return 0

    #beta: accretion efficiency
    #gamma: angular momentum loss mode
    if conservative_CBD == True: # Conservative mass transfer
        beta = 1
        gamma = 0
    elif iso == True: # Isotropic mass ejection
        beta = 0
        gamma = m_acc / m_donor
    else: # Isotropic re-emission
        beta = 0
        gamma = m_donor / m_acc
      
    adot = -2. * a * mass_transfer_rate/m_donor * (1.-beta*m_donor/m_acc - (1.-beta)*(gamma+0.5)*m_donor/(m_donor+m_acc)) 
    
    return adot


def accretion_rate_onto_binary(mass_transfer_rate, conservative=True, mdot=0|units.MSun/units.yr):
    # Change in mass of inner binary due to mass accretion
    if (conservative == True) & (mdot == 0|units.MSun/units.yr):
        mdot = -1.*mass_transfer_rate
    elif conservative == True: # In case dm was already specified, possibly needed in future to test mt stability
        mdot = mdot
    else:
        mdot = 0|units.MSun/units.Myr #for now assume no accretion at all        
        
    return mdot

 
def derivatives(params, incl, mass_transfer_rate, CBD, conservative_CBD, r1, r2, self):
   # Function that calculates the time-derivatives of the system's properties
   # params = [a_in, e_in, mbin, m1, m2]

   # Constrain unphysical values for the properties of the system
   if params[0] < 0|units.RSun:
       params[0] = 1e-5|units.RSun
   if params[1] < 0:
       params[1] = 0
       
   # The mass ratio should always be smaller than or equal to one
   q_inner = params[3]/params[4]
   if q_inner < 1:
       q_inner = 1./q_inner

   adot_GDF = calculate_adot_GDF(params[0], params[1], q_inner, params[3], params[4], params[2], CBD, r1, r2)
   adot_CBD = calculate_adot_CBD(params[0], params[1], q_inner, incl, params[2], -1*mass_transfer_rate, CBD, self)
   adot_GW = calculate_adot_GW(params[0], params[1], params[3], params[4])
#   print('adot', adot_GDF, adot_GW, adot_CBD)
   adot_in = adot_GDF + adot_CBD + adot_GW
      
   edot_GDF = calculate_edot_GDF(params[0], params[1], q_inner, params[3], params[4], params[2], CBD, r1, r2)
   edot_CBD = calculate_edot_CBD(params[1], q_inner, incl, params[2], -1*mass_transfer_rate, CBD, self)
   edot_GW = calculate_edot_GW(params[0], params[1], params[3], params[4])
#   print('edot',edot_GDF, edot_GW, edot_CBD)
   edot_in = edot_GDF + edot_CBD + edot_GW

   mdot_bin = 0|units.MSun/units.yr
   mdot1 = 0|units.MSun/units.yr
   mdot2 = 0|units.MSun/units.yr
   if conservative_CBD:
       mdot_bin = accretion_rate_onto_binary(mass_transfer_rate, conservative_CBD)
       mdot1 = mdot_bin * 1/(1+q_inner**(-0.9))
       mdot2 = mdot_bin * q_inner**(-0.9)/(1+q_inner**(-0.9))

   return np.array([adot_in, edot_in, mdot_bin, mdot1, mdot2])
        
def RK4_step(params, incl, mass_transfer_rate, CBD, conservative_CBD, dt, r1, r2, self):
    # Fourth order Runge-Kutta method
    # params = [a_in, e_in, mbin, m1, m2]
    
    dt_vec = np.array([dt,dt,dt,dt,dt])   
    k1 = derivatives(params, incl, mass_transfer_rate, CBD, conservative_CBD, r1, r2, self)
    k2 = derivatives(params + 0.5 * k1 * dt_vec, incl, mass_transfer_rate, CBD, conservative_CBD, r1, r2, self)
    k3 = derivatives(params + 0.5 * k2 * dt_vec, incl, mass_transfer_rate, CBD, conservative_CBD, r1, r2, self)
    k4 = derivatives(params + k3 * dt_vec, incl, mass_transfer_rate, CBD, conservative_CBD, r1, r2, self)
    
    return params + (dt_vec / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def adaptive_RK4_step(params, incl, mass_transfer_rate, CBD, conservative_CBD, dt, r1, r2, self):
    # Function that adaptively determines the new time step
    
    iter = 1
    while iter < max_iter_TSMT:
        y_single = RK4_step(params, incl, mass_transfer_rate, CBD, conservative_CBD, dt, r1, r2, self)
        y_double = RK4_step(params, incl, mass_transfer_rate, CBD, conservative_CBD, dt/2, r1, r2, self)
        y_double = RK4_step(y_double, incl, mass_transfer_rate, CBD, conservative_CBD, dt/2, r1, r2, self)
        #@Floris: when eccentricity = 0 , eps_rel -> inf
        
        eps_rel = abs((y_double - y_single)/y_double)
        eps_rel = max(eps_rel)
        if eps_rel == 0: # To avoid numerical complications
            eps_rel = eps_TSMT
        dt_new = dt*(eps_TSMT/eps_rel)**0.2
        dt_new = max(dt_new, minimum_time_step)
            
        # Check if the difference between y_single and y_double is within a certain error
        # @Floris: Second criterion redundant?
        if (eps_rel <= eps_TSMT) | (iter >= max_iter_TSMT) | (dt == minimum_time_step):
            params = y_double
            dt_next = dt_new
            return params, dt, dt_next
            return
        else:
            iter += 1
            dt = dt_new

    #when max iter is reached
    params = y_double
    dt_next = dt_new
              
    return params, dt, dt_next


def triple_stable_mass_transfer(bs, donor, accretor, self):
    # based on Kummer et al. 2024 
    # For now the envelope is stripped in one go aka without updating the stellar evolution
    # Therefore the TRES timestep doesn't matter so much -> internally here smaller timesteps are taken to solve the orbit

    if REPORT_FUNCTION_NAMES:
        print('Triple stable mass transfer')

    if bs.bin_type != bin_type['stable_mass_transfer']:
        bs.bin_type = bin_type['stable_mass_transfer']                
        self.save_snapshot()        
    else:
        bs.bin_type = bin_type['stable_mass_transfer']                
        
    a_outer = bs.semimajor_axis
    e_outer = bs.eccentricity    
    a_inner = accretor.semimajor_axis
    e_inner = accretor.eccentricity
        
    m1 = accretor.child1.mass
    m2 = accretor.child2.mass
    m3 = donor.mass 
    m3_env = donor.mass - donor.core_mass 
    r1 = accretor.child1.radius
    r2 = accretor.child2.radius
    #use Schwarschildradius for NS & BH? @floris
#    if accretor.child1.stellar_type in stellar_types_SN_remnants:
#        r1 = 2*constants.G*m1/(constants.c**2) 
#    if accretor.child2.stellar_type in stellar_types_SN_remnants:
#        r2 = 2*constants.G*m2/(constants.c**2) 


#    time_step = minimum_time_step 
    initial_time_step = abs(0.01*m3_env/bs.mass_transfer_rate)
    time_step = initial_time_step
    total_time_passed = 0|units.yr
    dm3 = abs(bs.mass_transfer_rate) * time_step 
          
    #Whether there is a circumbinary disk in stead of ballistic accretion
    CBD, CBD_fraction = CBD_check(m3, m1+m2, a_outer, e_outer, a_inner, e_inner)
    conservative_CBD = False
    if (not INCLUDE_OUTFLOW_CBD_IN_TSMT) and (CBD): 
        conservative_CBD = True #only true when there is a CBD
    #Whether GWs dominate the orbital evolution        
    GW = False 
    
    while (m3_env > 0|units.MSun):
#        print('m3 env', m3_env, m3, m1,m2,bs.mass_transfer_rate, a_outer, a_inner, time_step, minimum_time_step, self.previous_dt, CBD, GW, a_inner, a_outer, bs.relative_inclination)
        CBD_prev = CBD 
        GW_prev = GW
        
        # Adjust outer orbit and star 
        if dm3 > m3_env:
            dm3 = m3_env
            time_step = time_step * m3_env / m3
#            m3_env = 0|units.MSun

        # Adjust inner binary and stars
        # For printing purpose
        params = np.array([a_inner, e_inner, m1+m2, m1, m2])
        params_new, time_step, next_time_step = adaptive_RK4_step(params, bs.relative_inclination, bs.mass_transfer_rate, CBD, conservative_CBD, time_step, r1, r2, self)
        total_time_passed += time_step
        
        #when eccentric mass transfer is implemented for the outer orbit, this should probably move inside of the RK4
        adot_outer = calculate_adot_AM(a_outer, m3, m1+m2, bs.mass_transfer_rate, conservative_CBD) 
        a_outer = a_outer + adot_outer*time_step  
        
        #do not do this before calculating the outer orbit as it resets m1 & m2
        a_inner, e_inner, mbin, m1, m2 = params_new
        e_inner = max(e_inner, 0)
                    
        #assuming no core growth due to stellar evolution
        dm3 = abs(bs.mass_transfer_rate) * time_step 
        m3 = m3 - dm3
        m3_env = max(0|units.MSun, m3_env - dm3) 

        # Check for RLOF in inner binary. If so: merge.
        Rl_accretor_child1 = roche_radius_dimensionless(m1, m2)*a_inner*(1-e_inner)
        Rl_accretor_child2 = roche_radius_dimensionless(m2, m2)*a_inner*(1-e_inner)

        #note that TRES object itself hasn't changed yet -> original outer orbit will therefore not change in adjust_system_after_ce_in_inner_binary                                                                               
        #any mass accretion prior to the merger is currently ignored: Silvia 
        #i can add mass, just need to reset previous mass as well. to prevent outer orbit from changing 
        if (r1 > Rl_accretor_child1) or (r2 > Rl_accretor_child2):#assuming radii of inner stars haven't changed
            donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
            #reduce_mass not subtrac mass, want geen adjust_donor_radius #maybe doesn't matter here 
            #check if star changes type     
            donor_in_stellar_code.change_mass(-1*(donor.mass - m3 +(small_numerical_error|units.MSun)), 0.|units.yr)    
            if not INCLUDE_OUTFLOW_CBD_IN_TSMT: 
                #stellar type may change
                accretor_child1_in_stellar_code = accretor.child1.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
                accretor_child1_in_stellar_code.change_mass(-1*(accretor.child1.mass - m1 +(small_numerical_error|units.MSun)), -1.|units.yr)    
                accretor_child2_in_stellar_code = accretor.child2.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
                accretor_child2_in_stellar_code.change_mass(-1*(accretor.child2.mass - m2 +(small_numerical_error|units.MSun)), -1.|units.yr)    
                accretor.previous_mass = self.get_mass(accretor) 
#                self.stellar_code.evolve_model(minimum_time_step) #to get updates radii, not just inflation of stars due to accretion#needed? silvia
            self.copy_from_stellar()
                   
            if (r1 > Rl_accretor_child1):
                accretor.child1.is_donor = True
                stopping_condition = perform_inner_merger(accretor, accretor.child1, accretor.child2, self) 
            else:                 
                accretor.child2.is_donor = True
                stopping_condition = perform_inner_merger(accretor, accretor.child2, accretor.child1, self) 

            self.instantaneous_evolution = True #skip secular evolution    
            return stopping_condition

        time_step = next_time_step
        # Check for change of TMT regimes 
        CBD, CBD_fraction = CBD_check(m3, m1+m2, a_outer, e_outer, a_inner, e_inner)
        if (CBD != CBD_prev): # In case a transition from CBD to BA occurs or vice versa, we manually reduce the time step
            time_step = min(time_step, initial_time_step)
        if (not INCLUDE_OUTFLOW_CBD_IN_TSMT) and (CBD): 
            conservative_CBD = True            

#       to work this requires updating the parameters of the triple object, not just the parameters in this function
#         Printing data if transition from ballistic accretion to CBD occurs or vice versa
#        if (CBD != CBD_prev) and (not GW): 
#            self.save_snapshot()       
#
#         Printing data if transition to or out of GW regime occurs
#        q_inner = params_new[3]/params_new[4]
#        if q_inner < 1:
#           q_inner = 1./q_inner
#        adot_GDF = calculate_adot_GDF(a_inner, e_inner, q_inner, m1, m2, m1+m2, CBD)
#        adot_CBD = calculate_adot_CBD(a_inner, e_inner, q_inner, bs.relative_inclination, m1+m2, -1*bs.mass_transfer_rate, CBD, self)
#        adot_GW = calculate_adot_GW(a_inner, e_inner, m1, m2)    
#        GW = False
#        if abs(adot_GW) > max(abs(adot_GDF), abs(adot_CBD)):
#           GW = True
#        if (GW != GW_prev): 
#            self.save_snapshot()       
            
    #small difference between total_time_passed and timestep taken by TRES/stellar evolution
#    print(total_time_passed, self.triple.time-self.previous_time) 
        
    bs.semimajor_axis = a_outer
    bs.eccentricity = e_outer
    accretor.semimajor_axis = a_inner
    accretor.eccentricity = e_inner

    donor_in_stellar_code = donor.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
    #reduce_mass not subtrac mass, want geen adjust_donor_radius #maybe doesn't matter here 
    #check if star changes type     
    donor_in_stellar_code.change_mass(-1*(donor.mass - m3 +(small_numerical_error|units.MSun)), 0.|units.yr)    

    if not INCLUDE_OUTFLOW_CBD_IN_TSMT: 
        #stellar type may change
        accretor_child1_in_stellar_code = accretor.child1.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
        accretor_child1_in_stellar_code.change_mass(-1*(accretor.child1.mass - m1 +(small_numerical_error|units.MSun)), -1.|units.yr)    
        accretor_child2_in_stellar_code = accretor.child2.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
        accretor_child2_in_stellar_code.change_mass(-1*(accretor.child2.mass - m2 +(small_numerical_error|units.MSun)), -1.|units.yr)    

    self.stellar_code.evolve_model(minimum_time_step) #to get updates radii, not just inflation of stars due to accretion
    self.copy_from_stellar()

    donor.is_donor = False
    bs.is_mt_stable = True
    bs.bin_type = bin_type['detached']
    self.instantaneous_evolution = True #skip secular evolution    
    self.save_snapshot()   
    return True



def triple_common_envelope_phase(bs, donor, accretor, self):
    # mass transfer of both inner and outer orbit is not yet considered here
    
    # orbital evolution is being taken into account in secular_code        
    if REPORT_FUNCTION_NAMES:
        print('Triple common envelope')

    bs.bin_type = bin_type['common_envelope']                
    self.save_snapshot()       
    
    #implementation is missing
    return False

#when the tertiary star transfers mass to the inner binary
def outer_mass_transfer(bs, donor, accretor, self):
#only for stellar systems consisting of a star and a binary
    if REPORT_FUNCTION_NAMES:
        print('Triple mass transfer')
        bs.semimajor_axis, donor.mass, self.get_mass(accretor), donor.stellar_type

    if bs.is_mt_stable:
        stopping_condition = triple_stable_mass_transfer(bs, donor, accretor, self)
        
        # possible the outer binary needs part_dt_mt as well. 
        #adjusting triple is done in secular evolution code
    else:        
        stopping_condition = triple_common_envelope_phase(bs, donor, accretor, self)


    #stopping condition 0:False, 1:True, -1: calculate through outer mass transfer - effect on inner & outer orbit is taken care off here. 
    return stopping_condition            

#-------------------------

#-------------------------
#Functions for detached evolution
## Calculates stellar wind velocoty.
## Steller wind velocity is 2.5 times stellar escape velocity
#def wind_velocity(star):
#    v_esc2 = constants.G * star.mass / star.radius
#    return 2.5*np.sqrt(v_esc2)
#}
#
#
## Bondi, H., and Hoyle, F., 1944, MNRAS 104, 273 (wind accretion.
## Livio, M., Warner, B., 1984, The Observatory 104, 152.
#def accretion_efficiency_from_stellar_wind(accretor, donor):
#velocity needs to be determined -> velocity average?
# why is BH dependent on ecc as 1/np.sqrt(1-e**2)

#    alpha_wind = 0.5
#    v_wind = wind_velocity(donor)
#    acc_radius = (constants.G*accretor.mass)**2/v_wind**4
#    
#    wind_acc = alpha_wind/np.sqrt(1-bs.eccentricity**2) / bs.semimajor_axis**2
#    v_factor = 1/((1+(velocity/v_wind)**2)**3./2.)
#    mass_fraction = acc_radius*wind_acc*v_factor
#
#    print('mass_fraction:', mass_fraction)
##    mass_fraction = min(0.9, mass_fraction)
#



def detached(bs, self):
    # orbital evolution is being taken into account in secular_code        
    if REPORT_FUNCTION_NAMES:
        print('Detached')

    if bs.bin_type == bin_type['detached'] or bs.bin_type == bin_type['unknown']:
        bs.bin_type = bin_type['detached']
    else:
        bs.bin_type = bin_type['detached']
        self.save_snapshot()        
    
    # wind mass loss is done by stellar_code
    # wind accretion here:
    # update accretion efficiency of wind mass loss
    if self.is_binary(bs):
            
        bs.accretion_efficiency_wind_child1_to_child2 = 0.0
        bs.accretion_efficiency_wind_child2_to_child1 = 0.0

#        child1_in_stellar_code = bs.child1.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
#        child2_in_stellar_code = bs.child2.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
#
#        dt = self.triple.time - self.previous_time
#        dm_child1_to_child2 = -1 * child1.wind_mass_loss_rate * bs.accretion_efficiency_wind_child1_to_child2 * dt
#        child2_in_stellar_code.change_mass(dm_child1_to_child2, -1*dt)
#        dm_child12to_child1 = -1 * child2.wind_mass_loss_rate * bs.accretion_efficiency_wind_child2_to_child1 * dt
#        child1_in_stellar_code.change_mass(dm_child2_to_child1, -1*dt)
# check if this indeed is accreted conservatively        


    elif bs.child1.is_star and self.is_binary(bs.child2):
        #Assumption: an inner binary is not effected by wind from an outer star
        bs.accretion_efficiency_wind_child1_to_child2 = 0.0

        bs.accretion_efficiency_wind_child2_to_child1 = 0.0
        
#        child1_in_stellar_code = bs.child1.as_set().get_intersecting_subset_in(self.stellar_code.particles)[0]
#        dt = self.triple.time - self.previous_time
        
         #effect of wind from bs.child2.child1 onto bs.child1
#        mtr_w_in1_1 =  bs.child2.child1.wind_mass_loss_rate * (1-bs.child2.accretion_efficiency_wind_child1_to_child2)       
#        beta_w_in1_1 = 0.0
#        dm_in1_1 = -1 * mtr_w_in1_1 * beta_w_in1_1 * dt
#        
         #effect of wind from bs.child2.child2 onto bs.child1
#        mtr_w_in2_1 =  bs.child2.child2.wind_mass_loss_rate * (1-bs.child2.accretion_efficiency_wind_child2_to_child1)       
#        beta_w_in2_1 = 0.0
#        dm_in2_1 = -1 * mtr_w_in2_1 * beta_w_in2_1 * dt
#                    
#        dm = dm_in1_1 + dm_in2_1  
#        mtr = mtr_w_in1_1 + mtr_w_in2_1)


         #effect of mass transfer in the binary bs.child2 onto bs.child1
#        if bs.child2.child1.is_donor and bs.child2.child2.is_donor:
#            print('contact binary in detached...')
#            exit(1)
#        elif bs.child2.child1.is_donor or bs.child2.child2.is_donor:
#            #Assumption:
#            #Stable mass transfer in the inner binary, affects the outer binary as a wind.
#            mtr_rlof_in_1 = bs.child2.mass_transfer_rate * (1-bs.child2.accretion_efficiency_mass_transfer)
#            beta_rlof_in_1 = 0.0
#            dm_rlof_in_1 = -1 * mtr_rlof_in_1 * beta_rlof_in_1 * dt
#            dm += dm_rlof_in_1
#            mtr += mtr_rlof_in_1 

#        bs.accretion_efficiency_wind_child2_to_child1 = dm / ( mtr* -1 * dt)
            
#        child1_in_stellar_code.change_mass(dm, dt)
# check if this indeed is accreted conservatively        

    else:
        self.save_snapshot()
        print('detached: type of system unknown')
        print( bs.child1.is_star, bs.child2.is_star)
        sys.exit('error in detached')                    
              
    #reset parameters after mass transfer
#    bs.mass_transfer_rate = 0.0 | units.MSun/units.yr

#    return bs
#-------------------------

#-------------------------
def perform_stellar_interaction(bs, self):
   if REPORT_FUNCTION_NAMES:
        print('Perform stellar interaction')
    
   stopping_condition = True 
   if not bs.is_star and bs.child1.is_star:
        if REPORT_BINARY_EVOLUTION:
            Rl1 = roche_radius(bs, bs.child1, self)
            print("Check for RLOF:", bs.child1.mass, bs.child1.previous_mass)
            print("Check for RLOF:", Rl1, bs.child1.radius)
                
        if bs.child2.is_star:
            if REPORT_BINARY_EVOLUTION:
                Rl2 = roche_radius(bs, bs.child2, self)
                print("Check for RLOF:", bs.child2.mass, bs.child2.previous_mass)
                print("Check for RLOF:", Rl2, bs.child2.radius)
                
#            if bs.child1.is_donor or bs.child2.is_donor:
#                print("start mt")
#                exit(0)

            if bs.child1.is_donor and bs.child2.is_donor:
                stopping_condition = contact_system(bs, bs.child1, bs.child2, self)
            elif bs.child1.is_donor and not bs.child2.is_donor:
                stopping_condition = semi_detached(bs, bs.child1, bs.child2, self)
            elif not bs.child1.is_donor and bs.child2.is_donor:
                stopping_condition = semi_detached(bs, bs.child2, bs.child1, self)
            else:
                detached(bs, self)
                                        
        elif not bs.child2.is_star:
            if REPORT_BINARY_EVOLUTION:
                print(self.get_mass(bs), bs.child1.mass, self.get_mass(bs.child2))
    
            if bs.child1.is_donor:
#                if bs.child2.child1.is_donor or bs.child2.child2.is_donor:
#                    print(3)
#                    stopping_condition = outer_mass_transfer(bs, bs.child1, bs.child2, self)
                stopping_condition = outer_mass_transfer(bs, bs.child1, bs.child2, self)
            else:
                detached(bs, self)
                
        else:
            print(bs.is_star, bs.child1.is_star, bs.child2.is_star)
            sys.exit('error in perform stellar interaction, system type unknown') 
                               
   else:
        print(bs.is_star, bs.child1.is_star, bs.child1.is_donor)
        sys.exit('error in perform stellar interaction, system type unknown') 
        
   return stopping_condition            
        
#-------------------------
        
#-------------------------
#functions for the stability of mass transfer
def q_crit(self, donor, companion):
    #following Hurley, Tout, Pols 2002
    if donor.stellar_type in [9]|units.stellar_type:
#    if donor.stellar_type in [8,9]|units.stellar_type:
        return 0.784
    elif donor.stellar_type in [3,4,5,6]|units.stellar_type:
        x=0.3
        return (1.67-x+2*(donor.core_mass/donor.mass)**5)/2.13
    elif donor.stellar_type == 0|units.stellar_type:
        return 0.695
    elif donor.stellar_type == 1|units.stellar_type:
        return 1./0.625 #following claeys et al. 2014 based on de mink et al 2007
    elif donor.stellar_type in stellar_types_compact_objects:#eventhough ns & bh shouldn't be donors... 
        return 0.25 #0.628
    elif donor.stellar_type in [18,19]|units.stellar_type:#planet or brown dwarf. 
        #metzger et al 2012,425,2778, 
        return 1. * (donor.radius/self.get_size(companion))**3
    else: #stellar type 2, and 8
        return 3 # high for hg?
        
    

def mass_transfer_stability(binary, self):
    if REPORT_FUNCTION_NAMES:
        print('Mass transfer stability')

    if self.is_binary(binary):
        Js_1 = self.spin_angular_momentum(binary.child1)
        Js_2 = self.spin_angular_momentum(binary.child2)        
        Jb = self.orbital_angular_momentum(binary)
        if REPORT_MASS_TRANSFER_STABILITY:
            print("Mass transfer stability: Binary ")
            print(binary.semimajor_axis, binary.child1.mass, binary.child2.mass, binary.child1.stellar_type, binary.child2.stellar_type)
            print(binary.child1.spin_angular_frequency, binary.child2.spin_angular_frequency)
            print(Js_1, Js_2, Jb, Jb/3.   )
        
        Js = max(Js_1, Js_2)
        if Js >= Jb/3. :
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Darwin Riemann instability", Js_1, Js_2, Jb, Jb/3.)
            mt1 = -1.* binary.child1.mass / dynamic_timescale(binary.child1)
            mt2 = -1.* binary.child2.mass / dynamic_timescale(binary.child2)  
            binary.mass_transfer_rate = min(mt1, mt2) # minimum because mt<0
            binary.is_mt_stable = False
            
        elif binary.child1.is_donor and binary.child2.is_donor:
            if binary.child1.is_OLOF_donor and binary.child2.is_OLOF_donor:
                if REPORT_MASS_TRANSFER_STABILITY:
                    print("Mass transfer stability: Unstable OLOF contact")
                mt1 = -1.* binary.child1.mass / dynamic_timescale(binary.child1)
                mt2 = -1.* binary.child2.mass / dynamic_timescale(binary.child2)  
                binary.mass_transfer_rate = min(mt1, mt2) # minimum because mt<0
                binary.is_mt_stable = False
            #already included in next elif + mass transfer rate doesn't matter for olof     
            # elif binary.child1.is_OLOF_donor and binary.child1.stellar_type <= 1|units.stellar_type:
            #     if REPORT_MASS_TRANSFER_STABILITY:
            #         print("Mass transfer stability: Stable OLOF")
            #     binary.mass_transfer_rate = -1.* binary.child1.mass / nuclear_evolution_timescale(binary.child1)
            #     binary.is_mt_stable = True
            # elif binary.child2.is_OLOF_donor and binary.child2.stellar_type <= 1|units.stellar_type:
            #     if REPORT_MASS_TRANSFER_STABILITY:
            #         print("Mass transfer stability: Stable OLOF")
            #     binary.mass_transfer_rate = -1.* binary.child2.mass / nuclear_evolution_timescale(binary.child2)
            #     binary.is_mt_stable = True 
            elif binary.child1.stellar_type <= 1|units.stellar_type and binary.child2.stellar_type <= 1|units.stellar_type:
                if REPORT_MASS_TRANSFER_STABILITY:
                    print("Mass transfer stability: stable case A contact")
                mt1 = -1.* binary.child1.mass / nuclear_evolution_timescale(binary.child1)         
                mt2 = -1.* binary.child2.mass / nuclear_evolution_timescale(binary.child2)         
                binary.mass_transfer_rate = min(mt1, mt2) # minimum because mt<0
                binary.is_mt_stable = False
            else:
                if REPORT_MASS_TRANSFER_STABILITY:
                    print("Mass transfer stability: Unstable contact")
                mt1 = -1.* binary.child1.mass / dynamic_timescale(binary.child1)
                mt2 = -1.* binary.child2.mass / dynamic_timescale(binary.child2)  
                binary.mass_transfer_rate = min(mt1, mt2) # minimum because mt<0
                binary.is_mt_stable = False
        elif binary.child1.is_donor and binary.child1.mass > binary.child2.mass*q_crit(self, binary.child1, binary.child2):
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Mdonor1>Macc*q_crit at q_crit = ", q_crit(self, binary.child1, binary.child2))
            binary.mass_transfer_rate = -1.* binary.child1.mass / dynamic_timescale(binary.child1)
            binary.is_mt_stable = False
        elif binary.child2.is_donor and binary.child2.mass > binary.child1.mass*q_crit(self, binary.child2, binary.child1):
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Mdonor2>Macc*q_crit at q_crit = ", q_crit(self, binary.child2, binary.child1))
            binary.mass_transfer_rate= -1.* binary.child2.mass / dynamic_timescale(binary.child2) 
            binary.is_mt_stable = False
            
        elif binary.child1.is_donor:
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Donor1 stable ")
            binary.mass_transfer_rate = -1.* binary.child1.mass / mass_transfer_timescale(binary, binary.child1)         
            binary.is_mt_stable = True
        elif binary.child2.is_donor:
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Donor2 stable")
            binary.mass_transfer_rate = -1.* binary.child2.mass / mass_transfer_timescale(binary, binary.child2)         
            binary.is_mt_stable = True
            
        else:     
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Detached")
            #detached system
            mt1 = -1.* binary.child1.mass / mass_transfer_timescale(binary, binary.child1)
            mt2 = -1.* binary.child2.mass / mass_transfer_timescale(binary, binary.child2)  
            binary.mass_transfer_rate = min(mt1, mt2) # minimum because mt<0
            binary.is_mt_stable = True

    else:
        if binary.child1.is_star and not binary.child2.is_star:
            star = binary.child1
            companion = binary.child2
        elif binary.child2.is_star and not binary.child1.is_star:
            star = binary.child2
            companion = binary.child1
        else: 
            print(binary.is_star, binary.child1.is_star, binary.child2.is_star)
            sys.exit('error in Mass transfer stability: type of system unknown') 

            
        if REPORT_MASS_TRANSFER_STABILITY:
            print("Mass transfer stability: Binary ")
            print(binary.semimajor_axis, self.get_mass(companion), star.mass, star.stellar_type)
    
        Js = self.spin_angular_momentum(star)
        Jb = self.orbital_angular_momentum(binary)
        
        if Js >= Jb/3. :
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Darwin Riemann instability: ", Js, Jb, Jb/3.)
            binary.mass_transfer_rate = -1.* star.mass / dynamic_timescale(star)
            binary.is_mt_stable = False          
            
        elif star.is_donor and star.mass > self.get_mass(companion)*q_crit(self, star, companion):
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Mdonor1>Macc*q_crit at q_crit = ", q_crit(self, star, companion))
            binary.mass_transfer_rate = -1.* star.mass / dynamic_timescale(star)
            binary.is_mt_stable = False
            
        elif star.is_donor:
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Donor1 stable ")
            binary.mass_transfer_rate = -1.* star.mass / mass_transfer_timescale(binary, star)         
            binary.is_mt_stable = True
            
        else:                     
            if REPORT_MASS_TRANSFER_STABILITY:
                print("Mass transfer stability: Detached")
            #detached system
            binary.mass_transfer_rate = -1.* star.mass / mass_transfer_timescale(binary, star)
            binary.is_mt_stable = True
                        
            
       
       
def mass_transfer_timescale(binary, star):
    if REPORT_FUNCTION_NAMES:
        print('Mass transfer timescale')
    
    if not star.is_star:
        sys.exit('error in mass transfer timescale:  type of system unknown, donor star is not a star')
    
    #For now thermal timescale donor
    mtt = kelvin_helmholds_timescale(star)
#    mtt = nuclear_evolution_timescale(star)
    return mtt
#-------------------------

# Mass loss recipes for the energy limited photoevaporation of planets.
        
# X-ray luminosity fraction as prescripted by Wright 2011, based on the Rossby number.
def Rx_wright11(mass, p_rot):
    Rx_sat = 10**(-3.13)
    Ro_sat = 0.16
    tau_conv = 10**(1.16 - 1.49* np.log(mass.value_in(units.MSun)) - 0.54*(np.log(mass.value_in(units.MSun)))**2)
    Ro = p_rot/tau_conv
    if Ro > Ro_sat:
        B = -2.70
        R_X = Rx_sat * (Ro / Ro_sat)**B
    else:
        R_X = Rx_sat
    return R_X

# compute the specific flux of a BB, in erg/s /m3 (/sterad)
def blackbody(wavel, T):
    h = constants.h.value_in( units.erg * units.s )
    c = constants.c.value_in( units.m / units.s )
    KB = constants.kB.value_in( units.erg / units.K )
    B_l = (2* h * c**2 / wavel**5) / (np.exp( h*c/(wavel*KB*T), dtype=np.float128) - 1)
    return B_l
        
    
# Compute the high energy luminosity from the bolometric one.
def xuv_luminosity(star):
    L_bol = star.luminosity.value_in(units.erg/units.s)	

    #silvia: right now WDs+NS, should it just be wds? include bhs? 
    if star.stellar_type.value in [10,11,12]:     # we have a WD, we integrate a Black Body from 1 nm to 91.2 nm
        F_xuv = integrate.quad(blackbody, 1e-09, 9.12e-08, args=(star.temperature.value_in(units.K)))[0]
        L_XUV = 4*np.pi**2 * star.radius.value_in(units.m)**2 * F_xuv
        # print('M WD:', star.mass, '\t T WD:', star.temperature)
        return L_XUV    # erg/s
    elif star.stellar_type in stellar_types_planetary_objects:
        return 0; 
    elif star.stellar_type in stellar_types_SN_remnants: #NS, BH or massless SN
        return 0; 
    else: 
        if (star.mass <= 2|units.MSun):
            p_rot_star = 2*np.pi / star.spin_angular_frequency.value_in(1/units.s)
            L_X = L_bol * Rx_wright11(star.mass, p_rot_star)             # Rossby number approach, Wright 2011
            L_EUV = 10**4.8 * L_X**0.86                 # Sanz-Forcada 2011 
        elif 2|units.MSun < star.mass <= 3|units.MSun:
            if star.stellar_type.value in [1,2,7,8]: #ms & hg : radiative envelope            
                L_X = 10**(-3.5) * L_bol 	            # Flaccomio 2003   
            elif star.stellar_type.value in [3,4,5,6,9]:  # during giant phases, rossby approach again, having convective envelopes
                p_rot_star = 2*np.pi / star.spin_angular_frequency.value_in(1/units.s)
                L_X = L_bol * Rx_wright11(star.mass, p_rot_star)
            else: 
                sys.exit('stellar type unknown in xuv_luminosity')
            L_EUV = 10**4.8 * L_X**0.86                 # Sanz-Forcada 2011

        elif 3|units.MSun < star.mass < 10|units.MSun:
            L_X = 1e-06 * L_bol  # 10**31 	# erg/s     #Flaccomio 2003
            L_EUV = L_X 	# actually EUV should be stronger than X emission in this mass range
        else:
            # Star mass out of implemented range for evaporation: default factor employed
            L_X = 1e-06 * L_bol
            L_EUV = 1e-06 * L_bol
        
        return (L_X + L_EUV ) #|units.erg/units.s		# erg/s
    
    
# Compute instantaneous flux at time t from given star, for a planet in circular orbit.
# lum input has to be erg/s
def flux_inst(t, r_plan, a_st_i, P_plan, P_binary, lum, star_number, i_orb ):
    phi = 2*np.pi * t / P_plan                              # planet's phase angle (inclined)
    st_ang = 2*np.pi* t / P_binary + star_number * np.pi      # star phase angle (on plane)
    d_z2 = ( r_plan * np.sin(phi) * np.sin(i_orb) )**2
    d_p2 = ( r_plan *np.cos(phi) - a_st_i *np.cos(st_ang) )**2 + ( r_plan *np.sin(phi)*np.cos(i_orb) - a_st_i *np.sin(st_ang) )**2
    distance_sq = d_p2 + d_z2
    return lum/distance_sq
    
    
def mass_lost_due_to_evaporation_tertiary(stellar_system, dt, outer_planet, inner_binary, self):
    if REPORT_FUNCTION_NAMES:
        print("Mass lost due to evaporation tertiary")
    
    e_pl = stellar_system.eccentricity
    a_pl = stellar_system.semimajor_axis						
    P_pl = self.orbital_period(stellar_system)
    i_orbits = stellar_system.relative_inclination
    # time-averaged circular radius of elliptical orbits
    r_pl = a_pl * ( 1 + 0.5* e_pl**2 )          

    a_bin = inner_binary.semimajor_axis
    P_bin = self.orbital_period(inner_binary)
    M_bin = self.get_mass(inner_binary)        
    xi = roche_radius_dimensionless(outer_planet.mass, M_bin) * a_pl / outer_planet.radius
    #Erkaev (2007) escape factor
    K_Erk = 1 - 1.5/xi + 0.5* xi**(-3)		

    # compute the high-energy flux average from the two stars
    t_start = 0.
    #average on one orbital period of the outer planet (~ 5 P_binary)
    t_end = P_pl.value_in(units.Myr)		

    L_xuv1_erg_s = xuv_luminosity(inner_binary.child1)
    a_st_1 = a_bin/(1+inner_binary.child1.mass/inner_binary.child2.mass)
    F1 = integrate.quad(flux_inst, t_start, t_end, args=(r_pl.value_in(units.RSun), a_st_1.value_in(units.RSun), P_pl.value_in(units.Myr), P_bin.value_in(units.Myr), L_xuv1_erg_s, 0, i_orbits), limit=100, full_output=1)[0]

    L_xuv2_erg_s = xuv_luminosity(inner_binary.child2)
    a_st_2 = a_bin/(1+inner_binary.child2.mass/inner_binary.child1.mass)
    F2 = integrate.quad(flux_inst, t_start, t_end, args=(r_pl.value_in(units.RSun), a_st_2.value_in(units.RSun), P_pl.value_in(units.Myr), P_bin.value_in(units.Myr), L_xuv2_erg_s, 1, i_orbits), limit=100, full_output=1)[0]
    
    Flux_XUV = (F1+F2)/(4*np.pi* (t_end-t_start) )  | (units.erg/units.s / units.RSun**2)
    eta = 0.2 # evaporation efficiency parameter
    M_dot = eta * np.pi * (outer_planet.radius)**3 * Flux_XUV / ( constants.G * outer_planet.mass * K_Erk )
    mass_lost = M_dot * dt
    return mass_lost


def mass_lost_due_to_evaporation_in_binary(stellar_system, dt, planet, star, self):
    if REPORT_FUNCTION_NAMES:
        print("Mass lost due to evaporation binary")

    e_pl = stellar_system.eccentricity
    a_pl = stellar_system.semimajor_axis						
    P_pl = self.orbital_period(stellar_system)
    i_orbits = stellar_system.relative_inclination
    # time-averaged circular radius of elliptical orbits
    r_pl = a_pl * ( 1 + 0.5* e_pl**2 )          

    xi = roche_radius_dimensionless(planet.mass, star.mass) * a_pl / planet.radius
    #Erkaev (2007) escape factor
    K_Erk = 1 - 1.5/xi + 0.5* xi**(-3)		
    Flux_XUV = star.luminosity / r_pl**2 

    eta = 0.2 # evaporation efficiency parameter
    M_dot = eta * np.pi * (planet.radius)**3 * Flux_XUV / ( constants.G * planet.mass * K_Erk )
    mass_lost = M_dot * dt
    return mass_lost

