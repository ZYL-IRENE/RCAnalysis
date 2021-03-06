 #!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import division
import warnings
import copy
import numpy as np
import math


def distance_euclidean(instance1, instance2):

    def detect_value_type(attribute):
      
        from numbers import Number
        attribute_type = None
        if isinstance(attribute, Number):
            attribute_type = float
            attribute = float(attribute)
        else:
            attribute_type = str
            attribute = str(attribute)
        return attribute_type, attribute
    # check if instances are of same length
    if len(instance1) != len(instance2):
        raise AttributeError("Instances have different number of arguments.")
    # init differences vector
    differences = [0] * len(instance1)
    # compute difference for each attribute and store it to differences vector
    for i, (attr1, attr2) in enumerate(zip(instance1, instance2)):
        type1, attr1 = detect_value_type(attr1)
        type2, attr2 = detect_value_type(attr2)
        # raise error is attributes are not of same data type.
        if type1 != type2:
            raise AttributeError("Instances have different data types.")
        if type1 is float:
            # compute difference for float
            differences[i] = attr1 - attr2
        else:
            # compute difference for string
            if attr1 == attr2:
                differences[i] = 0
            else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                differences[i] = 1
    # compute RMSE (root mean squared error)
    rmse = (sum(map(lambda x: x**2, differences)) / len(differences))**0.5
    return rmse

class LOF:
   
    def __init__(self, instances, normalize=True, distance_function=distance_euclidean):
        self.instances = instances
        self.normalize = normalize
        self.distance_function = distance_function
        self.initialize_k_list()
        if normalize:
            self.normalize_instances()

    def compute_instance_attribute_bounds(self):
        min_values = [float("inf")] * len(self.instances[0]) #n.ones(len(self.instances[0])) * n.inf
        max_values = [float("-inf")] * len(self.instances[0]) #n.ones(len(self.instances[0])) * -1 * n.inf
        for instance in self.instances:
            min_values = tuple(map(lambda x,y: min(x,y), min_values,instance)) #n.minimum(min_values, instance)
            max_values = tuple(map(lambda x,y: max(x,y), max_values,instance)) #n.maximum(max_values, instance)

        diff = [dim_max - dim_min for dim_max, dim_min in zip(max_values, min_values)]
        if not all(diff):
            problematic_dimensions = ", ".join(str(i+1) for i, v in enumerate(diff) if v == 0)
            warnings.warn("No data variation in dimensions: %s. You should check your data or disable normalization." % problematic_dimensions)

        self.max_attribute_values = max_values
        self.min_attribute_values = min_values

    def normalize_instances(self):
        """Normalizes the instances and stores the infromation for rescaling new instances."""
        if not hasattr(self, "max_attribute_values"):
            self.compute_instance_attribute_bounds()
        new_instances = []
        for instance in self.instances:
            new_instances.append(self.normalize_instance(instance)) # (instance - min_values) / (max_values - min_values)
        self.instances = new_instances

    def normalize_instance(self, instance):
        return tuple(map(lambda value,max,min: (value-min)/(max-min) if max-min > 0 else 0,
                         instance, self.max_attribute_values, self.min_attribute_values))

    def local_outlier_factor(self, min_pts, instance):
        if self.normalize:
            instance = self.normalize_instance(instance)
        return local_outlier_factor(min_pts, instance, self.instances, distance_function=self.distance_function)

    def  initialize_k_list(self):
        data_size = len(self.instances)
        self.k_list = []
        k = 2
        while k <data_size:
            self.k_list.append(k)
            k*=2

    def local_outlier_factor_kinf(self,instance):
        min_lof = float("inf")
        k_inf = 2
        lof_list = []
        for k in self.k_list:
            lof = self.local_outlier_factor(k,instance)
            lof_list.append(lof)
        min_lof = min(lof_list)
        k_inf = self.k_list[lof_list.index(min_lof)]
        return min_lof,k_inf,lof_list

    '''def confidence1(self, instance, k_inf, lof_list, min_lof):
        if self.k_list.index(k_inf) == (len(self.k_list)-1):# if k_inf is the last k in k list
            confidence1 = 1
        else:
            confidence1 = lof_list[lof_list.index(min_lof)+1] / min_lof

        return confidence1'''

    '''def confidence2(self,instance, k_inf, lof_list, min_lof):
        instances_without_instance = set(self.instances)
        instances_without_instance.discard(instance)
        (k_distance_value, neighbours) = k_distance(k_inf, instance, instances_without_instance, **kwargs)
        sum_c1 = 0
        for neighbour in neighbours:
            (min_lof_b, kb_inf_b, lof_list_b) = local_outlier_factor_kinf(neighbour)
            sum_c1 += self.confidence1(neighbour, kb_inf_b, lof_list_b, min_lof_b)
        confidence2 = sum_c1 / len(neighbours)
        return confidence2'''

    '''def confidence3(self,instance, k_inf, lof_list, min_lof):
        k_avg = (self.k_list[0] + k_inf) / 2

        instances_without_instance = set(self.instances)
        instances_without_instance.discard(instance)
        if self.k_list.index(k_inf) == (len(self.k_list)-1):
            (kinfp1_distance_value, neighbours) = k_distance(k_inf, instance, instances_without_instance, **kwargs)
        else:
            (kinfp1_distance_value, neighbours) = k_distance(self.k_list[self.k_list.index(k_inf)+1], instance, instances_without_instance, **kwargs)
        
        (kavg_distance_value, neighbours) = k_distance(k_avg, instance, instances_without_instance, **kwargs)
        return kinfp1_distance_value / kavg_distance_value'''

    '''def confidence4(self,instance, k_inf, lof_list, min_lof):
        instances_without_instance = set(self.instances)
        instances_without_instance.discard(instance)
        (k_distance_value, neighbours) = k_distance(k_inf, instance, instances_without_instance, **kwargs)
        sum_kb = 0
        for neighbour in neighbours:
            (min_lof_b, k_inf_b, lof_list_b) = local_outlier_factor_kinf(neighbour)
            sum_kb += k_inf_b
        avg_kb = sum_kb / len(neighbour)
        return math.exp((k_inf / avg_kb) - 1)'''

    def confidence(self,instance, k_inf, min_lof, lof_list):
        #instances_without_instance = set(self.instances)
        if self.normalize:
            instance = self.normalize_instance(instance)
        instances_without_instance = copy.deepcopy(self.instances)
        instances_without_instance.remove(instance)
        (k_distance_value, neighbours) = k_distance(k_inf, instance, instances_without_instance, distance_function=distance_euclidean)

        #confidence1
        if self.k_list.index(k_inf) == (len(self.k_list)-1):# if k_inf is the last k in k list
            confidence1 = 1
        else:
            confidence1 = lof_list[lof_list.index(min_lof)+1] / min_lof
        
        #confidence2 & confidence4
        sum_c1_b = 0
        sum_kb = 0
        for neighbour in neighbours:
            (min_lof_b, k_inf_b, lof_list_b) = self.local_outlier_factor_kinf(neighbour)
            if self.k_list.index(k_inf_b) == (len(self.k_list)-1):# if k_inf is the last k in k list
                confidence1_b = 1
            else:
                confidence1_b = lof_list_b[lof_list_b.index(min_lof_b)+1] / min_lof_b
            sum_c1_b += confidence1_b #confidence2
            sum_kb += k_inf_b #confidence4
        confidence2 = sum_c1_b / len(neighbours)
        avg_kb = sum_kb / len(neighbour) #confidence4
        confidence4 = math.exp((k_inf / avg_kb) - 1)

        #confidence3
        k_avg = math.floor((self.k_list[0] + k_inf) / 2)
        if self.k_list.index(k_inf) == (len(self.k_list)-1):
            #(kinfp1_distance_value, neighbours) = k_distance(k_inf, instance, instances_without_instance, **kwargs)
            kinfp1_distance_value = k_distance_value
        else:
            (kinfp1_distance_value, neighbours_kp1) = k_distance(self.k_list[self.k_list.index(k_inf)+1], instance, instances_without_instance, distance_function=distance_euclidean)
        (kavg_distance_value, neighbours_kavg) = k_distance(k_avg, instance, instances_without_instance, distance_function=distance_euclidean)
        confidence3 = kinfp1_distance_value / kavg_distance_value

        #confidence
        confidence = (confidence1 * confidence2 * confidence3 * confidence4)**(1/4)
        return confidence


def k_distance(k, instance, instances, distance_function=distance_euclidean):
    distances = {}
    for instance2 in instances:
        distance_value = distance_function(instance, instance2)
        if distance_value in distances:
            distances[distance_value].append(instance2)
        else:
            distances[distance_value] = [instance2]
        
    distances = sorted(distances.items())
    neighbours = []
    for n in distances[:k]:
        neighbours.extend(n[1])
    k_distance_value = distances[k - 1][0] if len(distances) >= k else distances[-1][0]
    return k_distance_value, neighbours

def reachability_distance(k, instance1, instance2, instances, distance_function=distance_euclidean):
    (k_distance_value, neighbours) = k_distance(k, instance2, instances, distance_function=distance_function)
    return max([k_distance_value, distance_function(instance1, instance2)])

def local_reachability_density(min_pts, instance, instances, **kwargs):

    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    reachability_distances_array = [0]*len(neighbours) #n.zeros(len(neighbours))

    for i, neighbour in enumerate(neighbours):
        reachability_distances_array[i] = reachability_distance(min_pts, instance, neighbour, instances, **kwargs)
   
    if not any(reachability_distances_array):
        warnings.warn("Instance %s (could be normalized) is identical to all the neighbors. Setting local reachability density to inf." % repr(instance))
        return float("inf")
    else:
        return len(neighbours) / sum(reachability_distances_array)


def local_outlier_factor(min_pts, instance, instances, **kwargs):

    (k_distance_value, neighbours) = k_distance(min_pts, instance, instances, **kwargs)
    instances_without_instance = set(instances)
    instances_without_instance.discard(instance)# kick instance itself out
    instance_lrd = local_reachability_density(min_pts, instance, instances_without_instance, **kwargs)
    #instance_lrd = local_reachability_density(min_pts, instance, instances, **kwargs)
    lrd_ratios_array = [0]* len(neighbours)
    for i, neighbour in enumerate(neighbours):
        instances_without_instance = set(instances)
        instances_without_instance.discard(neighbour)
        neighbour_lrd = local_reachability_density(min_pts, neighbour, instances_without_instance, **kwargs)
        lrd_ratios_array[i] = neighbour_lrd / instance_lrd
    return sum(lrd_ratios_array) / len(neighbours)

 
def outliers(k,instances, candidate=None,**kwargs):
    """Simple procedure to identify outliers in the dataset."""
    instances_value_backup = copy.deepcopy(instances)
    outliers = []
    if not candidate:
        candidate = instances

    """ import k-means calucate outlier candidate dataset ClOF """
    l = LOF(instances, **kwargs)
    for i,instance in enumerate(candidate):
        #instance = tuple(instance) 
        #instances_value_backup.remove(instance)
        #l = LOF(instances_value_backup, **kwargs)
        #instances_value_backup = copy.deepcopy(instances)
        (value,k_inf,lof_list) = l.local_outlier_factor_kinf(instance)
        confidence = l.confidence(instance, k_inf, value, lof_list)
        print(confidence)
        if value > 1.01:
            outliers.append({"lof": value, "instance": instance, "index": instances.index(instance), "kinf": k_inf})
    outliers.sort(key=lambda o: o["lof"], reverse=True)
    return outliers





