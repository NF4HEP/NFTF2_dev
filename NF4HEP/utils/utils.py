import itertools
import numpy as np
from numpy import typing as npt
from sklearn import datasets # type: ignore

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from .custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr

import math
import os
from fpdf import FPDF # type: ignore
from PIL import Image # type: ignore
import sys
from matplotlib import pyplot as plt # type: ignore
from datetime import datetime
from timeit import default_timer as timer

import os
import codecs
import random
import json
import tensorflow as tf # type: ignore
import tensorflow.compat.v1 as tf1 # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras import layers, initializers, regularizers, constraints, callbacks, optimizers, metrics, losses # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Layer #type: ignore
import tensorflow_probability as tfp # type: ignore
tfd = tfp.distributions
tfb= tfp.bijectors
import pandas as pd # type: ignore

from .verbosity import print

header_string = "=============================="
footer_string = "------------------------------"

#class InputError(Exception):
#    """Base class for data error exceptions"""
#    pass#

#class DataError(Exception):
#    """Base class for data error exceptions"""
#    pass#

#class MissingModule(Exception):
#    """Base class for missing package exceptions"""
#    pass

#def flatten_list(l):
#    l = [item for sublist in l for item in sublist]
#    return l

def generate_timestamp():
    return "datetime_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]

def append_without_duplicate(list,element):
    if element not in list:
        list.append(element)
    return list

def make_pdf_from_img(img):
    """Make pdf from image
    Used to circumvent bud in plot_model which does not allow to export pdf"""
    img_pdf = os.path.splitext(img)[0]+".pdf"
    cover = Image.open(img)
    width, height = cover.size
    pdf = FPDF(unit = "pt", format = (width, height))
    pdf.add_page()
    pdf.image(img, 0, 0)
    pdf.output(img_pdf, "F") # type: ignore

def chunks(lst, n):
    """Return list of chunks from lst."""
    res = []
    for i in range(0, len(lst), n):
        res.append(lst[i:i + n])
    return res


def savefig(path,**kwargs):
    """
    Function that patches the ``plt.savefig`` method for long filenames on Windows platforms.
    """
    if 'win32' in sys.platform or "win64" in sys.platform:
        plt.savefig("\\\\?\\" + path, **kwargs)
    else:
        plt.savefig(path, **kwargs)

def build_method_string_from_dict(class_name: Optional[str] = None, 
                                  method_name: Optional[str] = None, 
                                  args: Optional[list] = None, 
                                  kwargs: Optional[dict] = None
                                 ) -> str:
    if class_name is None:
        class_name = ""
    if method_name is None:
        method_name = ""
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    method_string = class_name+"."+method_name+"("
    if args != []:
        for arg in args:
            if isinstance(arg,str):
                if "(" in arg:
                    method_string = method_string+arg+", "
                else:
                    method_string = method_string+"'"+arg+"', "
            else:
                method_string = method_string+str(arg)+", "
    for key, val in kwargs.items():
        if not isinstance(val,dict):
            if "(" in str(val):
                if "initializer" in key:
                    val = str(val).lstrip("initializers.")
                    try:
                        eval("initializers."+val)
                        method_string = method_string+key + "="+"initializers."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "regularizer" in key:
                    val = str(val).lstrip("regularizer.")
                    try:
                        eval("regularizers."+val)
                        method_string = method_string+key + "="+"regularizers."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "constraint" in key:
                    val = str(val).lstrip("constraints.")
                    try:
                        eval("constraints."+val)
                        method_string = method_string+key + "="+"constraints."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "callback" in key:
                    val = str(val).lstrip("callbacks.")
                    try:
                        eval("callbacks."+val)
                        method_string = method_string+key + "="+"callbacks."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "optimizers" in key:
                    val = str(val).lstrip("optimizers.")
                    try:
                        eval("optimizers."+val)
                        method_string = method_string+key + "="+"optimizers."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "losses" in key:
                    val = str(val).lstrip("losses.")
                    try:
                        eval("losses."+val)
                        method_string = method_string+key + "="+"losses."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "metrics" in key:
                    val = str(val).lstrip("metrics.")
                    try:
                        eval("metrics."+val)
                        method_string = method_string+key + "="+"metrics."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                else:
                    try:
                        str_val = str(eval(str(val)))
                    except:
                        str_val = ""
                    if "<built-in function" in str_val:
                        method_string = method_string+key+"='"+val+"', "
                    else:
                        method_string = method_string+key+"="+val+", "
            elif isinstance(val,str):
                try:
                    str_val = str(eval(str(val)))
                except:
                    str_val = ""
                if "<built-in function" in str_val:
                    method_string = method_string+key+"='"+val+"', "
                else:
                    try:
                        eval(method_string+key+"="+val+")")
                        method_string = method_string+key+"="+val+", "
                    except Exception as e:
                        if "\\" in val:
                            method_string = method_string+key+"=r'"+val+"', "
                        else:
                            method_string = method_string+key+"='"+val+"', "
            else:
                method_string = method_string+key+"="+str(val)+", "
            #else:
            #    try:
            #        eval(str(val))
            #        method_string = method_string+key+"="+str(val)+", "
            #    except:
            #        method_string = method_string+key+"='"+str(val)+"', "
        else:
            if "initializer" in key:
                method_string = method_string+key+"=initializers."
            elif "regularizer" in key:
                method_string = method_string+key+"=regularizers."
            elif "constraint" in key:
                method_string = method_string+key+"=constraints."
            elif "callback" in key:
                method_string = method_string+key+"=callbacks."
            elif "optimizer" in key:
                method_string = method_string+key+"=optimizers."
            method_string = method_string+build_method_string_from_dict(
                class_name=None, method_name=val["name"], args=val["args"], kwargs=val["kwargs"])+", "
    method_string = method_string.rstrip(", ")+")"
    return method_string

def check_set_dict_keys(dic, keys, vals,verbose=None):
    #keys = np.array([keys]).flatten()
    #vals = np.array([vals]).flatten()
    if len(keys) != len(vals):
        raise Exception("Keys and values should have the same dimension.")
    for i in range(len(keys)):
        try:
            dic[keys[i]]
        except:
            dic[keys[i]] = vals[i]
            print("The key '"+str(keys[i])+"' was not specified and has been set to the default value '"+str(vals[i])+"'.", show = verbose)

def check_repeated_elements_at_start(lst):
    x0 = lst[0]
    n = 0
    for x in lst[1:]:
        if x == x0:
            n += 1
        else:
            return n
    return n

def get_spaced_elements(array, numElems=5):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return out

def next_power_of_two(x):
    i = 1
    while i < x:
        i = i << 1
    return i

def closest_power_of_two(x):
    op = math.floor if bin(int(x))[3] != "1" else math.ceil
    return 2**(op(math.log(x, 2)))

def convert_types_dict(d):
    dd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            dd[k] = convert_types_dict(v)
        elif isinstance(v,np.ndarray):
            dd[k] = v.tolist()
        elif isinstance(v,list):
            if str in [type(q) for q in flatten_list(v)]:
                dd[k] = np.array(v, dtype=object).tolist()
            else:
                dd[k] = np.array(v).tolist()
        else:
            dd[k] = np.array(v).tolist()
    return dd

def sort_dict(dictionary):
    new_dict={}
    keys_sorted = sorted(list(dictionary.keys()),key= lambda v: (str(v).upper(), str(v)[0].islower()))
    for k in keys_sorted:
        if isinstance(dictionary[k], dict):
            new_dict[k] = sort_dict(dictionary[k])
        else:
            new_dict[k] = dictionary[k]
    return new_dict

def normalize_weights(w):
    return w/np.sum(w)*len(w)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def dic_minus_keys(dictionary: Dict[Any, Any],
                   keys: List[str],
                  ) -> Dict[Any, Any]:
    if isinstance(keys,str):
        shallow_copy = dict(dictionary)
        try:
            del shallow_copy[keys]
        except:
            pass
        return shallow_copy
    elif isinstance(keys,list):
        shallow_copy = dict(dictionary)
        for i in keys:
            try:
                del shallow_copy[i]
            except:
                pass
    else:
        raise ValueError("Invalid value for 'keys' argument. The argument should be either a string or a list.")
    return shallow_copy

def string_split_at_char(s, c):
    mid = len(s)//2
    try:
        break_at = mid + min(-s[mid::-1].index(c), s[mid:].index(c), key=abs)
    except ValueError:  # if '\n' not in s
        break_at = len(s)
    firstpart, secondpart = s[:break_at + 1].rstrip(), s[break_at:].lstrip(c).rstrip()
    return [firstpart, secondpart]

def string_add_newline_at_char(s, c):
    firstpart, secondpart = string_split_at_char(s, c)
    return firstpart+"\n"+"\t"+secondpart

def strip_suffix(s, suff):
    if s.endswith(suff):
        return s[:len(s)-len(suff)]
    return s

def check_add_suffix(s: str, 
                     suff: str
                    ) -> str:
    if s.endswith(suff):
        return s
    else:
        return s+suff

def minus_logprob(y_true, y_pred):
    """
    Function used as custom loss function
    """
    return -y_pred

def strip_prefix(s, pref):
    if s.startswith(pref):
        return s[len(s)-len(pref):]
    return s

def check_add_prefix(s, pref):
    if s.startswith(pref):
        return s
    else:
        return pref+s
    
def get_sorted_grid(pars_ranges, spacing="grid"):
    totpoints = int(np.product(np.array(pars_ranges)[:, -1]))
    npars = len(pars_ranges)
    if spacing == "random":
        grid = [np.random.uniform(*par) for par in pars_ranges]
    elif spacing == "grid":
        grid = [np.linspace(*par) for par in pars_ranges]
    else:
        print(header_string,"\nInvalid spacing argument. It should be one of: 'random' and 'grid'. Continuing with 'grid'.\n")
        grid = [np.linspace(*par) for par in pars_ranges]
    #np.meshgrid(*grid)
    #np.vstack(np.meshgrid(*grid)).reshape(npoints**len(pars),-1).T
    #np.meshgrid(*grid)#.reshape(125,3)
    pars_vals = np.stack(np.meshgrid(*grid), axis=npars).reshape(totpoints, -1)
    q = npars-1
    for i in range(npars):
        pars_vals = pars_vals[pars_vals[:, q].argsort(kind='mergesort')]
    q = q-1
    return pars_vals

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base == "1":
            return r"10^{{{0}}}".format(int(exponent))
        else:
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def dict_structure(dic):
    excluded_keys = []
    def dict_structure_sub(dic,excluded_keys = []):
        res = {}
        for key, value in dic.items():
            if isinstance(value, dict):
                res[key], kk = dict_structure_sub(value,excluded_keys)
                excluded_keys = np.unique(excluded_keys+kk+list(value.keys())).tolist()
                for k in excluded_keys:
                    try:
                        res.pop(k)
                    except:
                        for i in res.keys():
                            if res[i] == {}:
                                res[i] = "..."
            else:
                res[key] = type(value)
                for k in excluded_keys:
                    try:
                        res.pop(k)
                    except:
                        for i in res.keys():
                            if res[i] == {}:
                                res[i] = "..."
        return res, excluded_keys
    res = {}
    for key, value in dic.items():
        if isinstance(value, dict):
            res[key], kk = dict_structure_sub(value,excluded_keys)
            excluded_keys = np.unique(excluded_keys+kk+list(value.keys())).tolist()
            for k in excluded_keys:
                try:
                    res.pop(k)
                except:
                    for i in res.keys():
                        if res[i] == {}:
                            res[i] = "..."
        else:
            res[key] = type(value)
            for k in excluded_keys:
                try:
                    res.pop(k)
                except:
                    for i in res.keys():
                        if res[i] == {}:
                            res[i] = "..."
    return res

def inspect_object(obj: Any,
                   get_types: bool = False,
                   types_str: bool = True
                  ) -> Dict[str,Dict[str,Any]]:
    if get_types:
        if types_str:
            attrs = {k: [str(type(v)),v] for k,v in list(obj.__dict__.items())}
            meths = {x: [str(type(getattr(obj, x))),getattr(obj, x)] for x in dir(obj) if callable(getattr(obj, x))}
            props = {x: [str(type(getattr(obj, x))),getattr(obj, x)] for x in dir(obj) if not callable(getattr(obj, x))}
        else:
            attrs = {k: [type(v),v] for k,v in list(obj.__dict__.items())}
            meths = {x: [type(getattr(obj, x)),getattr(obj, x)] for x in dir(obj) if callable(getattr(obj, x))}
            props = {x: [type(getattr(obj, x)),getattr(obj, x)] for x in dir(obj) if not callable(getattr(obj, x))}
    else:
        attrs = {k: v for k,v in list(obj.__dict__.items())}
        meths = {x: getattr(obj, x) for x in dir(obj) if callable(getattr(obj, x))}
        props = {x: getattr(obj, x) for x in dir(obj) if not callable(getattr(obj, x))}
    for x in attrs.keys():
        if x in list(meths.keys()):
            meths.pop(x)
        if x in list(props.keys()):
            props.pop(x)
    private_attrs = {k: v for k,v in attrs.items() if k.startswith("_")}
    public_attrs = {k: v for k,v in attrs.items() if not k.startswith("_")}
    builtin_props = {k: v for k,v in props.items() if k.endswith("__")}
    private_props = {k: v for k,v in props.items() if k.startswith("_")}
    for x in builtin_props.keys():
        if x in list(private_props.keys()):
            private_props.pop(x)
    public_props = {k: v for k,v in props.items() if not k.startswith("_")}
    builtin_meths = {k: v for k,v in meths.items() if k.endswith("__")}
    private_meths = {k: v for k,v in meths.items() if k.startswith("_")}
    for x in builtin_meths.keys():
        if x in list(private_meths.keys()):
          private_meths.pop(x)
    public_meths = {k: v for k,v in meths.items() if not k.startswith("_")}
    result = {"private attributes": private_attrs, 
              "public attributes": public_attrs,
              "builtin properties": builtin_props,
              "private properties": private_props,
              "public properties": public_props,
              "builtin methods": builtin_meths, 
              "private methods": private_meths, 
              "public methods": public_meths}
    return result

def compare_objects(obj1,obj2,string="",only_dict=True,excluded_attrs=[],strong_exclusion=True,verbose=False):
    verbose_sub = verbose
    if verbose < 0:
        verbose_sub = 0
    diffs = []
    print("Comparing obejects", string, ".", show = verbose_sub)
    if only_dict:
        dict1=obj1.__dict__
        dict2=obj2.__dict__
    else:
        dict1 = dic_minus_keys(inspect_object(obj1, get_types=True, types_str=True),["builtin properties","builtin methods"])
        dict2 = dic_minus_keys(inspect_object(obj2, get_types=True, types_str=True),["builtin properties","builtin methods"])
    diffs = compare_dictionaries(dict1,dict2,string,only_dict=only_dict,excluded_attrs=excluded_attrs,strong_exclusion=strong_exclusion,verbose=verbose)
    return diffs
    
def compare_dictionaries(dict1,dict2,string="",only_dict=True,excluded_attrs=[],strong_exclusion=True,verbose=False):
    verbose_sub = verbose
    if verbose < 0:
        verbose_sub = 0
    diffs = []
    print("Comparing dictionaries", string, ".", show = verbose_sub)
    try:
        if dict1 == dict2:
            print("-----> OK: ", string, ": Dictionaries are equal.\n", show = verbose)
            return diffs
    except:
        pass
    dict1tmp = dic_minus_keys(dict1,excluded_attrs)
    dict2tmp = dic_minus_keys(dict2,excluded_attrs)
    dict1_removed = list(set(dict1.keys())-set(dict1tmp.keys()))
    dict2_removed = list(set(dict2.keys())-set(dict2tmp.keys()))
    if dict1_removed != [] or dict2_removed != []:
        for k in dict1_removed:
            string_print = string + " - " + str(k)
            #if "-" in string:
            #    string_print = string + " - " + str(k)
            #else:
            #    string_print = " - "+str(k)
            if k in dict2_removed:
                print("!!!!!> EXCLUDED: ",string_print,": Values are",dict1[k],"and",dict2[k],".\n", show = verbose)
            else:
                print("!!!!!> EXCLUDED: ",string_print,"(only present in firse dictionary)",": Value is",dict1[k],".\n", show = verbose)
        for k in dict2_removed:
            string_print = string + " - " + str(k)
            #if "-" in string:
            #    string_print = string
            #else:
            #    string_print = str(k)
            if k not in dict1_removed:
                print("!!!!!> EXCLUDED: ",string_print,"(only present in second dictionary)",": Value is",dict2[k],".\n", show = verbose)
    dict1 = dict1tmp
    dict2 = dict2tmp
    def intersection(lst1, lst2): 
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    try:
        keys1 = sorted(keys1)
        keys2 = sorted(keys2)
    except:
        pass
    diff1 = list(set(keys1) - set(keys2))
    diff2 = list(set(keys2) - set(keys2))
    keys = intersection(keys1, keys2)
    if diff1 != []:
        print("=====> DIFFERENCE: ",string,": Keys",diff1,"are in dict1 but not in dict2.\n", show = verbose)
        diffs.append([string,keys1,keys2])
    if diff2 != []:
        print("=====> DIFFERENCE: ",string,": Keys",diff2,"are in dict2 but not in dict1.\n", show = verbose)
        diffs.append([string,keys1,keys2])
    #if diff1 == [] and diff2 == []:
    #    print(tabstr,"OK: Keys in the two dictionaries are equal.")
    for k in keys:
        prestring = string + " - " + str(k)
        print("Comparing keys", prestring, ".", show = verbose_sub)
        #print(tabstr,"Checking key",k,".")
        areobjects=False
        try:
            dic_minus_keys(dict1[k].__dict__,excluded_attrs)
            dic_minus_keys(dict2[k].__dict__,excluded_attrs)
            areobjects=True
        except:
            pass
        if areobjects:
            print("Keys", prestring, "are objects.", show = verbose_sub)
            diffs=diffs + compare_objects(dict1[k],dict2[k],prestring,only_dict=only_dict,excluded_attrs=excluded_attrs,verbose=verbose)
        elif isinstance(dict1[k],dict) and isinstance(dict2[k],dict):
            print("Keys", prestring, "are dictionaries.", show = verbose_sub)
            diffs=diffs + compare_dictionaries(dict1[k],dict2[k],prestring,only_dict=only_dict,excluded_attrs=excluded_attrs,verbose=verbose)
        elif isinstance(dict1[k],(np.ndarray,list,tuple)) and isinstance(dict2[k],(np.ndarray,list,tuple)):
            print("Keys", prestring, "are lists, numpy arrays, or tuple.", show = verbose_sub)
            list1 = dict1[k]
            list2 = dict2[k]
            try:
                if strong_exclusion:
                    list1type = []
                    list2type = []
                    for i in excluded_attrs:
                        list1type = [x for x  in list1 if type(x)==type(i)]
                        list2type = [x for x  in list2 if type(x)==type(i)]
                    if len([x for x in excluded_attrs if x in list1type]) != 0 or len([x for x in excluded_attrs if x in list2type]) != 0:
                        print("!!!!!> EXCLUDED: ",prestring,": Values are",list1,"and",list2,".\n", show = verbose)
                        #print("Entry removed",show=verbose)
                        list1 = []
                        list2 = []
            except:
                print("Failed on", prestring,":\nvalue1 =",list1,"\nvalue2 = ",list2,".")
            diffs=diffs +compare_lists_arrays_tuple(list1,list2,prestring,only_dict=only_dict,excluded_attrs=excluded_attrs,strong_exclusion=strong_exclusion,verbose=verbose)
        else:
            try:
                if not dict1[k] == dict2[k]:
                    print("=====> DIFFERENCE: ",prestring,": Values are",dict1[k],"and",dict2[k],".\n", show = verbose)
                    diffs.append([prestring,dict1[k],dict2[k]])
                else:
                    print("-----> OK: ",prestring,": Values are equal.\n", show = verbose)
            except:
                print("xxxxx> FAILED: ",prestring,": Values could not be compared. Values are",dict1[k],"and",dict2[k],".\n", show = verbose)
                diffs.append([prestring+" - FAILED TO COMPARE",dict1[k],dict2[k]])
    if diffs == []:
        print("-----> OK: ",string,": Dictionaries are equal.\n", show = verbose)
    return diffs

def compare_lists_arrays_tuple(list1,list2,string="",only_dict=True,excluded_attrs=[],strong_exclusion=True,verbose=False):
    verbose_sub = verbose
    if verbose < 0:
        verbose_sub = 0
    diffs = []
    print("Comparing list or arrays", string, ".", show = verbose_sub)
    try:
        if list1 == list2:
            print("-----> OK: ", string, ": Lists are equal.\n", show = verbose)
            return diffs
    except:
        pass
    try:
        arr1 = np.array(list1, dtype=object)
        arr2 = np.array(list2, dtype=object)
        if np.all(np.equal(arr1, arr2)):
            print("-----> OK: ", string, ": Lists are equal.\n", show = verbose)
            return diffs
    except:
        pass
    if strong_exclusion:
        excluded = False
        new_list1 = []
        new_list2 = []
        for e in list1:
            #print(e)
            if isinstance(e,(np.ndarray,list,tuple)):
                list1type = []
                for i in excluded_attrs:
                    list1type = [x for x  in e if type(x)==type(i)]
                if len([x for x in excluded_attrs if x in list1type]) == 0:
                    new_list1.append(e)
                else:
                    new_list1.append([])
                    print("!!!!!> EXCLUDED: ",string,": Values are",list1,"and",list2,".\n", show = verbose)
                    excluded = True
            else:
                new_list1.append(e)
        for e in list2:
            #print(e)
            if isinstance(e,(np.ndarray,list,tuple)):
                list2type = []
                for i in excluded_attrs:
                    list2type = [x for x  in e if type(x)==type(i)]
                if len([x for x in excluded_attrs if x in list2type]) == 0:
                    new_list2.append(e)
                else:
                    new_list2.append([])
                    if not excluded:
                        print("!!!!!> EXCLUDED: ",string,": Values are",list1,"and",list2,".\n", show = verbose)
                    excluded = False
            else:
                new_list2.append(e)
        list1 = new_list1
        list2 = new_list2
    try:
        if list1 == list2:
            print("-----> OK: ", string, ": Lists are equal.\n", show = verbose)
            return diffs
    except:
        pass
    try:
        arr1 = np.array(list1, dtype=object)
        arr2 = np.array(list2, dtype=object)
        if np.all(np.equal(arr1, arr2)):
            print("-----> OK: ", string, ": Lists are equal.\n", show = verbose)
            return diffs
    except:
        pass
    if len(list1)!=len(list2):
        print("=====> DIFFERENCE: ",string,": Lists have different length.\n", show = verbose)
        diffs.append([string, list1, list2])
    else:
        for i in range(len(list1)):
            prestring = string + " - list entry " + str(i)
            print("Comparing", prestring, ".", show = verbose_sub)
            areobjects=False
            try:
                dic_minus_keys(list1[i].__dict__,excluded_attrs)
                dic_minus_keys(list2[i].__dict__,excluded_attrs)
                areobjects=True
            except:
                pass
            if areobjects:
                print("Items", prestring, "are objects.", show = verbose_sub)
                diffs = diffs + compare_objects(list1[i],list2[i],prestring,only_dict=only_dict,excluded_attrs=excluded_attrs,strong_exclusion=strong_exclusion,verbose=verbose)
            elif isinstance(list1[i],dict) and isinstance(list2[i],dict):
                print("Items", prestring, "are dictionaries.", show = verbose_sub)
                diffs = diffs + compare_dictionaries(list1[i],list2[i],prestring,only_dict=only_dict,excluded_attrs=excluded_attrs,strong_exclusion=strong_exclusion,verbose=verbose)
            elif isinstance(list1[i],(np.ndarray,list,tuple)) and isinstance(list2[i],(np.ndarray,list,tuple)):
                print("Items", prestring,
                      "are lists, numpy arrays, or tuple.", show = verbose_sub)
                diffs = diffs + compare_lists_arrays_tuple(list1[i],list2[i],prestring,only_dict=only_dict,excluded_attrs=excluded_attrs,strong_exclusion=strong_exclusion,verbose=verbose)
            else:
                try:
                    if not list1[i] == list2[i]:
                        print("=====> DIFFERENCE: ",prestring,": Values are",list1[i],"and",list2[i],".\n", show = verbose)
                        diffs.append([prestring,list1[i],list2[i]])
                    else:
                        print("-----> OK: ", prestring, " Items are equal.\n", show = verbose)
                except:
                    print("xxxxx> FAILED: ", prestring, ": Values could not be compared. Values are",
                          list1[i], "and", list2[i], ".\n", show = verbose)
                    diffs.append([prestring,list1[i],list2[i]])
    return diffs

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def ResultsToDict(results_dict,run_n,run_seed,ndims,nsamples,corr,bijector_name,nbijectors,activation,spline_knots,range_min,kl_divergence,ks_mean,ks_median,ad_mean,ad_median,w_distance_median,w_distance_mean,frob_norm,hidden_layers,batch_size,eps_regulariser,regulariser,epochs_input,epochs_output,training_time):
    """
    Function that writes results to the a dictionary.
    """
    results_dict.get('run_n').append(run_n)
    results_dict.get('run_seed').append(run_seed)
    results_dict.get('ndims').append(ndims)
    results_dict.get('nsamples').append(nsamples)
    results_dict.get('correlation').append(corr)
    results_dict.get('bijector').append(bijector_name)
    results_dict.get('nbijectors').append(nbijectors)
    results_dict.get('activation').append(activation)
    results_dict.get('spline_knots').append(spline_knots)
    results_dict.get('range_min').append(range_min)
    results_dict.get('kl_divergence').append(kl_divergence)
    results_dict.get('ks_test_mean').append(ks_mean)
    results_dict.get('ks_test_median').append(ks_median)
    results_dict.get('ad_test_mean').append(ad_mean)
    results_dict.get('ad_test_median').append(ad_median)
    results_dict.get('Wasserstein_median').append(w_distance_median)
    results_dict.get('Wasserstein_mean').append(w_distance_mean)
    results_dict.get('frob_norm').append(frob_norm)
    results_dict.get('epochs_input').append(epochs_input)
    results_dict.get('epochs_output').append(epochs_output)
    results_dict.get('time').append(training_time)
    results_dict.get('hidden_layers').append(hidden_layers)
    results_dict.get('batch_size').append(batch_size)
    results_dict.get('eps_regulariser').append(eps_regulariser)
    results_dict.get('regulariser').append(regulariser)
    return results_dict

def logger(log_file_name,results_dict):
    """
    Logger that writes results of each run to a common log file.
    """
    log_file=open(log_file_name,'a')
    string_list=[]
    for key in results_dict.keys():
        string_list.append(str(results_dict.get(key)[-1]))
    string=','.join(string_list)
    log_file.write(string)
    log_file.write('\n')
    log_file.close()
    return

#def logger_nan(run_number):
#    """
#    Logger that takes care of nan runs.
#    """
#    log_file=open(log_file_name,'a')
#    log_file.write(str(run_number)+",")
#    log_file.write('\n')
#    log_file.close()
#    return

def results_current(path_to_results,results_dict):
    """
    Function that writes results of the current run to the results.txt file.
    """
    currrent_results_file=open(path_to_results+'results.txt','w')
    header=','.join(list(results_dict.keys()))
    currrent_results_file.write(header)
    currrent_results_file.write('\n')
    string_list=[]
    for key in results_dict.keys():
        string_list.append(str(results_dict.get(key)[-1]))
    string=','.join(string_list)
    currrent_results_file.write(string)
    currrent_results_file.write('\n')
    currrent_results_file.close()
    return

def save_hyperparams(path_to_results,hyperparams_dict,run_n,run_seed,ndims,nsamples,corr,bijector_name,nbijectors,spline_knots,range_min,hllabel,batch_size,activation,eps_regulariser,regulariser,dist_seed,test_seed):
    """
    Function that writes hyperparameters values to a dictionary and saves it to the hyperparam.txt file.
    """
    hyperparams_dict.get('run_n').append(run_n)
    hyperparams_dict.get('run_seed').append(run_seed)
    hyperparams_dict.get('ndims').append(ndims)
    hyperparams_dict.get('nsamples').append(nsamples)
    hyperparams_dict.get('correlation').append(corr)
    hyperparams_dict.get('bijector').append(bijector_name)
    hyperparams_dict.get('nbijectors').append(nbijectors)
    hyperparams_dict.get('spline_knots').append(spline_knots)
    hyperparams_dict.get('range_min').append(range_min)
    hyperparams_dict.get('hidden_layers').append(hllabel)
    hyperparams_dict.get('batch_size').append(batch_size)
    hyperparams_dict.get('activation').append(activation)
    hyperparams_dict.get('eps_regulariser').append(eps_regulariser)
    hyperparams_dict.get('regulariser').append(regulariser)
    hyperparams_dict.get('dist_seed').append(dist_seed)
    hyperparams_dict.get('test_seed').append(test_seed)
    hyperparams_frame=pd.DataFrame(hyperparams_dict)
    hyperparams_frame.to_csv(path_to_results+'hyperparams.txt',index=False)
    return hyperparams_dict

def load_model(nf_dist,path_to_results,ndims,lr=.00001):
    """
    Function that loads a model by recreating it, recompiling it and loading checkpointed weights.
    """
    x_ = Input(shape=(ndims,), dtype=tf.float32)
    log_prob_ = nf_dist.log_prob(x_)
    model = Model(x_, log_prob_)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=lambda _, log_prob: -log_prob)
    model.load_weights(path_to_results+'model_checkpoint/weights')
    return nf_dist,model

#def saver(nf_dist,path_to_weights,iter):
#    """
#    Function that saves the model.
#    """
#    for j in range(len(list(nf_dist.bijector.bijectors))):
#        weights_dir=path_to_weights+'iter_'+str(iter)
#        try:
#            os.mkdir(weights_dir)
#        except:
#            print(weights_dir+' file exists')
#        name=nf_dist.bijector.bijectors[j].name
#        if name=='MAFspline':
#            weights=nf_dist.bijector.bijectors[j].parameters.get('shift_and_log_scale_fn').get_weights()            
#            weights_file = open(weights_dir+'/'+name+'_'+str(j)+'.pkl', "wb")
#            pickle.dump(weights,weights_file)
#        else:
#            continue
#    return

def save_bijector_info(bijector,path_to_results):
    """
    Function that saves the bijecor.
    """
    bij_out_file=open(path_to_results+'bijector_chain.txt','w')
    for bij in list(bijector.bijectors):
        bij_out_file.write(bij.name)
        bij_out_file.write('\n')
    bij_out_file.close()
    return

@tf.function
def nf_sample_iter(nf_dist,iter_size,n_iters,seed=0):
    """
    To be decumented.
    """
    reset_random_seeds(seed)
    #first iter
    sample_all=nf_dist.sample(iter_size,seed=seed)
    for j in range(1,n_iters):
        #if j%100==0:
            #print(tf.shape(sample_all))
        sample=nf_dist.sample(iter_size,seed=seed)
        #sample=postprocess_data(sample,preprocess_params)
        sample_all=tf.concat([sample_all,sample],0)
        #if j%1==0:
        #    with open(path_to_results+'nf_sample_5_'+str(j)+'.npy', 'wb') as f:
        #        np.save(f, sample, allow_pickle=True)
        #tf.keras.backend.clear_session()
    return sample_all

#def nf_sample_save(nf_dist,path_to_results,sample_size=100000,iter_size=10000,rot=None,seed=0):
#    """
#    Function that saves the samples.
#    """
#    print('saving samples...')
#    n_iters=int(sample_size/iter_size)
#    sample_all=nf_sample_iter(nf_dist,iter_size,n_iters,seed=seed)
#    sample_all=sample_all.numpy() # type: ignore
#    if rot is not None:
#        sample_all = Distributions.inverse_transform_data(sample_all,rot)
#    with open(path_to_results+'nf_sample.npy', 'wb') as f:
#        np.save(f, sample_all, allow_pickle=True)
#    print('samples saved')
#    return sample_all

def flatten_list(lst):
    out = []
    for item in lst:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(flatten_list(item))
        else:
            out.append(item)
    return out

def save_details_json(hyperparams_dict,results_dict,train_loss_history,val_loss_history,path_to_results):
    """ Save results and hyperparameters json
    """
    if val_loss_history is None:
        val_loss_history = []
    if train_loss_history is None:
        train_loss_history = []
    train_loss_history = np.array(train_loss_history)
    val_loss_history = np.array(val_loss_history)
    if val_loss_history.tolist() != []:
        best_val_loss = np.min(val_loss_history)
        try:
            position_best_val_loss = np.where(val_loss_history == best_val_loss)[0][0]
        except:
            try:
                position_best_val_loss = np.where(val_loss_history == best_val_loss)[0]
            except:
                position_best_val_loss = None
        if position_best_val_loss is not None:
            best_train_loss = train_loss_history[position_best_val_loss]
        else:
            best_train_loss = None
    else:
        best_val_loss = None
        position_best_val_loss = None
        best_train_loss = None
    hd={}
    rd={}
    for k in hyperparams_dict.keys():
        hd[k] = hyperparams_dict[k][-1]
    for k in results_dict.keys():
        rd[k] = results_dict[k][-1]
    details_dict = {**hd,**rd,
                    "train_loss_history": train_loss_history.tolist(),
                    "val_loss_history": val_loss_history.tolist(),
                    "best_train_loss": best_train_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": position_best_val_loss}
    dictionary = convert_types_dict(details_dict)
    with codecs.open(path_to_results+'details.json', "w", encoding="utf-8") as f:
        json.dump(dictionary, f, separators=(",", ":"), indent=4)

def create_log_file(mother_output_dir,results_dict):
    log_file_name=mother_output_dir+'log_file_eternal.txt'
    if os.path.isfile(log_file_name)==False:
        log_file=open(log_file_name,'w')
        header=','.join(list(results_dict.keys()))
        print(header)
        log_file.write(header)
        log_file.write('\n')
        log_file.close()
    return log_file_name

def RandCorr(self,
             ndims: int,
             seed: Optional[int] = None,
            ) -> npt.NDArray:
    if seed is None:
        seed = self._seed
    np.random.seed(seed)
    V = datasets.make_spd_matrix(ndims,random_state=seed)
    D = np.sqrt(np.diag(np.diag(V)))
    Dinv = np.linalg.inv(D)
    Vnorm = np.matmul(np.matmul(Dinv,V),Dinv)
    return Vnorm
    
def RandCov(self,
            std: Array,
            seed: Optional[int] = None,
           ) -> npt.NDArray:
    if seed is None:
        seed = self._seed
    np.random.seed(seed)
    std = np.array(std)
    ndims = len(std)
    corr = self.RandCorr(ndims,seed)
    D = np.diag(std)
    V = np.matmul(np.matmul(D,corr),D)
    return V