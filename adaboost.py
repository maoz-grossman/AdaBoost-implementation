#!/usr/bin/env python
# coding: utf-8



from itertools import combinations
from itertools import permutations
from numpy import log as ln
from numpy import exp as e
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time 


#~~~~~~~~~~~~~~
#   classes
#~~~~~~~~~~~~~~
class point:
    def __init__(self , x=0,gender=1,y=0):
        self.x=float(x)
        self.y= float(y)
        self.gender=int(gender)
        self.weight=0
    
class  rec_hyp:
    def __init__(self, point1, point2):
        self.min_x= min(point1.x,point2.x)
        self.max_x=max(point1.x,point2.x)
        self.min_y= min(point1.y,point2.y)
        self.max_y=max(point1.y,point2.y)
        #self.point1=point1
        #self.point2=point2
        self.gender_classifier=1
    def include(self,point_other,gender):
        if self.min_x<=point_other.x<=self.max_x and self.min_y<=point_other.y<=self.max_y:
            return gender
        else: return -gender
    
class  circ_hyp:
    def __init__(self, point1, point2):
        self.center= point1
        self.radius= ((self.center.x-point2.x)**2 +(self.center.y-point2.y)**2)**0.5
        self.gender_classifier=1
    def include(self,point_other,gender):
        if ((self.center.x-point_other.x)**2 +(self.center.y-point_other.y)**2)**0.5 <= self.radius:
            return gender
        else: return -gender
   

#~~~~~~~~~~~~~~~~
#    Methods
#~~~~~~~~~~~~~~~~

def Rectangle(points):
    comb=combinations(points, 2)
    err_sum=1 # min sum of errors
    max_rect=rec_hyp(point(),point())#the hypothesis 
    for tupx in comb:
        rect= rec_hyp(tupx[0],tupx[1])
        temp_sum=0 #What inside the hypothesis is 1
        neg_temp_sum=0 #What inside the hypothesis is -1
        for x in points:
            temp_sum+=x.weight*int(x.gender!=rect.include(x,1))
            neg_temp_sum+=x.weight*int(x.gender!=rect.include(x,-1))
        if neg_temp_sum<temp_sum:
            temp_sum=neg_temp_sum
            rect.gender_classifier=-1
        if temp_sum< err_sum:
            err_sum=temp_sum
            max_rect= rect
    #print(err_sum)
    return(max_rect,err_sum)


def  Circle(points):
    comb=permutations(points, 2)
    err_sum=1 # min sum of errors
    max_circ=circ_hyp(point(),point())#the hypothesis 
    for tupx in comb:
        circ= circ_hyp(tupx[0],tupx[1])
        #sum of error:
        temp_sum=0
        neg_temp_sum=0
        for x in points:
            temp_sum+=x.weight*int(x.gender!=circ.include(x,1))
            neg_temp_sum+=x.weight*int(x.gender!=circ.include(x,-1))
        if neg_temp_sum<temp_sum:
            temp_sum=neg_temp_sum
            circ.gender_classifier=-1
        if temp_sum< err_sum:
            err_sum=temp_sum
            max_circ= circ
    return(max_circ,err_sum)
        




def adaboost(points, hyp , r):
    set_of_hyp=[]
    for x in points:
        x.weight= 1/len(points)
    for t in range(r):
        h_t,eps_t=hyp(points)
        if eps_t>0.5:
            print("eps too big:",eps_t)
            break
        alpha_t=0.5*ln((1-eps_t)/eps_t)
        Z_t=0
        for x in points:
            #Set the weight of the points
            x.weight= x.weight*e(-alpha_t*h_t.include(x,h_t.gender_classifier)*x.gender)
            Z_t+=x.weight
        for x in points:
            # normalization
            x.weight=x.weight/Z_t 
        set_of_hyp.append((h_t,alpha_t))
    return set_of_hyp
       
    



def get_data(file_name):
    f= open(file_name, "r")
    data_set=[]
    for line in f:
        txt = line
        str_points= txt.split()
        #print(str_points)
        if str_points[1] != str(1):
            x= point(str_points[0],-1,str_points[2])
        else:
            x= point(str_points[0],str_points[1],str_points[2])
        data_set.append(x)  
    return data_set



def run_adaboost(adaboost_rounds,rounds ,points_set, hypothesis):
    tic = time.perf_counter()
    print("~~~~~~",hypothesis.__name__,"~~~~~")
    s=hypothesis.__name__
    s+="\nNumber of iterations:"+str(rounds)+"\nNumber of iterations adaboost:"+str(adaboost_rounds-1)
    for r in range(1,adaboost_rounds):
        sum_total=0
        R_total=0
        print("~~~~~~~~~")
        for i in range(rounds):
            random.shuffle(points_set)
            R=points_set[0:65]# Sample
            T=points_set[66:]#Test points
            res= adaboost(R,hypothesis,r)
            for x in T:
                H_x=0
                for hyp, alpha in res:
                    H_x+=alpha*hyp.include(x,hyp.gender_classifier)
                sum_total+=int(H_x*x.gender<0)
            for x in R:
                H_x=0
                for hyp, alpha in res:
                    H_x+=alpha*hyp.include(x,hyp.gender_classifier)
                R_total+=int(H_x*x.gender<0)
        T_errors=(sum_total/rounds)/65
        R_errors=(R_total/rounds)/65
        T_error_string="\nThe average precentage error on T in round " +str(r)+": "+ "%.3f" % T_errors+"(" +"%.3f"%(1-T_errors)+" % were correct)"
        R_error_string="\nThe average precentage error on R in round " +str(r)+": "+ "%.3f" % R_errors+"(" +"%.3f"%(1-R_errors)+" % were correct)"
        s+=R_error_string
        s+=T_error_string
        print(R_error_string)
        print(T_error_string)
        s+="\n~~~~~~~~~~~~"
    toc=time.perf_counter()
    print("\nthe program took", "%.4f"%(toc - tic), "seconds")
    return s


#~~~~~~~~~~~~~~~~~~~~~~~~~
#-----Image methods-------
#~~~~~~~~~~~~~~~~~~~~~~~~~
#circle_hypothesis_img and Rectangle_hypothesis_img
#are not part of the algorithm
#it's just for saving the result as an image

def circle_hypothesis_img(points_set,r):
    print("~~~~~Circle image~~~~~")
    random.shuffle(points_set)
    R=points_set[0:65]# Sample
    T=points_set[66:]#Test points
    res= adaboost(R, Circle,r)
    x_i=[]
    y_i=[]
    fy=[]
    fx=[]
    for x in T:
        if x.gender==1:
            x_i.append(x.x)
            y_i.append(x.y)
        else:
            fx.append(x.x)
            fy.append(x.y)
    fig, ax = plt.subplots()
    color='green'
    for hyp,err in res:
        if hyp.gender_classifier==1:
            color='green'
        else:
            color='red'
        circle = plt.Circle((hyp.center.x, hyp.center.y), hyp.radius, color=color,fill=False)
        ax.add_patch(circle)
    plt.plot(x_i, y_i,'p')
    plt.plot(fx,fy,'p', color='pink')
    pink_patch = mpatches.Patch(color='pink', label='Female points')
    blue_patch = mpatches.Patch(color='blue', label='Male points')
    green_patch= mpatches.Patch(color='green', label='+1 hypothesis')
    red_patch= mpatches.Patch(color='red', label='-1 hypothesis')
    plt.legend(handles=[blue_patch, pink_patch,green_patch, red_patch],bbox_to_anchor=(0.3, -0.1),ncol=2)
    plt.xlabel('Temperature')
    plt.ylabel('Plus')
    plt.savefig('Adaboost_Circles.png',bbox_inches= 'tight')    
    

def rectangle_hypothesis_img(points_set,r):
    print("~~~~~Rectangle image~~~~~")
    random.shuffle(points_set)
    R=points_set[0:65]# Sample
    T=points_set[66:]#Test points
    res= adaboost(R, Rectangle,r)
    x_i=[]
    y_i=[]
    fy=[]
    fx=[]
    for x in T:
        if x.gender==1:
            x_i.append(x.x)
            y_i.append(x.y)
        else:
            fx.append(x.x)
            fy.append(x.y)
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    for hyp,err in res:
        if hyp.gender_classifier==1:
            color='green'
        else: color='red'
        rect = matplotlib.patches.Rectangle((hyp.min_x, hyp.min_y), height=(hyp.max_y-hyp.min_y),width=(hyp.max_x-hyp.min_x), color=color,fill=False)
        ax.add_patch(rect)
    plt.plot(x_i, y_i,'p')
    plt.plot(fx,fy,'p', color='pink')
    pink_patch = mpatches.Patch(color='pink', label='Female points')
    blue_patch = mpatches.Patch(color='blue', label='Male points')
    green_patch= mpatches.Patch(color='green', label='+1 hypothesis')
    red_patch= mpatches.Patch(color='red', label='-1 hypothesis')
    plt.legend(handles=[blue_patch, pink_patch,green_patch, red_patch],bbox_to_anchor=(0.3, -0.1),ncol=2)
    plt.xlabel('Temperature')
    plt.ylabel('Pluse')
    plt.savefig('Adaboost_Rectangles.png',bbox_inches= 'tight')
    

#~~~~~~~~~~~~~~~~~~~~~~
#--------main----------
#~~~~~~~~~~~~~~~~~~~~~~
data_set=get_data("HC_Body_Temperature")

#Save a png of the hypothesis Rectangle results
rectangle_hypothesis_img(data_set,9)

#Save a png of the hypothesis Circle results
circle_hypothesis_img(data_set,9)

#adaboost_rounds=r-> number of iteration for the adaboost function
#rounds->number of iterations of the algorithem(we were asked for 100)
#returns string to save in a file
f=open("Rectangle output.txt","w+")
s=run_adaboost(adaboost_rounds=9 ,rounds=10 ,  points_set=data_set,hypothesis=Rectangle)
f.write(s)
f.close()

print("\n\n")


w=open("Cirle output.txt","w+")
s=run_adaboost( adaboost_rounds=9 ,rounds=10, points_set=data_set, hypothesis=Circle)
w.write(s)
w.close()



