import random
import math
import matplotlib.pyplot as plt

#创建 Task 任务数据类型  
class Task:
    def __init__(self):
        #任务初始化
        # important 重要性 (0,1]
        self.important = None
        # urgency 紧要性 (0,1]
        self.urgency = None
        # time 需要时间
        self.time = None
        # correlation 和其他任务的关联程度，初始状态应该[0]*task_length，自己id的应该是1，无关的是0
        self.correlation = []
        # interest 有趣性 (0,1]
        self.interest = None
        # difficulty 困难性 (0,1]
        self.difficulty = None
        # q(task_i) 计算系数
        self.qtask = None
        # 1 - correlation[i]
        self.corroppo = []
    def Calculate_sortvalue(self):

        return 
    def print_variables(self):
        print(f"{self.important},{self.urgency},{self.time},{self.correlation},{self.interest},{self.difficulty},{self.qtask},{self.corroppo}")
    def Cal_qtask(self):
        self.qtask = (self.important)*(self.urgency**2)*(self.interest**0.5)*(1-self.difficulty**3)/self.time
    def Cal_1op(self):
        self.corroppo = [round(1-task_link,2) for task_link in self.correlation]


#随机生成一个task任务
def random_taskone(id):
    #需要输入一个task_length,用于下面correlation的随机生成
    task = Task()
    task.important = round(1 - random.random(),2)
    task.urgency = round(1 - random.random(),2)
    task.time = random.randint(1,max_taskedTime)#max_taskedTime 需要进一步确认
    task.correlation = [round(random.random(),2) for i in range(task_length)]
    task.correlation[id] = 1
    task.interest = round(1 - random.random(),2)
    task.difficulty = round(1 - random.random(),2)
    return task

#随机生成一个taskarray数组
def random_taskarray():
    taskarray = [random_taskone(i) for i in range(task_length)]
    for i in range(task_length):
        taskarray[i].Cal_qtask()
        taskarray[i].Cal_1op()
    return taskarray

#打印整个taskarray数组
def print_taskarray(taskarray):
    for i in range(task_length):
        taskarray[i].print_variables()

#先计算出任务时间求和 sum
def sum_taskTime():
    sum_taskTime = sum([task_array[i].time for i in range(task_length)])
    return sum_taskTime

#求出每个任务的起始点
def cal_taskStart_array():
    taskStart_array = [0]
    for i in range(task_length-1):
        taskStart_array+=[task_array[i].time + taskStart_array[i]]
    return taskStart_array

#将shuffle(num_array)数组转化成task编号的
def shuffle_changeto(array):
    changeto_array = []
    for i in range(total_time):
        i_index = 0
        while(i_index <=task_length - 1 and taskStart_array[i_index]<=array[i]):
            i_index+=1
        changeto_array.append(i_index-1)
    return changeto_array
#对于任务时间求和

#生成种群，初始化
def init_population():
    population = []
    #先计算出任务时间求和 sum
    sum_tasktime = sum_taskTime()
    #求出每个任务的起始点 taskStart_array = []
    num_array = [i for i in range(sum_tasktime)]
    #[task1:0,1,2,task2:3,4,...,sum]
    for i in range(population_size):
        random.shuffle(num_array)
        population.append(shuffle_changeto(num_array[0:total_time]))
    #生成初始种群
    return population


#计算适应度
def fitness_cal():
    #适应度计算公式
    fitness = []
    for i in range(population_size):#每一个个体
        fitness_per = 0
        #population[i]
        #对每个时刻的适应度求和
        #finish_task 用于下面存储每个时刻已完成的任务进度
        finish_task = [0 for i in range(task_length)]
        for T in range(total_time):
            fitness_timeT = task_array[population[i][T]].qtask
            for j in range(task_length):
                if j == population[i]:
                    continue
                fitness_timeT*=task_array[population[i][T]].corroppo[j] + task_array[population[i][T]].correlation[j]*finish_task[j]/timelength_array[j]
            fitness_per+= fitness_timeT
            finish_task[population[i][T]]+=1
        fitness.append(fitness_per*math.pow(10,task_length//2))
    return fitness
#存在一个问题数值太小 当task取到50左右

#选择优良父辈
def select_betterparents():
    fitness_deal = [(fitness[i],i) for i in range(population_size)]
    sorted_population = [num[1] for num in sorted(fitness_deal,key= lambda compared_one:compared_one[0],reverse= True)]
    return sorted_population[:selected_size]#返回前selected_size的父辈

#获取一个各个任务的时间总长的函数
def Timelength_array():
    timelength_array = [task_array[i].time for i in range(task_length)]
    return timelength_array
#交叉繁衍
def crossover(parent1,parent2):
    #该情境下不能直接交换，并且顺序交换后可能会使任务学习时间超过任务需要时间
    #和timelength_array相比
    startGene = random.randint(0,total_time-1)
    endGene = random.randint(0,total_time-1)

    while(startGene == endGene):
        endGene = random.randint(0,total_time-1)
    #由于parent1,parent2等价所以对两者进行不同处理其实可能相同
    #不妨parent1取两边，parent2取中间
    child = parent1[0:startGene]+parent2[startGene:endGene]+parent1[endGene:total_time]
    thistimelength_array = [0 for i in range(task_length)]
    for item in child:
        thistimelength_array[item]+=1
    flag = 1
    check_array = [thistimelength_array[i]>timelength_array[i] for i in range(task_length)]
    if(sum(check_array)>0):
        return None
    else:
        return child
    #如果超出说明两者不能交配，那么重新选择

def mutate(ones):
    for swapone in range(total_time):
        if random.random() < mutate_rate:
            swapanother = random.randint(0,total_time-1)
            ones[swapone],ones[swapanother] = ones[swapanother],ones[swapone]
    return ones

#进化
def evolve_population(t):
    sorted_population = select_betterparents()#是一个整数数组

    data_generation.append(fitness[sorted_population[0]])

    unchanged_parents = [population[i] for i in sorted_population] #是上一代种群的优种
    children = []
    #
    for _ in range(population_size - selected_size):
        parent1 = random.choice(sorted_population)
        parent2 = random.choice(sorted_population)
        while(parent2 == parent1):
            parent2 = random.choice(sorted_population)
        
        child = crossover(population[sorted_population[0]],population[sorted_population[1]])
        while(child == None):
            child = crossover(population[sorted_population[0]],population[sorted_population[1]])
        children.append(mutate(child))
    return unchanged_parents + children

#迭代图
def plot_iterations():
    plt.figure(figsize=(15,15))
    for i in range(generations-1):
        plt.plot([i+1,i+2],[data_generation[i],data_generation[i+1]],color = 'gray')
    plt.title('iterations')
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.show()

#输入变量 task_length 任务数量 max_taskedTime 单个任务最大时间 
task_length = 20
max_taskedTime = 24
total_time = 24

population_size = 100
selected_size = 20
generations = 500
mutate_rate = 0.01
data_generation = []#存储每一代的最佳适宜度
# task_array 是初始的任务
'''

'''
#随机生成任务
task_array = random_taskarray()
#计算任务起始点
taskStart_array = cal_taskStart_array()
#种群初始化
population = init_population()

#获取任务总长函数
timelength_array = Timelength_array()

fitness = fitness_cal()


print_taskarray(task_array)
print(population)
print(fitness)
# 初始化种群


for i in range(generations):
    population = evolve_population(i)
    fitness = fitness_cal()
best_route = population[select_betterparents()[0]]
print(best_route)

plot_iterations()
