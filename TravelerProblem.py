import random
import math
import matplotlib.pyplot as plt
#import panda as pd

#生成地图,total 地图 是从(0,length)两边取到
def generate_map(length,total):
    flag = [[0 for j in range(total+1)] for i in range(total+1)]
    map = []
    while(length):
        coordinate_x = random.randint(0,100)
        coordinate_y = random.randint(0,100)
        if(flag[coordinate_x][coordinate_y] == 0):
            map.append((coordinate_x,coordinate_y))
            flag[coordinate_x][coordinate_y] = 1
            length-=1
    return map

#计算地图各城市的距离
def distance_cal(map):
    distance_twopoints = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            distance_twopoints[i][j] = math.sqrt((map[i][0]-map[j][0])**2+(map[i][1]-map[j][1])**2)
    return distance_twopoints

#初始化种群
def generate_initial_population():
    population = []
    for _ in range(population_size):
        population.append(random.sample(range(length),length))
    return population

#计算总距离
def sum_distance():
    sum_distance = []
    for i in range(population_size):#规模 = population_size
        sum = 0
        for j in range(length):
            sum+=distance[population[i][j]][population[i][(j+1)%length]]
        sum_distance.append(sum)
    return sum_distance

#适应度计算
def fitness_cal():
    sum = sum_distance()
    fitness = [total/dis for dis in sum]
    return fitness

#选择优良父辈
def select_betterparents():
    fitness_deal = [(fitness[i],i) for i in range(population_size)]
    sorted_population = [num[1] for num in sorted(fitness_deal,key= lambda compared_one:compared_one[0],reverse= True)]
    return sorted_population[:selected_size]#返回前selected_size的父辈

#交叉繁衍
def crossover(parent1,parent2):
    #旅行商问题的交叉不能直接选取节点交换，因为每个城市都需要遍历，并且只能遍历一次
    startGene = random.randint(0,length-1)
    endGene = random.randint(0,length-1)

    while(startGene == endGene):
        endGene = random.randint(0,length-1)
    
    if(startGene>endGene):
        startGene,endGene = endGene,startGene
    
    #print(startGene,endGene) #便于调试
    #print(parent1)
    #print(parent2)
    childpart1 = [parent1[i] for i in range(startGene,endGene+1)]
    childpart2 = [item for item in parent2 if item not in childpart1]
    child = childpart1 + childpart2
    #print(child)
    return child

#变异
def mutate(ones):
    for swapone in range(length):
        if random.random() < mutate_rate:
            swapanother = random.randint(0,length-1)
            ones[swapone],ones[swapanother] = ones[swapanother],ones[swapone]
    return ones

#进化,每一代要经历的
def evolve_population(t):
    
    sorted_population = select_betterparents()#是一个整数数组
    
    if(t == 0 or t == generations - 1):
        way = [(map[i][0],map[i][1]) for i in population[sorted_population[0]]]
        print(way)
        print("sum distance")
        print(sum_distance()[sorted_population[0]])
        
        
        if(t == 0):
            color = 'red'
            label = 'init_route'
            linestyle = '--'
        else:
            color = 'blue'
            label = 'best_route'
            linestyle = '-.'
        for j in range(length):
            if(j==0):
                #暂时不知道怎么修改
                plt.plot([way[j][0],way[(j+1)%length][0]],[way[j][1],way[(j+1)%length][1]],color = color,linestyle = linestyle,label = label)
            plt.plot([way[j][0],way[(j+1)%length][0]],[way[j][1],way[(j+1)%length][1]],color = color,linestyle = linestyle)


        
    data_generation.append(fitness[sorted_population[0]])

    unchanged_parents = [population[i] for i in sorted_population] #是上面数组的种群
    children = []
    for _ in range(population_size - selected_size):
        parent1 = random.choice(sorted_population)
        parent2 = random.choice(sorted_population)
        while(parent2 == parent1):
            parent2 = random.choice(sorted_population)
        child = crossover(population[sorted_population[0]],population[sorted_population[1]])
        children.append(mutate(child))
    return unchanged_parents + children

#整合运行程序
def GA_TravelerProblem():
    return 

#轨迹图初始化
def plot_trace():
    plt.figure(figsize=(10,10))
    plt.scatter([point[0] for point in map],[point[1] for point in map],label = 'City')
    plt.title('City Location')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    #下面这部分必须要数据迭代完才能画
    '''
    plt.legend()
    plt.show()
    '''

#迭代图
def plot_iterations():
    plt.figure(figsize=(15,15))
    for i in range(generations-1):
        plt.plot([i+1,i+2],[data_generation[i],data_generation[i+1]],color = 'gray')
    plt.title('iterations')
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.show()

#旅行商问题
#输入量
#total 表示地图大小 ; length 表示城市数量 ; map 表示随机生成的地图网络 ； distance 表示两个城市之间的距离
#population_size 表示种群规模 ; selected_size 表示选择交叉繁衍的父辈规模 ; generations 表示遗传代数 ; mutation_rate = 0.01
run_num = 10

total = 100
length = 20
population_size = 100
selected_size = 20
generations = 500
mutate_rate = 0.01

init_fitness = []
best_fitness = []


#生成地图，和计算各个城市的距离
map = generate_map(length,total)
distance = distance_cal(map)

#表示城市坐标图
plot_trace()

print(map)

data_generation = [] #存储不同迭代次数下的适应度最值

#初始化种群,初始化适应度
population = generate_initial_population()
fitness = fitness_cal()

#种群更新
for i in range(generations):
    population =evolve_population(i)
    fitness = fitness_cal()
#打印最优策略
best_route = population[select_betterparents()[0]]
print(best_route)

plt.legend()
plt.show()

plot_iterations()
        
